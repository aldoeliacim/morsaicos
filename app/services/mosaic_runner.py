"""
Canalización de ejecución en segundo plano para trabajos de fotomosaico.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional
import time

from PIL import Image, ImageOps

from app.config import settings
from app.logging_config import get_logger
from app.models import MosaicJobStatus, MosaicParameters
from color_analyzer import ColorAnalyzer  # noqa: F401  (ensures heavy deps preloaded)
from mosaic_enhancer import MosaicEnhancer
from photomosaic_generator import PhotomosaicGenerator

from .color_index import ColorIndexLoader
from .jobs import JobStore, MosaicJob


LOGGER = get_logger("morsaicos.pipeline")


class MosaicPipeline:
    """Coordina el envío y la ejecución de los trabajos."""

    def __init__(self, job_store: JobStore, color_index: ColorIndexLoader) -> None:
        self._store = job_store
        self._color_index = color_index
        self._executor = ThreadPoolExecutor(max_workers=settings.max_background_workers, thread_name_prefix="mosaic")

    def submit(self, job: MosaicJob) -> None:
        """Ejecuta el trabajo en segundo plano."""
        self._executor.submit(self._run_job, job.id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_job(self, job_id: str) -> None:
        job = self._store.get(job_id)
        if not job:
            return

        try:
            LOGGER.info("Iniciando generacion para job %s", job_id)
            self._store.mark_running(job_id, "Preparando indice de color...")
            color_index = self._color_index.load()

            generator = PhotomosaicGenerator(color_index=color_index, base_directory=str(settings.project_root))
            job.parameters = MosaicParameters.parse_obj(job.parameters.dict())
            generator.match_weights = job.parameters.distance_metric

            preview_path = settings.temp_dir / f"{job.id}_preview.jpg"
            last_preview = {"time": 0.0}

            def progress_callback(processed: int, total: int, message: str) -> None:
                self._store.update_progress(job_id, processed, total, message)

            def live_preview_callback(image: Image.Image, processed: int, total: int) -> None:
                now = time.perf_counter()
                if now - last_preview["time"] < 0.75:
                    return
                last_preview["time"] = now
                try:
                    preview_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(preview_path, "JPEG", quality=80, optimize=True)
                    self._store.update_preview(job_id, preview_path)
                except Exception:
                    # Preview updates are best-effort; ignore errors.
                    pass

            self._store.update_progress(job_id, 0, None, "Analizando celdas...")
            self._store.update_preview(job_id, None)

            overall_start = time.perf_counter()

            selection_start = time.perf_counter()
            summary_text = generator.generate_mosaic_description_parallel(
                image_path=str(job.source_path),
                cell_size=job.parameters.cell_size,
                output_path=str(job.mosaic_path),
                repetition_penalty=job.parameters.repetition_penalty,
                max_repetitions=job.parameters.max_repeated_tiles,
                top_candidates=job.parameters.top_candidates,
                neighbor_radius=job.parameters.neighbor_radius,
                progress_callback=progress_callback,
                live_preview_callback=live_preview_callback,
            )
            selection_duration = time.perf_counter() - selection_start

            stats = generator.get_mosaic_stats()
            total_cells = stats.get("total_cells", 0)
            self._store.update_progress(job_id, total_cells, total_cells, "Ensamblando mosaico...")

            post_process_start = time.perf_counter()
            blended_created, grayscale_created = self._build_artifacts(job)
            artifacts_duration = time.perf_counter() - post_process_start

            metrics_start = time.perf_counter()
            metrics = self._generate_metrics(job, generator, stats, summary_text)
            metrics_duration = time.perf_counter() - metrics_start

            self._store.update_progress(job_id, total_cells, total_cells, "Finalizando...")

            total_duration = time.perf_counter() - overall_start
            timing = {
                "total_seconds": round(total_duration, 3),
                "selection_seconds": round(selection_duration, 3),
                "artifact_seconds": round(artifacts_duration, 3),
                "metrics_seconds": round(metrics_duration, 3),
                "eta_seconds": 0.0,
            }
            if total_cells:
                timing["cells_per_second"] = round(total_cells / total_duration, 2) if total_duration > 0 else None
                timing["milliseconds_per_cell"] = round((total_duration / total_cells) * 1000, 3)
            metrics["timing"] = timing

            self._store.mark_completed(
                job_id,
                message="Mosaico generado con exito",
                metrics=metrics,
                timing=timing,
            )
            LOGGER.info("Job %s completado en %.2fs", job_id, timing.get("total_seconds", 0.0))

            if not blended_created:
                # Remove blended path reference if we didn't produce the file.
                job.blended_path = None

            # Point preview to the final mosaic for post-run inspection.
            if job.mosaic_path.exists():
                self._store.update_preview(job_id, job.mosaic_path)

        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception("Fallo job %s: %s", job_id, exc)
            self._store.mark_failed(job_id, f"Error durante la generacion: {exc}")

    def _build_artifacts(self, job: MosaicJob) -> tuple[bool, bool]:
        """
        Construye los artefactos secundarios (imagen mezclada y versión en escala de grises).

        Devuelve una tupla que indica si se generaron los archivos de mezcla y escala de grises.
        """
        parameters = job.parameters
        blended_created = False
        grayscale_created = False

        if job.grayscale_path and getattr(parameters, "grayscale_output", True):
            try:
                with Image.open(job.mosaic_path) as mosaic_img:
                    grayscale = ImageOps.grayscale(mosaic_img).convert("RGB")
                    grayscale.save(job.grayscale_path, "JPEG", quality=96, optimize=True)
                grayscale_created = True
            except Exception:
                job.grayscale_path = None
        else:
            job.grayscale_path = None

        if (
            job.blended_path
            and getattr(parameters, "blend_output", True)
            and getattr(parameters, "blend_ratio", 0) > 0
        ):
            with Image.open(job.mosaic_path) as mosaic_img, Image.open(job.source_path) as original_img:
                blended = MosaicEnhancer.apply_blending(
                    mosaic_image=mosaic_img,
                    original_image=original_img,
                    blend_percentage=float(parameters.blend_ratio),
                    detail_strength=0.3,
                )

                if parameters.enable_enhancement:
                    blended = MosaicEnhancer.enhance_mosaic_with_filters(
                        blended,
                        enhance_contrast=1.05,
                        enhance_sharpness=1.15,
                        enhance_color=1.03,
                    )

                blended.save(job.blended_path, "JPEG", quality=96, optimize=True)
                blended_created = True
        else:
            job.blended_path = None

        return blended_created, grayscale_created

    def _generate_metrics(
        self,
        job: MosaicJob,
        generator: PhotomosaicGenerator,
        stats: dict,
        summary_text: str,
    ) -> dict:
        """
        Calcula las métricas de calidad y guarda un reporte JSON ligero.
        """
        mosaic_description = generator.mosaic_description
        metrics = MosaicEnhancer.calculate_quality_metrics(mosaic_description, str(job.source_path), str(job.mosaic_path))
        metrics.update(
            {
                "grid_size": stats.get("grid_size"),
                "run_summary": summary_text,
                "blend_output": bool(job.blended_path),
                "blend_ratio": getattr(job.parameters, "blend_ratio", 0.0),
                "grayscale_output": bool(job.grayscale_path),
                "distance_metric": getattr(job.parameters, "distance_metric", "hybrid"),
            }
        )

        report_path = job.report_path
        if report_path:
            payload = {
                "job_id": job.id,
                "status": MosaicJobStatus.completed,
                "parameters": job.parameters.dict(),
                "metrics": metrics,
                "mosaic_description": mosaic_description,
            }
            with report_path.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, ensure_ascii=False)

        return metrics


__all__ = ["MosaicPipeline"]
