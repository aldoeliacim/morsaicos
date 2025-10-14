"""
Generador principal de fotomosaicos de alta fidelidad.
Integra firmas de color perceptuales, control avanzado de repeticiones y
corrección de color por celda para maximizar la calidad visual.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import psutil
import requests
from io import BytesIO

from color_analyzer import ColorAnalyzer


class PhotomosaicGenerator:
    """Generador principal de fotomosaicos con enfoque en calidad."""

    def __init__(self,
                 color_index: Optional[Dict[str, Dict[str, Any]]] = None,
                 base_directory: str = "."):
        self.color_index = color_index or {}
        self.base_directory = Path(base_directory)
        self.mosaic_description: List[List[str]] = []
        self.cell_metadata: Dict[Tuple[int, int], Dict[str, Any]] = {}

        self.cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
        self.num_workers = max(1, min(self.cpu_count, 8))
        self.match_weights: Optional[Dict[str, float]] = None

        self.tile_names: List[str] = []
        self.tile_features: Dict[str, Any] = {}
        self._prepare_feature_matrix()
        self._tile_image_cache: Dict[str, Image.Image] = {}
        self._remote_original_cache: Dict[str, Image.Image] = {}
        self._remote_session = requests.Session()

    # ------------------------------------------------------------------
    # Preparación de datos
    # ------------------------------------------------------------------
    def _prepare_feature_matrix(self) -> None:
        if not self.color_index:
            self.tile_names = []
            self.tile_features = {
                "lab": np.empty((0, 3), dtype=np.float32),
                "luma": np.empty(0, dtype=np.float32),
                "saturation": np.empty(0, dtype=np.float32),
                "texture": np.empty(0, dtype=np.float32),
                "contrast": np.empty(0, dtype=np.float32),
                "signatures": [],
            }
            return

        self.tile_names, self.tile_features = ColorAnalyzer.build_feature_matrix(
            self.color_index, self.base_directory
        )

    # ------------------------------------------------------------------
    # Extracción de firmas de celdas
    # ------------------------------------------------------------------
    def _extract_cell_signatures(self, image_array: np.ndarray, cell_size: int) -> Tuple[Dict[Tuple[int, int], Dict[str, Any]], int, int]:
        height, width = image_array.shape[:2]
        grid_height = height // cell_size
        grid_width = width // cell_size

        cell_signatures: Dict[Tuple[int, int], Dict[str, Any]] = {}

        for row in range(grid_height):
            for col in range(grid_width):
                y0 = row * cell_size
                x0 = col * cell_size
                cell = image_array[y0:y0 + cell_size, x0:x0 + cell_size]
                signature = ColorAnalyzer.compute_signature_from_array(cell)
                cell_signatures[(row, col)] = signature

        return cell_signatures, grid_width, grid_height

    # ------------------------------------------------------------------
    # Selección de candidatos
    # ------------------------------------------------------------------
    def _select_candidate(self,
                          position: Tuple[int, int],
                          candidates: List[Tuple[str, float, Dict[str, Any]]],
                          usage_counts: Dict[str, int],
                          assignments: Dict[Tuple[int, int], Dict[str, Any]],
                          max_repetitions: int,
                          repetition_penalty: float,
                          neighbor_radius: int) -> Dict[str, Any]:
        row, col = position
        best_choice: Optional[Dict[str, Any]] = None
        best_score = float("inf")

        for image_key, base_score, tile_signature in candidates:
            usage = usage_counts.get(image_key, 0)
            if max_repetitions > 0 and usage >= max_repetitions:
                continue

            diversity_penalty = (1.0 + repetition_penalty) ** usage
            neighbor_penalty = self._neighbor_penalty(position, image_key, assignments, neighbor_radius)
            total_score = base_score + diversity_penalty * 0.05 + neighbor_penalty * 0.35

            if total_score < best_score:
                best_score = total_score
                best_choice = {
                    "image_key": image_key,
                    "image_path": image_key,
                    "score": total_score,
                    "base_score": base_score,
                    "tile_signature": tile_signature,
                }

        if best_choice is None and candidates:
            image_key, base_score, tile_signature = candidates[0]
            best_choice = {
                "image_key": image_key,
                "image_path": image_key,
                "score": base_score,
                "base_score": base_score,
                "tile_signature": tile_signature,
            }

        if best_choice is None:
            raise ValueError("No se encontraron candidatos válidos para la celda")

        return best_choice

    def _neighbor_penalty(self,
                          position: Tuple[int, int],
                          image_path: str,
                          assignments: Dict[Tuple[int, int], Dict[str, Any]],
                          radius: int) -> float:
        if radius <= 0:
            return 0.0

        row, col = position
        penalty = 0.0

        for r in range(row - radius, row + radius + 1):
            for c in range(col - radius, col + radius + 1):
                if (r, c) == (row, col):
                    continue
                neighbor = assignments.get((r, c))
                if neighbor and neighbor["image_path"] == image_path:
                    distance = abs(r - row) + abs(c - col)
                    penalty += 1.0 / (distance + 1.0)

        return penalty

    # ------------------------------------------------------------------
    # Generación del mosaico
    # ------------------------------------------------------------------
    def generate_mosaic_description_parallel(self,
                                             image_path: str,
                                             cell_size: int = 20,
                                             output_path: Optional[str] = None,
                                             output_size: Optional[Tuple[int, int]] = None,
                                             repetition_penalty: float = 0.35,
                                             max_repetitions: int = 2,
                                             top_candidates: int = 40,
                                             neighbor_radius: int = 1,
                                             weights: Optional[Dict[str, float]] = None,
                                             progress_callback: Optional[Any] = None,
                                             live_preview_callback: Optional[Any] = None) -> str:
        if not self.tile_names:
            raise ValueError("El índice de colores está vacío. Genera o carga un índice primero.")

        start_time = time.time()

        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            if output_size:
                img = img.resize(output_size, Image.Resampling.LANCZOS)

            if min(img.size) < cell_size:
                raise ValueError("La imagen es más pequeña que el tamaño de celda solicitado")

            target_array = np.asarray(img, dtype=np.float32)

        cell_signatures, grid_width, grid_height = self._extract_cell_signatures(target_array, cell_size)
        total_cells = grid_width * grid_height

        assignments: Dict[Tuple[int, int], Dict[str, Any]] = {}
        usage_counts: Dict[str, int] = {}
        mosaic_description: List[List[str]] = []

        processed = 0
        last_preview_time = 0.0
        for row in range(grid_height):
            row_description: List[str] = []
            for col in range(grid_width):
                signature = cell_signatures[(row, col)]
                candidates = ColorAnalyzer.rank_candidates(
                    signature,
                    self.tile_names,
                    self.tile_features,
                    weights=weights or self.match_weights,
                    top_n=top_candidates
                )

                selection = self._select_candidate(
                    (row, col),
                    candidates,
                    usage_counts,
                    assignments,
                    max_repetitions=max_repetitions,
                    repetition_penalty=repetition_penalty,
                    neighbor_radius=neighbor_radius
                )

                assignments[(row, col)] = {
                    **selection,
                    "target_signature": signature,
                }

                image_key = selection.get("image_key") or selection.get("image_path")
                row_description.append(image_key)
                usage_counts[image_key] = usage_counts.get(image_key, 0) + 1

                processed += 1
                if progress_callback and processed % 50 == 0:
                    progress_callback(processed, total_cells, "Analizando celdas...")

            mosaic_description.append(row_description)

            if live_preview_callback:
                now = time.time()
                if now - last_preview_time >= 0.8:
                    last_preview_time = now
                    try:
                        preview_image = self._assemble_mosaic_fast(
                            assignments,
                            grid_width,
                            grid_height,
                            cell_size,
                            cell_signatures
                        )
                        live_preview_callback(preview_image, processed, total_cells)
                    except Exception:
                        pass

        self.mosaic_description = mosaic_description
        self.cell_metadata = assignments

        mosaic_image = self._assemble_mosaic_fast(
            assignments,
            grid_width,
            grid_height,
            cell_size,
            cell_signatures
        )

        if output_path:
            output_path = str(output_path)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            mosaic_image.save(output_path, "JPEG", quality=96, optimize=True)

        elapsed = time.time() - start_time

        repetition_values = list(usage_counts.values())
        unique_images = len(usage_counts)
        max_usage = max(repetition_values) if repetition_values else 0
        average_usage = np.mean(repetition_values) if repetition_values else 0

        description = (
            "\n".join([
                "Fotomosaico completado",
                f"Tiempo total: {elapsed:.2f}s",
                f"Celdas: {grid_width}×{grid_height} ({total_cells:,})",
                f"Imágenes únicas: {unique_images:,}",
                f"Uso máximo por imagen: {max_usage}",
                f"Uso promedio por imagen: {average_usage:.2f}",
            f"Top candidatos evaluados: {top_candidates}",
        ])
        )

        if live_preview_callback:
            try:
                live_preview_callback(mosaic_image.copy(), total_cells, total_cells)
            except Exception:
                pass

        return description

    # Compatibilidad con API original
    def generate_mosaic_description(self, source_image_path: str,
                                    cell_size: int = 20,
                                    output_path: Optional[str] = None,
                                    **kwargs) -> List[List[str]]:
        self.generate_mosaic_description_parallel(
            source_image_path,
            cell_size=cell_size,
            output_path=output_path,
            **kwargs
        )
        return self.mosaic_description

    # ------------------------------------------------------------------
    # Ensamblaje y corrección de color
    # ------------------------------------------------------------------
    def _assemble_mosaic_fast(self,
                              assignments: Dict[Tuple[int, int], Dict[str, Any]],
                              grid_width: int,
                              grid_height: int,
                              cell_size: int,
                              cell_signatures: Dict[Tuple[int, int], Dict[str, Any]]) -> Image.Image:
        mosaic_width = grid_width * cell_size
        mosaic_height = grid_height * cell_size
        mosaic = Image.new('RGB', (mosaic_width, mosaic_height))

        for (row, col), info in assignments.items():
            image_key = info.get("image_key") or info.get("image_path")
            base_tile = self._load_tile_image(image_key, info["tile_signature"], cell_size)
            corrected = self._apply_tile_correction(
                base_tile.copy(),
                info["tile_signature"],
                cell_signatures[(row, col)]
            )

            x = col * cell_size
            y = row * cell_size
            mosaic.paste(corrected, (x, y))

        mosaic = mosaic.filter(ImageFilter.GaussianBlur(radius=0.35))
        mosaic = ImageEnhance.Contrast(mosaic).enhance(1.03)
        return mosaic

    def _load_tile_image(self, image_key: str, tile_signature: Dict[str, Any], cell_size: int) -> Image.Image:
        if not image_key:
            raise ValueError("Clave de imagen no válida para ensamblar el mosaico")
        cache_key = f"{image_key}_{cell_size}"
        cached = self._tile_image_cache.get(cache_key)
        if cached is not None:
            return cached

        base_tile = self._fetch_base_tile(image_key, tile_signature)
        tile_copy = base_tile.copy() if base_tile is not None else Image.new('RGB', (cell_size, cell_size), (128, 128, 128))
        if tile_copy.mode != 'RGB':
            tile_copy = tile_copy.convert('RGB')
        resized = tile_copy.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
        self._tile_image_cache[cache_key] = resized
        return resized

    def _fetch_base_tile(self, image_key: str, tile_signature: Dict[str, Any]) -> Image.Image:
        url = tile_signature.get('url') or image_key
        source_path = tile_signature.get('source_path')

        if url and str(url).lower().startswith(('http://', 'https://')):
            return self._load_remote_image(url)

        if source_path:
            path = Path(source_path)
            if not path.is_absolute():
                path = (self.base_directory / path).resolve()
        else:
            path = (self.base_directory / image_key).resolve()

        with Image.open(path) as img:
            tile = img.convert('RGB')
        return tile

    def _load_remote_image(self, url: str) -> Image.Image:
        cache = getattr(self, '_remote_original_cache', None)
        if cache is None:
            cache = {}
            self._remote_original_cache = cache
        cached = cache.get(url)
        if cached is not None:
            return cached

        response = self._remote_session.get(url, timeout=30)
        response.raise_for_status()
        tile = Image.open(BytesIO(response.content)).convert('RGB')
        cache[url] = tile
        return tile

    def _apply_tile_correction(self,
                               tile_image: Image.Image,
                               tile_signature: Dict[str, Any],
                               target_signature: Dict[str, Any]) -> Image.Image:
        corrected = tile_image

        tile_luma = tile_signature.get("luma", target_signature.get("luma", 0.0))
        target_luma = target_signature.get("luma", tile_luma)
        if tile_luma > 0:
            brightness_factor = np.clip((target_luma + 1e-3) / (tile_luma + 1e-3), 0.6, 1.6)
            if abs(brightness_factor - 1.0) > 0.05:
                corrected = ImageEnhance.Brightness(corrected).enhance(float(brightness_factor))

        tile_sat = tile_signature.get("saturation", 0.0)
        target_sat = target_signature.get("saturation", tile_sat)
        if tile_sat > 0:
            sat_factor = np.clip((target_sat + 1e-3) / (tile_sat + 1e-3), 0.5, 1.6)
        else:
            sat_factor = np.clip(1.0 + (target_sat - tile_sat), 0.5, 1.6)
        if abs(sat_factor - 1.0) > 0.05:
            corrected = ImageEnhance.Color(corrected).enhance(float(sat_factor))

        tile_contrast = tile_signature.get("contrast", 0.0)
        target_contrast = target_signature.get("contrast", tile_contrast)
        if tile_contrast > 0:
            contrast_factor = np.clip((target_contrast + 1e-3) / (tile_contrast + 1e-3), 0.7, 1.4)
            if abs(contrast_factor - 1.0) > 0.05:
                corrected = ImageEnhance.Contrast(corrected).enhance(float(contrast_factor))

        return corrected

    # ------------------------------------------------------------------
    # Guardado / carga de descripciones
    # ------------------------------------------------------------------
    def save_mosaic_description(self, output_path: str,
                                metadata: Optional[Dict[str, Any]] = None) -> None:
        import json

        if not self.mosaic_description:
            raise ValueError("No hay mosaico generado para guardar")

        data = {
            "mosaic_description": self.mosaic_description,
            "metadata": metadata or {},
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_mosaic_description(self, description_path: str) -> List[List[str]]:
        import json

        with open(description_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.mosaic_description = data.get("mosaic_description", [])
        return self.mosaic_description

    def get_mosaic_stats(self) -> Dict[str, Any]:
        if not self.mosaic_description:
            return {}

        total_cells = sum(len(row) for row in self.mosaic_description)
        unique_images = len({image for row in self.mosaic_description for image in row})

        return {
            "total_cells": total_cells,
            "unique_images": unique_images,
            "grid_size": (len(self.mosaic_description), len(self.mosaic_description[0] if self.mosaic_description else [])),
        }

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    @classmethod
    def from_index_file(cls, index_path: str, base_directory: str,
                        format: str = "pickle") -> "PhotomosaicGenerator":
        import json
        import pickle

        index_path = Path(index_path)
        if format == "json":
            with open(index_path, 'r', encoding='utf-8') as f:
                color_index = json.load(f)
        elif format == "pickle":
            with open(index_path, 'rb') as f:
                color_index = pickle.load(f)
        else:
            raise ValueError("Formato no soportado para cargar el índice")

        return cls(color_index=color_index, base_directory=base_directory)

    def get_performance_stats(self) -> Dict[str, Any]:
        return {
            "cpu_count": psutil.cpu_count(),
            "num_workers": self.num_workers,
            "tile_count": len(self.tile_names),
        }


__all__ = ["PhotomosaicGenerator"]
