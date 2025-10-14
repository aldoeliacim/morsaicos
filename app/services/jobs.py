"""
Registro de trabajos seguro para tareas de generación de fotomosaicos.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

from app.config import settings

from app.models import (
    MosaicJobProgress,
    MosaicJobResponse,
    MosaicJobStatus,
    MosaicParameters,
    MosaicResultPaths,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class MosaicJob:
    """Representación interna de un trabajo en ejecución o finalizado."""

    id: str
    parameters: MosaicParameters
    source_path: Path
    mosaic_path: Path
    blended_path: Optional[Path] = None
    grayscale_path: Optional[Path] = None
    report_path: Optional[Path] = None
    status: MosaicJobStatus = MosaicJobStatus.pending
    message: Optional[str] = None
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    finished_at: Optional[datetime] = None
    _total_cells: Optional[int] = None
    _cells_done: int = 0
    metrics: Optional[dict] = None
    preview_path: Optional[Path] = None
    timing: Optional[dict] = None
    _elapsed_seconds: Optional[float] = None
    _eta_seconds: Optional[float] = None
    _estimated_total_seconds: Optional[float] = None

    def set_status(self, status: MosaicJobStatus, message: Optional[str] = None) -> None:
        self.status = status
        self.message = message
        self.updated_at = _utcnow()
        if status in (MosaicJobStatus.completed, MosaicJobStatus.failed):
            self.finished_at = self.updated_at
            self._eta_seconds = 0.0
            if self._elapsed_seconds is None and self.finished_at:
                self._elapsed_seconds = (self.finished_at - self.created_at).total_seconds()
            self._estimated_total_seconds = self._elapsed_seconds

    def set_progress(self, processed: int, total: Optional[int], message: Optional[str]) -> None:
        self._cells_done = processed
        if total is not None and total > 0:
            self._total_cells = total
        self.message = message or self.message
        now = _utcnow()
        self.updated_at = now

        elapsed = (now - self.created_at).total_seconds()
        self._elapsed_seconds = max(elapsed, 0.0)
        self._eta_seconds = None
        self._estimated_total_seconds = None
        if self._total_cells and self._cells_done > 0:
            ratio = self._cells_done / self._total_cells
            if ratio > 0:
                estimated_total = self._elapsed_seconds / ratio
                self._estimated_total_seconds = estimated_total
                self._eta_seconds = max(estimated_total - self._elapsed_seconds, 0.0)

    def percent(self) -> float:
        if not self._total_cells:
            return 0.0
        return min(100.0, (self._cells_done / self._total_cells) * 100.0)

    def to_response(self) -> MosaicJobResponse:
        progress = MosaicJobProgress(
            total=self._total_cells,
            completed=self._cells_done,
            percent=round(self.percent(), 2),
            message=self.message,
            elapsed_seconds=self._elapsed_seconds,
            eta_seconds=self._eta_seconds,
            estimated_total_seconds=self._estimated_total_seconds,
        )

        def _relative_url(path: Optional[Path]) -> Optional[str]:
            if path is None:
                return None
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            if not resolved.exists():
                return None
            outputs_root = settings.outputs_dir.resolve()
            temp_root = settings.temp_dir.resolve()
            try:
                if resolved.is_relative_to(outputs_root):
                    return f"/outputs/{resolved.name}"
                if resolved.is_relative_to(temp_root):
                    return f"/temp/{resolved.name}"
            except AttributeError:
                if str(resolved).startswith(str(outputs_root)):
                    return f"/outputs/{resolved.name}"
                if str(resolved).startswith(str(temp_root)):
                    return f"/temp/{resolved.name}"
            return f"/outputs/{resolved.name}"

        result_paths = MosaicResultPaths(
            mosaic=_relative_url(self.mosaic_path),
            blended=_relative_url(self.blended_path),
            grayscale=_relative_url(self.grayscale_path),
            report=_relative_url(self.report_path),
            preview=_relative_url(self.preview_path),
        )

        return MosaicJobResponse(
            id=self.id,
            status=self.status,
            created_at=self.created_at,
            updated_at=self.updated_at,
            finished_at=self.finished_at,
            message=self.message,
            parameters=self.parameters,
            progress=progress,
            result=result_paths,
            metrics=self.metrics,
            preview_url=result_paths.preview,
            timing=self.timing,
        )


class JobStore:
    """Registro en memoria de los trabajos de mosaico."""

    def __init__(self, retention_limit: int = 20) -> None:
        self._jobs: Dict[str, MosaicJob] = {}
        self._lock = threading.Lock()
        self._retention = max(1, retention_limit)

    def create(
        self,
        parameters: MosaicParameters,
        source_path: Path,
        mosaic_path: Path,
        blended_path: Optional[Path],
        grayscale_path: Optional[Path],
        report_path: Optional[Path],
        job_id: Optional[str] = None,
    ) -> MosaicJob:
        with self._lock:
            job_id = job_id or uuid.uuid4().hex
            job = MosaicJob(
                id=job_id,
                parameters=parameters,
                source_path=source_path,
                mosaic_path=mosaic_path,
                blended_path=blended_path,
                grayscale_path=grayscale_path,
                report_path=report_path,
            )
            self._jobs[job_id] = job
            self._evict_if_needed()
            return job

    def _evict_if_needed(self) -> None:
        if len(self._jobs) <= self._retention:
            return
        # Elimina primero los trabajos completados más antiguos para acotar la memoria.
        completed: Iterable[tuple[str, MosaicJob]] = sorted(
            ((job_id, job) for job_id, job in self._jobs.items() if job.finished_at),
            key=lambda item: item[1].finished_at or item[1].created_at,
        )
        for job_id, _ in completed:
            del self._jobs[job_id]
            if len(self._jobs) <= self._retention:
                return

    def get(self, job_id: str) -> Optional[MosaicJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> Iterable[MosaicJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def update_progress(
        self,
        job_id: str,
        processed: int,
        total: Optional[int],
        message: Optional[str],
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.set_progress(processed, total, message)

    def mark_completed(
        self,
        job_id: str,
        message: Optional[str] = None,
        metrics: Optional[dict] = None,
        timing: Optional[dict] = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.metrics = metrics
                job.timing = timing
                job.set_status(MosaicJobStatus.completed, message)

    def mark_failed(self, job_id: str, error_message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.set_status(MosaicJobStatus.failed, error_message)

    def mark_running(self, job_id: str, message: Optional[str] = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.set_status(MosaicJobStatus.running, message)

    def update_preview(self, job_id: str, preview_path: Optional[Path]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.preview_path = preview_path
                job.updated_at = _utcnow()


__all__ = ["JobStore", "MosaicJob"]
