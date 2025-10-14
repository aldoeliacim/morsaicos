"""
Utilidades de configuración para la aplicación web de fotomosaicos.

Centraliza rutas de archivos y parámetros de ejecución para que la API y los
procesos en segundo plano compartan una sola fuente de verdad.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from paths import OUTPUT_DIR, PROJECT_ROOT, STATIC_DIR, TEMP_DIR, TEMPLATES_DIR, UPLOAD_DIR


def _detect_color_index() -> Optional[Path]:
    """Devuelve el primer archivo de índice de color disponible, si existe."""
    candidates = (
        PROJECT_ROOT / "color_index.pkl",
        PROJECT_ROOT / "color_index.json",
        PROJECT_ROOT / "color_index.txt",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@dataclass(slots=True)
class Settings:
    """Contenedor de parámetros de ejecución."""

    project_root: Path = PROJECT_ROOT
    static_dir: Path = STATIC_DIR
    templates_dir: Path = TEMPLATES_DIR
    uploads_dir: Path = UPLOAD_DIR
    outputs_dir: Path = OUTPUT_DIR
    temp_dir: Path = TEMP_DIR
    color_index_path: Optional[Path] = field(default_factory=_detect_color_index)
    max_background_workers: int = 2
    job_retention: int = 20  # Número de trabajos completados que permanecen en memoria.
    polling_interval_seconds: float = 1.0  # Intervalo de sondeo predeterminado para la interfaz.

    def ensure_directories(self) -> None:
        """Crea los directorios grabables si no existen."""
        for directory in (self.uploads_dir, self.outputs_dir, self.temp_dir):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()

__all__ = ["settings", "Settings"]
