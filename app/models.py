"""
Esquemas Pydantic que sustentan la API de fotomosaicos.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, validator


class MosaicJobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class MosaicParameters(BaseModel):
    """
    Parámetros ajustables por la persona usuaria del generador de fotomosaicos.

    Los valores predeterminados equilibran la calidad visual con un tiempo de
    procesamiento razonable en ejecuciones dependientes de CPU.
    """

    cell_size: int = Field(32, ge=8, le=128, description="Longitud del lado en píxeles de cada celda.")
    top_candidates: int = Field(
        35,
        ge=5,
        le=128,
        description="Número de teselas candidatas evaluadas por celda antes de elegir.",
    )
    max_repeated_tiles: int = Field(
        8,
        ge=0,
        le=250,
        description="Máximo de veces que se puede reutilizar una tesela. Cero desactiva el límite.",
    )
    repetition_penalty: float = Field(
        0.35,
        ge=0.0,
        le=2.0,
        description="Penalización acumulativa aplicada cuando se repite la misma tesela.",
    )
    neighbor_radius: int = Field(
        1,
        ge=0,
        le=5,
        description="Radio (en celdas) para evitar vecinos con la misma tesela.",
    )
    blend_output: bool = Field(
        True,
        description="(Obsoleto) Indicador legado para la salida mezclada; ahora se controla desde la vista.",
    )
    grayscale_output: bool = Field(
        True,
        description="(Obsoleto) Indicador legado para la salida en escala de grises; hoy se activa desde la vista.",
    )
    blend_ratio: float = Field(  # Se mantiene por compatibilidad; solo se usa si blend_output es True.
        0.28,
        ge=0.0,
        le=0.9,
        description="Porcentaje de mezcla del mosaico final con la imagen original.",
    )
    enable_enhancement: bool = Field(
        True,
        description="Indica si se aplica un realce leve de contraste y nitidez al mosaico mezclado.",
    )
    distance_metric: str = Field(
        "hybrid",
        description="Métrica usada para comparar colores (ciede2000, riemersma, hybrid).",
    )

    @validator("top_candidates")
    def _ensure_reasonable_top(cls, value: int) -> int:  # noqa: N805
        if value % 5 != 0 and value >= 10:
            # Normaliza a múltiplos de 5 para ofrecer incrementos más cómodos.
            return min(128, max(5, round(value / 5) * 5))
        return value

    @validator("distance_metric")
    def _validate_metric(cls, value: str) -> str:  # noqa: N805
        allowed = {"ciede2000", "riemersma", "hybrid"}
        lowered = (value or "hybrid").lower()
        if lowered not in allowed:
            raise ValueError(f"distance_metric debe ser una de {sorted(allowed)}")
        return lowered


class MosaicJobRequest(BaseModel):
    """Carga útil que acompaña al archivo de imagen para solicitar un nuevo mosaico."""

    parameters: MosaicParameters = Field(default_factory=MosaicParameters)


class MosaicJobProgress(BaseModel):
    """Estructura con el avance que la API devuelve."""

    total: Optional[int] = None
    completed: int = 0
    percent: float = 0.0
    message: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    eta_seconds: Optional[float] = None
    estimated_total_seconds: Optional[float] = None


class MosaicResultPaths(BaseModel):
    """Rutas relativas de los artefactos generados."""

    mosaic: Optional[str] = None
    blended: Optional[str] = None
    preview: Optional[str] = None
    grayscale: Optional[str] = None
    report: Optional[str] = None


class MosaicJobResponse(BaseModel):
    """Representación detallada de un trabajo que recibe la interfaz."""

    id: str
    status: MosaicJobStatus
    created_at: datetime
    updated_at: datetime
    finished_at: Optional[datetime] = None
    message: Optional[str] = None
    parameters: MosaicParameters
    progress: MosaicJobProgress = Field(default_factory=MosaicJobProgress)
    result: MosaicResultPaths = Field(default_factory=MosaicResultPaths)
    metrics: Optional[dict] = None
    preview_url: Optional[str] = None
    timing: Optional[dict] = None


class MosaicJobCreated(BaseModel):
    """Respuesta devuelta cuando un nuevo trabajo fue aceptado."""

    id: str
    status: MosaicJobStatus
    parameters: MosaicParameters


class ApiError(BaseModel):
    """Contenedor estándar para respuestas de error."""

    detail: str


__all__ = [
    "ApiError",
    "MosaicJobCreated",
    "MosaicJobProgress",
    "MosaicJobRequest",
    "MosaicJobResponse",
    "MosaicJobStatus",
    "MosaicParameters",
    "MosaicResultPaths",
]
