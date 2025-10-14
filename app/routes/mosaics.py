"""
Rutas de la API para la generación de fotomosaicos.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from app.config import settings
from app.models import (
    ApiError,
    MosaicJobCreated,
    MosaicJobRequest,
    MosaicJobResponse,
    MosaicJobStatus,
    MosaicParameters,
)
from app.logging_config import get_logger
from app.services import ColorIndexLoader, JobStore, MosaicPipeline

LOGGER = get_logger("morsaicos.api")
router = APIRouter(prefix="/api/mosaics", tags=["mosaics"])


# ---------------------------------------------------------------------------
# Ayudantes de dependencias
# ---------------------------------------------------------------------------
def _job_store(request: Request) -> JobStore:
    return request.app.state.job_store


def _pipeline(request: Request) -> MosaicPipeline:
    return request.app.state.pipeline


def _color_index(request: Request) -> ColorIndexLoader:
    return request.app.state.color_index


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
@router.get("/defaults", response_model=MosaicParameters)
async def get_default_parameters() -> MosaicParameters:
    """Expone los parámetros predeterminados para inicializar la interfaz."""
    return MosaicParameters()


@router.get("", response_model=list[MosaicJobResponse])
async def list_jobs(store: JobStore = Depends(_job_store)) -> list[MosaicJobResponse]:
    """Lista los trabajos del más reciente al más antiguo."""
    return [job.to_response() for job in store.list()]


@router.get("/{job_id}", response_model=MosaicJobResponse, responses={404: {"model": ApiError}})
async def get_job(job_id: str, store: JobStore = Depends(_job_store)) -> MosaicJobResponse:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job no encontrado")
    return job.to_response()


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=MosaicJobCreated,
    responses={400: {"model": ApiError}, 415: {"model": ApiError}, 503: {"model": ApiError}},
)
async def create_job(
    request_payload: str = Form(""),
    source_image: UploadFile = File(...),
    store: JobStore = Depends(_job_store),
    pipeline: MosaicPipeline = Depends(_pipeline),
    color_index: ColorIndexLoader = Depends(_color_index),
) -> MosaicJobCreated:
    """Recibe una nueva solicitud de fotomosaico."""
    try:
        payload_dict = json.loads(request_payload) if request_payload else {}
    except json.JSONDecodeError as exc:
        LOGGER.warning("Payload invalido: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"JSON invalido: {exc}")

    try:
        job_request = MosaicJobRequest(**payload_dict)
    except Exception as exc:
        LOGGER.warning("Parametros invalidos: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Parametros invalidos: {exc}")

    if not source_image.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Se requiere una imagen de entrada")

    extension = (Path(source_image.filename).suffix or "").lower()
    if extension not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Formato no soportado. Usa JPG o PNG.",
        )

    try:
        color_index.ensure_available()
    except FileNotFoundError as exc:
        LOGGER.error("Indice de color no disponible: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    job_id = uuid.uuid4().hex
    source_path = settings.uploads_dir / f"{job_id}_source{extension}"
    mosaic_path = settings.outputs_dir / f"{job_id}_mosaic.jpg"
    blended_path = settings.outputs_dir / f"{job_id}_blended.jpg"
    grayscale_path = settings.outputs_dir / f"{job_id}_grayscale.jpg"
    report_path = settings.outputs_dir / f"{job_id}_report.json"

    _save_upload_to_disk(source_image, source_path)

    LOGGER.info("Nuevo job %s recibido", job_id)
    job = store.create(
        parameters=job_request.parameters,
        source_path=source_path,
        mosaic_path=mosaic_path,
        blended_path=blended_path,
        grayscale_path=grayscale_path,
        report_path=report_path,
        job_id=job_id,
    )

    pipeline.submit(job)
    LOGGER.info("Job %s encolado", job_id)

    return MosaicJobCreated(id=job.id, status=MosaicJobStatus.pending, parameters=job.parameters)


def _save_upload_to_disk(upload: UploadFile, destination: Path) -> None:
    """Copia un UploadFile a disco sin cargarlo por completo en memoria."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    upload.file.seek(0)
    with destination.open("wb") as target:
        shutil.copyfileobj(upload.file, target)
    upload.file.close()
