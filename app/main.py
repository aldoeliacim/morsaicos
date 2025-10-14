"""
Fábrica de aplicaciones FastAPI para el estudio web de fotomosaicos renovado.
El proyecto conserva las raíces académicas de *Un estudio empírico de los
fotomosaicos* presentado por la UNAM, incorporando sus métricas al flujo
de procesamiento.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from app.logging_config import get_logger
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.routes import get_api_router
from app.services import ColorIndexLoader, JobStore, MosaicPipeline


def create_app() -> FastAPI:
    logger = get_logger("morsaicos.main")
    logger.info("Inicializando aplicacion Morsaicos Studio")
    app = FastAPI(
        title="Morsaicos Studio",
        description="Generador de fotomosaicos con una experiencia web renovada.",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    job_store = JobStore(retention_limit=settings.job_retention)
    color_index = ColorIndexLoader(settings.color_index_path)
    pipeline = MosaicPipeline(job_store=job_store, color_index=color_index)

    app.state.job_store = job_store
    app.state.color_index = color_index
    app.state.pipeline = pipeline

    app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
    app.mount("/outputs", StaticFiles(directory=str(settings.outputs_dir)), name="outputs")
    app.mount("/temp", StaticFiles(directory=str(settings.temp_dir)), name="temp")

    templates = Jinja2Templates(directory=str(settings.templates_dir))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "poll_interval": settings.polling_interval_seconds,
            },
        )

    app.include_router(get_api_router())

    @app.get("/api/diagnostics")
    async def diagnostics() -> dict:
        count, index_path = color_index.info()
        return {
            "color_index_items": count,
            "color_index_path": str(index_path) if index_path else None,
            "outputs_dir": str(settings.outputs_dir),
            "poll_interval": settings.polling_interval_seconds,
        }

    return app


app = create_app()


__all__ = ["app", "create_app"]
