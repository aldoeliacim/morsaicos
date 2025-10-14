"""Top-level router assembly."""

from fastapi import APIRouter

from .mosaics import router as mosaics_router


def get_api_router() -> APIRouter:
    api = APIRouter()
    api.include_router(mosaics_router)
    return api


__all__ = ["get_api_router"]
