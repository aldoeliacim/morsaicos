"""
Application package for the redesigned photomosaic service.

The package exposes a `create_app` factory (see `app.main`) that is used by
`web_app.py` to stay backwards compatible with existing entry points.
"""

from .main import create_app

__all__ = ["create_app"]
