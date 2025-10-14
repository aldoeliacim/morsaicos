"""
Project-wide filesystem helpers for the Morsaicos Studio project.

Provides absolute paths for common directories so that code does not rely on
the current working directory (which varies between CLI, server reloads, tests,
or IDE tasks). Importing this module guarantees that the expected writable
folders exist before they are used elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TEMP_DIR = PROJECT_ROOT / "temp"


def ensure_directories(directories: Iterable[Path]) -> None:
    """Create the given directories (and parents) if they do not exist."""
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Ensure writable directories are present once the module is imported.
ensure_directories((UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR))

