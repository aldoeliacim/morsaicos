"""
Utilidades para cargar el índice de color.

El generador de fotomosaicos requiere un diccionario con descriptores por imagen.
Cargar esta estructura puede tomar tiempo, por lo que este módulo implementa un
cargador con caché y recarga segura entre hilos.
"""

from __future__ import annotations

import json
import pickle
import threading
from pathlib import Path
from typing import Dict, Iterable, Tuple


class ColorIndexLoader:
    """Cargador diferido del índice de color precalculado."""

    def __init__(self, index_path: Path | None) -> None:
        self._index_path = index_path
        self._lock = threading.Lock()
        self._cache: Dict[str, dict] | None = None

    @property
    def path(self) -> Path | None:
        return self._index_path

    def ensure_available(self) -> None:
        if self._index_path is None or not self._index_path.exists():
            raise FileNotFoundError(
                "No se encontró el índice de color. Ejecuta `generate_color_index.py` "
                "para crearlo antes de usar la aplicación web."
            )

    def load(self) -> Dict[str, dict]:
        """Devuelve el índice completo; lo carga una vez y guarda el resultado en caché."""
        cache = self._cache
        if cache is not None:
            return cache

        with self._lock:
            if self._cache is not None:
                return self._cache
            self.ensure_available()
            assert self._index_path is not None  # for type-checkers
            suffix = self._index_path.suffix.lower()
            if suffix in {".pkl", ".pickle"}:
                with self._index_path.open("rb") as stream:
                    data = pickle.load(stream)
            elif suffix in {".json"}:
                with self._index_path.open("r", encoding="utf-8") as stream:
                    data = json.load(stream)
            elif suffix in {".txt"}:
                # La exportación en texto es un respaldo: se espera un objeto JSON por línea.
                records: Dict[str, dict] = {}
                with self._index_path.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        key = entry.get("filename") or entry.get("image") or str(len(records))
                        records[key] = entry
                data = records
            else:
                raise ValueError(f"Formato no soportado para el índice: {self._index_path}")

            if not isinstance(data, dict):
                raise TypeError("El índice de color debe ser un diccionario {filename: metadata}")

            self._cache = data
            return data

    def reload(self) -> Dict[str, dict]:
        """Fuerza la recarga del índice desde disco."""
        with self._lock:
            self._cache = None
        return self.load()

    def sample(self, limit: int) -> Dict[str, dict]:
        """
        Devuelve una muestra determinista del índice, útil para previsualizaciones ligeras.

        El muestreo conserva el orden por nombre de archivo para mantener resultados estables.
        """
        data = self.load()
        if limit <= 0 or limit >= len(data):
            return data

        keys: Iterable[str] = sorted(data.keys())
        selected = {}
        step = max(len(data) // limit, 1)
        for key in keys[::step]:
            selected[key] = data[key]
            if len(selected) >= limit:
                break
        return selected

    def info(self) -> Tuple[int, Path | None]:
        """Pequeño asistente utilizado por la API para reportar diagnósticos."""
        cache = self._cache
        return (len(cache) if cache is not None else 0, self._index_path)


__all__ = ["ColorIndexLoader"]
