#!/usr/bin/env python3
"""
Lanzador del estudio web de Morsaicos Studio.

Ofrece un asistente ligero que verifica dependencias y (opcionalmente) recuerda
generar el índice de color antes de iniciar la aplicación FastAPI. Usa este
script (`python start_webapp.py`) como punto de entrada oficial en lugar de
invocar `uvicorn` de manera directa; quienes requieran mayor control pueden
añadir los parámetros de host y puerto manualmente.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


def check_requirements(skip_checks: bool = False) -> bool:
    """Verifica que las dependencias web e índice de color estén disponibles."""
    if skip_checks:
        return True

    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
        print("[ok] Dependencias web instaladas")
    except ImportError as exc:
        print(f"[warn] Dependencia faltante: {exc}")
        print("       Ejecuta `pip install -r requirements.txt` y vuelve a intentarlo.")
        return False

    if not Path("color_index.pkl").exists() and not Path("color_index.json").exists():
        print("[warn] No se detectó un índice de color.")
        print("       Genéralo con:")
        print("       python generate_color_index.py --input img --output color_index.pkl")
        return False

    print("[ok] Índice de color listo")
    return True


def start_server(host: str, port: int, reload: bool) -> None:
    """Inicia el servidor FastAPI."""
    import uvicorn

    print(f"\n[info] Iniciando Morsaicos Studio en http://{host}:{port}")
    if reload:
        print("[info] Recarga automática activa (modo desarrollo)")
    print("[info] Presiona Ctrl+C para detener el servidor\n")
    print("[info] Basado en la investigación de Benítez y López (UNAM, 2023)")

    try:
        uvicorn.run(
            "web_app:app",
            host=host,
            port=port,
            reload=reload,
        )
    except KeyboardInterrupt:
        print("\n[info] Servidor detenido. ¡Gracias por usar Morsaicos!")
    except Exception as exc:
        print(f"[error] No se pudo iniciar el servidor: {exc}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inicia la aplicación FastAPI de Morsaicos Studio."
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Interfaz a enlazar (predeterminado: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8088,
        help="Puerto de escucha (predeterminado: 8088)",
    )
    parser.add_argument(
        "--reload",
        dest="reload",
        action="store_true",
        default=True,
        help="Activa la recarga automática (por defecto encendida)",
    )
    parser.add_argument(
        "--no-reload",
        dest="reload",
        action="store_false",
        help="Desactiva la recarga automática, incluso en desarrollo",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Omite la verificación de dependencias e índice de color.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    print("== Morsaicos Studio Web Application ==")
    print("   Proyecto academico inspirado en la UNAM - IIMAS")
    print("=" * 50)

    if not check_requirements(skip_checks=args.skip_checks):
        print("\n[error] Configuración incompleta. Corrige los puntos anteriores e inténtalo de nuevo.")
        return 1

    start_server(host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
