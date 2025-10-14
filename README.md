# Fotomorsaicos — EGC, PCIC, UNAM

Morsaicos Studio es un sistema para la generación y evaluación de fotomosaicos desarrollado en el marco del Examen General de Conocimientos de la Maestría en Ciencia e Ingeniería de la Computación. La propuesta integra una canalización reproducible de análisis, selección y composición de teselas, así como una interfaz web que facilita la inspección cualitativa y cuantitativa de resultados. El proyecto se sustenta en la línea UNAM/IIMAS sobre fotomosaicos y adopta las métricas y procedimientos reportados en la literatura reciente.

## Resumen

El sistema implementa: i) un índice perceptual de color para un gran catálogo de teselas; ii) un generador de fotomosaicos con control de repetición, vecindad y métrica de distancia cromática; iii) un módulo de artefactos derivados (mezcla y escala de grises) y realce tonal; iv) un servicio web (FastAPI) que coordina trabajos en segundo plano, reporta progreso y expone métricas; y v) una interfaz web de una sola página con previsualización en vivo y un panel de métricas y rendimiento.

## Objetivos

- Formalizar una canalización modular y verificable para la construcción de fotomosaicos.
- Emplear métricas de evaluación (repetición promedio, entropía, tiempos por etapa) para sustentar la comparación experimental.
- Proveer una interfaz de uso académico que facilite la inspección, descarga y documentación de resultados.

## Requisitos y entorno

- Python 3.13.
- Dependencias en `requirements.txt` / `pyproject.toml` (instalación mediante `pip` o `uv`).
- En Windows, se recomiendan herramientas de compilación (C/C++) para paquetes como `numba`/`torch`.
- Conjunto de teselas en `img/` para construir el índice de color.

### Instalación

1) Crear y activar entorno virtual:

```bash
python -m venv .venv
. .\.venv\Scripts\activate.ps1  # Windows PowerShell
# source .venv/bin/activate  # Linux/macOS
```

2) Instalar dependencias:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# o: uv pip install -r requirements.txt
```

## Índice de color (pre‑requisito)

El servicio requiere un índice de color precalculado en la raíz del proyecto; para generarlo a partir de `img/`:

```bash
python generate_color_index.py
```

El proceso crea `color_index.{pkl,json,txt}`. En catálogos amplios el tiempo y el tamaño resultante pueden ser elevados.

## Puesta en marcha del servicio web

```bash
python start_webapp.py --host 127.0.0.1 --port 8088
```

Parámetros relevantes: `--no-reload`, `--skip-checks`, `--host`, `--port`.
La interfaz en `http://127.0.0.1:8088` permite cargar una imagen, seleccionar presets (Rápido, Equilibrado, Alta calidad, UNAM), observar la previsualización en vivo, alternar vistas (mosaico, mezcla, escala de grises) y descargar resultados.

## API REST

| Método | Ruta | Descripción |
| --- | --- | --- |
| GET | `/api/mosaics` | Lista los trabajos recientes. |
| GET | `/api/mosaics/{id}` | Detalle de un trabajo (estado, métricas, artefactos). |
| GET | `/api/mosaics/defaults` | Parámetros predeterminados del generador. |
| POST | `/api/mosaics` | Encola un nuevo trabajo (multipart + JSON). |
| GET | `/api/diagnostics` | Estadísticas del índice de color y parámetros de sondeo. |

Modelo de solicitud (simplificado):

```bash
curl -X POST "http://127.0.0.1:8088/api/mosaics" \
  -F "source_image=@/ruta/a/imagen.jpg" \
  -F 'request_payload={"parameters":{"cell_size":32,"blend_ratio":0.28}}'
```

## Arquitectura y flujo

- `ColorIndexLoader` carga y cachea el índice de color (formatos `.pkl`, `.json`, `.txt`).
- `PhotomosaicGenerator` realiza análisis por celdas y selección de teselas con métrica de color (CIEDE2000, Riemersma, híbrida), control de repetición y vecindad.
- `MosaicPipeline` coordina la ejecución en segundo plano, emite previsualización, construye artefactos (mezcla, grises) y calcula métricas.
- `JobStore` mantiene un registro en memoria de trabajos con políticas de retención.
- Interfaz web: previsualización en vivo, historia de ejecuciones (clic para consultar resultados), tabla de métricas y gráfica de “celdas vs tiempo”.

### Métricas reportadas

- Celdas procesadas, teselas únicas, repetición promedio, entropía del mosaico.
- Tiempos por etapa (selección, artefactos, totales) y rendimiento (celdas/s).
- Parámetros efectivos (p. ej., mezcla y realce tonal) para trazabilidad experimental.

## Ejemplo programático

```py3
from pathlib import Path
from app.services.color_index import ColorIndexLoader
from photomosaic_generator import PhotomosaicGenerator

indice = ColorIndexLoader(Path("color_index.pkl")).load()
generador = PhotomosaicGenerator(color_index=indice, base_directory=".")
resumen = generador.generate_mosaic_description_parallel(
    image_path="img/ejemplo.jpg",
    cell_size=32,
    output_path="outputs/ejemplo_mosaico.jpg",
    top_candidates=48,
    repetition_penalty=0.35,
    max_repetitions=8,
    neighbor_radius=1,
)
print(resumen)
```

## Estructura del repositorio

- `app/` Aplicación FastAPI (configuración, rutas, servicios, canalización).
- `static/` Recursos estáticos de la interfaz (`css/app.css`, `js/app.js`).
- `templates/` Plantillas Jinja (entrada principal: `templates/index.html`).
- `photomosaic_generator.py`, `mosaic_enhancer.py`, `color_analyzer.py` Núcleo de procesamiento.
- `generate_color_index.py` Utilidad para construir el índice de color.
- `start_webapp.py` Lanzador del servicio web.
- `outputs/`, `uploads/`, `temp/` Directorios de trabajo.

## Reproducibilidad y pruebas

- Verificación rápida de sintaxis:

```bash
python -m compileall app
```

- Se recomienda fijar versiones y documentar el entorno (SO, CPU, RAM, versiones de `numpy`, `torch`, `numba`).

## Limitaciones y trabajo futuro

- Optimización adicional en selección de candidatos y/o estrategias de paralelismo.
- Integración de nuevas métricas perceptuales y validación con baterías de imágenes de referencia.
- Empaquetamiento para despliegue reproducible (contenedores) y/o GPU backend dedicado.

## Referencias

- Benítez Pérez, H., & López Michelone, M. (2023). *Un estudio empírico de los fotomosaicos*. Research in Computing Science, 152(6).

— Proyecto académico para el Examen General de Conocimientos (Maestría en Ciencia e Ingeniería de la Computación).
