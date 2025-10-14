"""
Generador de índices para la colección de imágenes.
Procesa todas las imágenes y crea un archivo índice con firmas de color
perceptuales listas para el motor de mosaicos de alta calidad.
"""
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
import concurrent.futures
import multiprocessing

from color_analyzer import ColorAnalyzer


class IndexGenerator:
    """Generador de firmas de color para la colección de imágenes."""

    def __init__(self,
                 image_directory: str,
                 output_directory: Optional[str] = None,
                 remote_base_url: Optional[str] = None):
        """Inicializa el generador de índices."""
        self.image_directory = Path(image_directory)
        self.output_directory = Path(output_directory) if output_directory else self.image_directory.parent
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.remote_base_url = remote_base_url.rstrip('/') if remote_base_url else None

    def get_image_files(self) -> List[Path]:
        """Obtiene la lista de archivos de imagen válidos."""
        image_files: List[Path] = []
        for file_path in self.image_directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(file_path)
        return sorted(image_files)

    def _make_remote_key(self, relative_path: str) -> Tuple[str, Optional[str]]:
        if not self.remote_base_url:
            return relative_path, None
        web_path = relative_path.replace('\\', '/').lstrip('/')
        remote_url = f"{self.remote_base_url}/{web_path}"
        return remote_url, remote_url

    def process_single_image(self, image_path: Path) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Procesa una sola imagen para obtener su firma de color."""
        try:
            relative_media_path = str(image_path.relative_to(self.image_directory))
            relative_source_path = str(image_path.relative_to(self.image_directory.parent))
            signature = ColorAnalyzer.get_color_signature(str(image_path))
            if ColorAnalyzer.is_invalid_image(str(image_path)):
                return None
            remote_key, remote_url = self._make_remote_key(relative_media_path)
            signature['source_path'] = relative_source_path
            if remote_url:
                signature['url'] = remote_url
            return remote_key, signature
        except Exception as exc:
            print(f"Error procesando {image_path}: {exc}")
            return None

    def generate_index(self,
                       use_multiprocessing: bool = True,
                       max_workers: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """Genera el índice de firmas para todas las imágenes."""
        image_files = self.get_image_files()
        if not image_files:
            print("No se encontraron imágenes válidas.")
            return {}

        print(f"Procesando {len(image_files)} imágenes...")
        color_index: Dict[str, Dict[str, Any]] = {}

        if use_multiprocessing and len(image_files) > 100:
            if max_workers is None:
                max_workers = min(multiprocessing.cpu_count(), 8)
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.process_single_image, img_path) for img_path in image_files]
                for future in tqdm(concurrent.futures.as_completed(futures),
                                   total=len(futures), desc="Procesando imágenes"):
                    result = future.result()
                    if result:
                        key, signature = result
                        color_index[key] = signature
        else:
            for image_path in tqdm(image_files, desc="Procesando imágenes"):
                result = self.process_single_image(image_path)
                if result:
                    key, signature = result
                    color_index[key] = signature

        print(f"Índice generado con {len(color_index)} imágenes válidas.")
        return color_index

    def save_index(self, color_index: Dict[str, Dict[str, Any]],
                   format: str = "both") -> None:
        """Guarda el índice en archivo(s)."""
        if not color_index:
            print("No hay datos para guardar.")
            return

        self.output_directory.mkdir(parents=True, exist_ok=True)

        if format in ["json", "both"]:
            json_path = self.output_directory / "color_index.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(color_index, f, indent=2, ensure_ascii=False)
            print(f"Índice JSON guardado en: {json_path}")

        if format in ["pickle", "both"]:
            pickle_path = self.output_directory / "color_index.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(color_index, f)
            print(f"Índice pickle guardado en: {pickle_path}")

        if format in ["txt", "both"]:
            txt_path = self.output_directory / "color_index.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                for key, signature in color_index.items():
                    r, g, b = signature.get('avg_color', (0.0, 0.0, 0.0))
                    url = signature.get('url', '')
                    f.write(f"{key}\t{r:.2f}\t{g:.2f}\t{b:.2f}\t{url}\n")
            print(f"Índice TXT guardado en: {txt_path}")

    def load_index(self, format: str = "pickle") -> Dict[str, Dict[str, Any]]:
        """Carga un índice previamente guardado."""
        if format == "json":
            json_path = self.output_directory / "color_index.json"
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return {path: ColorAnalyzer.ensure_signature(signature)
                        for path, signature in data.items()}
        elif format == "pickle":
            pickle_path = self.output_directory / "color_index.pkl"
            if pickle_path.exists():
                with open(pickle_path, 'rb') as f:
                    data = pickle.load(f)
                return {path: ColorAnalyzer.ensure_signature(signature)
                        for path, signature in data.items()}
        elif format == "txt":
            txt_path = self.output_directory / "color_index.txt"
            if txt_path.exists():
                color_index: Dict[str, Dict[str, Any]] = {}
                with open(txt_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 4:
                            key, r, g, b = parts[:4]
                            url = parts[4] if len(parts) > 4 else ''
                            signature = ColorAnalyzer.ensure_signature((float(r), float(g), float(b)))
                            if url:
                                signature['url'] = url
                            color_index[key] = signature
                return color_index
        print(f"No se encontró índice en formato {format}")
        return {}

    def generate_and_save(self,
                          use_multiprocessing: bool = True,
                          save_format: str = "both") -> Dict[str, Dict[str, Any]]:
        """Genera y guarda el índice en un solo paso."""
        color_index = self.generate_index(use_multiprocessing=use_multiprocessing)
        if color_index:
            self.save_index(color_index, format=save_format)
        return color_index
