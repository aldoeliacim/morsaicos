"""
Ensamblador de fotomosaicos con corrección perceptual opcional.
"""
import numpy as np
from PIL import Image, ImageEnhance
import requests
from io import BytesIO
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


class MosaicAssembler:
    """Ensamblador de imágenes para crear el fotomosaico final."""

    def __init__(self, base_directory: str):
        self.base_directory = Path(base_directory)
        self._remote_session = requests.Session()
        self._remote_cache = {}

    def resize_image(self, image: Image.Image, target_size: Tuple[int, int],
                    resample: int = Image.Resampling.LANCZOS) -> Image.Image:
        return image.resize(target_size, resample)

    def load_and_resize_image(self, image_path: str,
                             target_size: Tuple[int, int]) -> Optional[Image.Image]:
        try:
            full_path = (self.base_directory / image_path).resolve()
            with Image.open(full_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return self.resize_image(img, target_size)
        except Exception as exc:
            print(f"Error cargando imagen {image_path}: {exc}")
            return Image.new('RGB', target_size, (128, 128, 128))

    def assemble_mosaic(self, mosaic_description: List[List[str]],
                       cell_width: int, cell_height: int,
                       output_path: str,
                       quality: int = 95,
                       cell_metadata: Optional[Dict[Tuple[int, int], Dict[str, any]]] = None) -> Image.Image:
        if not mosaic_description:
            raise ValueError("La descripción del mosaico está vacía")

        rows = len(mosaic_description)
        cols = len(mosaic_description[0])

        final_width = cols * cell_width
        final_height = rows * cell_height

        print(f"Ensamblando mosaico de {final_width}x{final_height} píxeles...")
        final_image = Image.new('RGB', (final_width, final_height))
        cache: Dict[str, Image.Image] = {}

        for row in tqdm(range(rows), desc="Ensamblando filas"):
            for col in range(cols):
                image_path = mosaic_description[row][col]
                cell_image = cache.get(image_path)

                if cell_image is None:
                    cell_image = self.load_and_resize_image(image_path, (cell_width, cell_height))
                    cache[image_path] = cell_image

                adjusted = cell_image
                if cell_metadata:
                    meta = cell_metadata.get((row, col))
                    if meta:
                        adjusted = self._apply_adjustments(cell_image.copy(), meta)

                final_image.paste(adjusted, (col * cell_width, row * cell_height))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            final_image.save(output_path, 'JPEG', quality=quality, optimize=True)
        else:
            final_image.save(output_path)

        print(f"Fotomosaico guardado en: {output_path}")
        return final_image

    def _apply_adjustments(self, tile_image: Image.Image, metadata: Dict[str, any]) -> Image.Image:
        tile_signature = metadata.get('tile_signature', {})
        target_signature = metadata.get('target_signature', {})

        tile_luma = tile_signature.get('luma', target_signature.get('luma', 0.0))
        target_luma = target_signature.get('luma', tile_luma)
        if tile_luma > 0:
            brightness = np.clip((target_luma + 1e-3) / (tile_luma + 1e-3), 0.6, 1.6)
            if abs(brightness - 1.0) > 0.05:
                tile_image = ImageEnhance.Brightness(tile_image).enhance(float(brightness))

        tile_sat = tile_signature.get('saturation', 0.0)
        target_sat = target_signature.get('saturation', tile_sat)
        if tile_sat > 0:
            saturation = np.clip((target_sat + 1e-3) / (tile_sat + 1e-3), 0.5, 1.6)
        else:
            saturation = np.clip(1.0 + (target_sat - tile_sat), 0.5, 1.6)
        if abs(saturation - 1.0) > 0.05:
            tile_image = ImageEnhance.Color(tile_image).enhance(float(saturation))

        tile_contrast = tile_signature.get('contrast', 0.0)
        target_contrast = target_signature.get('contrast', tile_contrast)
        if tile_contrast > 0:
            contrast = np.clip((target_contrast + 1e-3) / (tile_contrast + 1e-3), 0.7, 1.4)
            if abs(contrast - 1.0) > 0.05:
                tile_image = ImageEnhance.Contrast(tile_image).enhance(float(contrast))

        return tile_image
