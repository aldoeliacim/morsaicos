"""
GPU-aware wrapper for the high quality photomosaic generator.
The current high fidelity pipeline relies on perceptual metrics that are
computed efficiently on CPU; this class keeps the interface compatible
and prepares for future GPU-specific optimisations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from photomosaic_generator import PhotomosaicGenerator


class GPUMosaicGenerator(PhotomosaicGenerator):
    """Wrapper que mantiene compatibilidad con la interfaz GPU previa."""

    def __init__(self,
                 color_index: Optional[Dict[str, Dict[str, Any]]] = None,
                 base_directory: str = ".",
                 device: str = "auto"):
        super().__init__(color_index=color_index, base_directory=base_directory)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            info.update({
                "device_name": torch.cuda.get_device_name(0),
                "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
            })
        return info

    def generate_mosaic_gpu(self, *args, **kwargs):
        """Mantiene la firma original delegando al generador de alta fidelidad."""
        print("⚙️  Ejecutando pipeline de alta calidad (CPU optimizada).")
        return self.generate_mosaic_description_parallel(*args, **kwargs)


__all__ = ["GPUMosaicGenerator"]
