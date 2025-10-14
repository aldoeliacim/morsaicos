"""
Modulo de mejora y analisis de calidad de fotomosaicos.
"""
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import Counter

from color_analyzer import ColorAnalyzer


class MosaicEnhancer:
    """Clase para mejorar y analizar la calidad de fotomosaicos."""

    @staticmethod
    def apply_blending(mosaic_image: Image.Image,
                      original_image: Image.Image,
                      blend_percentage: float = 0.25,
                      detail_strength: float = 0.35) -> Image.Image:
        """Fusiona el mosaico con la imagen original inyectando detalle."""
        if mosaic_image.size != original_image.size:
            original_resized = original_image.resize(mosaic_image.size, Image.Resampling.LANCZOS)
        else:
            original_resized = original_image

        if mosaic_image.mode != 'RGB':
            mosaic_image = mosaic_image.convert('RGB')
        if original_resized.mode != 'RGB':
            original_resized = original_resized.convert('RGB')

        blended = Image.blend(mosaic_image, original_resized, blend_percentage)

        if detail_strength > 0:
            original_detail = original_resized.filter(ImageFilter.GaussianBlur(radius=2))
            detail_map = np.array(original_resized, dtype=np.float32) - np.array(original_detail, dtype=np.float32)
            mosaic_np = np.array(blended, dtype=np.float32)
            enhanced = np.clip(mosaic_np + detail_map * detail_strength, 0, 255).astype(np.uint8)
            blended = Image.fromarray(enhanced)

        return blended

    @staticmethod
    def calculate_repetition_index(mosaic_description: List[List[str]]) -> float:
        if not mosaic_description:
            return 0.0
        all_images = [image for row in mosaic_description for image in row]
        total_cells = len(all_images)
        unique_images = len(set(all_images))
        if unique_images == 0:
            return float('inf')
        return total_cells / unique_images

    @staticmethod
    def analyze_repetitions(mosaic_description: List[List[str]]) -> Dict[str, int]:
        all_images = [image for row in mosaic_description for image in row]
        return dict(Counter(all_images))

    @staticmethod
    def calculate_entropy_from_description(mosaic_description: List[List[str]]) -> float:
        repetitions = MosaicEnhancer.analyze_repetitions(mosaic_description)
        if not repetitions:
            return 0.0
        total_cells = sum(repetitions.values())
        probabilities = [count / total_cells for count in repetitions.values()]
        return -sum(p * np.log2(p) for p in probabilities if p > 0)

    @staticmethod
    def calculate_grayscale_entropy(image_path: str) -> float:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"No se encontro la imagen para calcular entropia gris: {image_path}")
        with Image.open(path) as img:
            gray = img.convert("L")
            histogram = np.array(gray.histogram(), dtype=np.float64)
            total = histogram.sum()
            if total == 0:
                return 0.0
            probabilities = histogram / total
            probabilities = probabilities[probabilities > 0]
            return float(-np.sum(probabilities * np.log2(probabilities)))

    @staticmethod
    def generate_repetition_plot(mosaic_description: List[List[str]],
                               output_path: str = None) -> None:
        repetitions = MosaicEnhancer.analyze_repetitions(mosaic_description)
        if not repetitions:
            print("No hay datos de repeticiones para graficar.")
            return

        sorted_repetitions = sorted(repetitions.values())
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(sorted_repetitions)), sorted_repetitions, 'b-', linewidth=2)
        plt.xlabel('Indice de imagen (ordenado por repeticiones)')
        plt.ylabel('Numero de repeticiones')
        plt.title('Distribucion de repeticiones en el fotomosaico')
        plt.grid(True, alpha=0.3)

        total_images = len(repetitions)
        total_cells = sum(repetitions.values())
        repetition_index = MosaicEnhancer.calculate_repetition_index(mosaic_description)

        stats_text = (f'Imagenes unicas: {total_images}\n'
                      f'Celdas totales: {total_cells}\n'
                      f'Indice de repeticion: {repetition_index:.3f}\n'
                      f'Repeticiones promedio: {total_cells / total_images:.2f}')

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Grafico guardado en: {output_path}")
        else:
            plt.show()
        plt.close()

    @staticmethod
    def calculate_quality_metrics(
        mosaic_description: List[List[str]],
        source_image_path: str = None,
        mosaic_image_path: str = None,
    ) -> Dict:
        metrics: Dict[str, Any] = {}
        repetitions = MosaicEnhancer.analyze_repetitions(mosaic_description)
        total_cells = sum(len(row) for row in mosaic_description)
        unique_images = len(repetitions)

        metrics["total_cells"] = total_cells
        metrics["unique_images"] = unique_images
        metrics["repetition_index"] = MosaicEnhancer.calculate_repetition_index(mosaic_description)
        metrics["entropy"] = MosaicEnhancer.calculate_entropy_from_description(mosaic_description)

        if repetitions:
            values = list(repetitions.values())
            metrics["max_repetitions"] = max(values)
            metrics["min_repetitions"] = min(values)
            metrics["avg_repetitions"] = np.mean(values)
            metrics["std_repetitions"] = np.std(values)
            metrics["repetition_variance"] = np.var(values)
            if metrics["avg_repetitions"] > 0:
                metrics["repetition_coefficient_variation"] = metrics["std_repetitions"] / metrics["avg_repetitions"]

        source_gray_entropy = None
        if source_image_path and Path(source_image_path).exists():
            try:
                source_entropy = ColorAnalyzer.calculate_entropy(source_image_path)
                metrics["source_entropy"] = source_entropy
                metrics["entropy_ratio"] = metrics["entropy"] / source_entropy if source_entropy > 0 else 0
            except Exception as exc:
                print(f"Error calculando entropia de la imagen original: {exc}")
            try:
                source_gray_entropy = MosaicEnhancer.calculate_grayscale_entropy(source_image_path)
                metrics["source_grayscale_entropy"] = source_gray_entropy
            except Exception as exc:
                print(f"Error calculando entropia gris de la imagen original: {exc}")

        if mosaic_image_path and Path(mosaic_image_path).exists():
            try:
                mosaic_gray_entropy = MosaicEnhancer.calculate_grayscale_entropy(mosaic_image_path)
                metrics["grayscale_entropy"] = mosaic_gray_entropy
                if source_gray_entropy and source_gray_entropy > 0:
                    metrics["grayscale_entropy_ratio"] = mosaic_gray_entropy / source_gray_entropy
            except Exception as exc:
                print(f"Error calculando entropia gris del mosaico: {exc}")

        return metrics
    @staticmethod
    def enhance_mosaic_with_filters(mosaic_image: Image.Image,
                                  enhance_contrast: float = 1.1,
                                  enhance_sharpness: float = 1.2,
                                  enhance_color: float = 1.05) -> Image.Image:
        enhanced = mosaic_image.copy()
        if enhance_contrast != 1.0:
            enhanced = ImageEnhance.Contrast(enhanced).enhance(enhance_contrast)
        if enhance_sharpness != 1.0:
            enhanced = ImageEnhance.Sharpness(enhanced).enhance(enhance_sharpness)
        if enhance_color != 1.0:
            enhanced = ImageEnhance.Color(enhanced).enhance(enhance_color)
        return enhanced

    @staticmethod
    def create_quality_report(mosaic_description: List[List[str]],
                            source_image_path: str = None,
                            mosaic_image_path: str = None,
                            output_path: str = None) -> Dict:
        metrics = MosaicEnhancer.calculate_quality_metrics(mosaic_description, source_image_path, mosaic_image_path)
        repetitions = MosaicEnhancer.analyze_repetitions(mosaic_description)

        report = {
            'metrics': metrics,
            'analysis': {
                'quality_assessment': 'excellent' if metrics['repetition_index'] < 2.0 else
                                    'good' if metrics['repetition_index'] < 3.0 else
                                    'fair' if metrics['repetition_index'] < 5.0 else 'poor',
                'recommendations': []
            },
            'repetition_details': {
                'most_repeated_images': sorted(repetitions.items(), key=lambda x: x[1], reverse=True)[:10],
                'single_use_images': len([img for img, count in repetitions.items() if count == 1])
            }
        }

        if metrics['repetition_index'] > 3.0:
            report['analysis']['recommendations'].append("Considerar una coleccion de imagenes mas grande")
        if metrics.get('entropy', 0) < 5.0:
            report['analysis']['recommendations'].append("Usar mas diversidad en la seleccion de imagenes")
        if metrics.get('max_repetitions', 0) > 10:
            report['analysis']['recommendations'].append("Implementar mejor control de repeticiones")

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Reporte de calidad guardado en: {output_path}")

        return report

