"""
Módulo de análisis de belleza y calidad de fotomosaicos.
Implementa el "índice de belleza" y analiza características de imágenes candidatas.
"""
import numpy as np
from PIL import Image, ImageStat, ImageFilter
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from collections import Counter

from color_analyzer import ColorAnalyzer

class BeautyAnalyzer:
    """Analizador de belleza y calidad de fotomosaicos."""

    @staticmethod
    def calculate_beauty_index(mosaic_description: List[List[str]],
                              source_image_path: str = None,
                              weights: Dict[str, float] = None) -> float:
        """
        Calcula el "índice de belleza" de un fotomosaico.
        Combina múltiples métricas para dar una puntuación objetiva de calidad.

        Args:
            mosaic_description: Descripción del mosaico
            source_image_path: Ruta de la imagen original (opcional)
            weights: Pesos para las diferentes métricas

        Returns:
            float: Índice de belleza (0-100, mayor = mejor)
        """
        if weights is None:
            # Pesos por defecto basados en el estudio empírico
            weights = {
                'repetition': 0.35,      # Menos repetición = mejor
                'entropy': 0.25,         # Mayor entropía = más diversidad
                'distribution': 0.20,    # Mejor distribución de colores
                'coverage': 0.10,        # Cobertura de espacio de color
                'variance': 0.10         # Varianza en las repeticiones
            }

        metrics = {}

        # 1. Métrica de repetición (invertida - menos repetición es mejor)
        repetition_index = BeautyAnalyzer._calculate_repetition_score(mosaic_description)
        metrics['repetition'] = repetition_index

        # 2. Métrica de entropía
        entropy_score = BeautyAnalyzer._calculate_entropy_score(mosaic_description)
        metrics['entropy'] = entropy_score

        # 3. Métrica de distribución
        distribution_score = BeautyAnalyzer._calculate_distribution_score(mosaic_description)
        metrics['distribution'] = distribution_score

        # 4. Métrica de cobertura de espacio de color
        coverage_score = BeautyAnalyzer._calculate_coverage_score(mosaic_description)
        metrics['coverage'] = coverage_score

        # 5. Métrica de varianza en repeticiones
        variance_score = BeautyAnalyzer._calculate_variance_score(mosaic_description)
        metrics['variance'] = variance_score

        # Calcular índice ponderado
        beauty_index = sum(metrics[key] * weights[key] for key in weights)

        # Escalar a 0-100
        beauty_index = max(0, min(100, beauty_index))

        return beauty_index, metrics

    @staticmethod
    def _calculate_repetition_score(mosaic_description: List[List[str]]) -> float:
        """Calcula puntuación basada en repeticiones (menos repetición = mejor)."""
        all_images = []
        for row in mosaic_description:
            all_images.extend(row)

        total_cells = len(all_images)
        unique_images = len(set(all_images))

        if unique_images == 0:
            return 0

        # Índice de repetición (total/únicos)
        rep_index = total_cells / unique_images

        # Convertir a puntuación (1.0 = perfecto, >5.0 = malo)
        # Usar función logarítmica para penalizar altas repeticiones
        score = max(0, 100 - (rep_index - 1) * 20)
        return min(100, score)

    @staticmethod
    def _calculate_entropy_score(mosaic_description: List[List[str]]) -> float:
        """Calcula puntuación basada en entropía de Shannon."""
        repetitions = Counter()
        for row in mosaic_description:
            for img in row:
                repetitions[img] += 1

        total_cells = sum(repetitions.values())

        # Calcular entropía
        entropy = 0
        for count in repetitions.values():
            p = count / total_cells
            if p > 0:
                entropy -= p * np.log2(p)

        # Normalizar entropía (máximo teórico es log2(total_cells))
        max_entropy = np.log2(total_cells)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        return normalized_entropy * 100

    @staticmethod
    def _calculate_distribution_score(mosaic_description: List[List[str]]) -> float:
        """Calcula puntuación basada en distribución espacial."""
        repetitions = Counter()
        for row in mosaic_description:
            for img in row:
                repetitions[img] += 1

        # Calcular coeficiente de variación
        counts = list(repetitions.values())
        if len(counts) < 2:
            return 50  # Puntuación neutra

        mean_count = np.mean(counts)
        std_count = np.std(counts)

        # Coeficiente de variación (menor = más uniforme = mejor)
        cv = std_count / mean_count if mean_count > 0 else 0

        # Convertir a puntuación (0 = perfecto, >1 = malo)
        score = max(0, 100 - cv * 50)
        return min(100, score)

    @staticmethod
    def _calculate_coverage_score(mosaic_description: List[List[str]]) -> float:
        """Calcula puntuación basada en cobertura del espacio de color."""
        unique_images = set()
        for row in mosaic_description:
            for img in row:
                unique_images.add(img)

        total_cells = sum(len(row) for row in mosaic_description)
        coverage_ratio = len(unique_images) / total_cells

        # Convertir a puntuación (más cobertura = mejor)
        return coverage_ratio * 100

    @staticmethod
    def _calculate_variance_score(mosaic_description: List[List[str]]) -> float:
        """Calcula puntuación basada en varianza de repeticiones."""
        repetitions = Counter()
        for row in mosaic_description:
            for img in row:
                repetitions[img] += 1

        counts = list(repetitions.values())
        if len(counts) < 2:
            return 50

        variance = np.var(counts)

        # Normalizar varianza
        mean_count = np.mean(counts)
        normalized_variance = variance / (mean_count ** 2) if mean_count > 0 else 0

        # Menor varianza = mejor distribución
        score = max(0, 100 - normalized_variance * 100)
        return min(100, score)

    @staticmethod
    def analyze_image_quality(image_path: str) -> Dict[str, float]:
        """
        Analiza las características de una imagen para determinar si es buen candidato para mosaicos.

        Args:
            image_path: Ruta de la imagen a analizar

        Returns:
            Dict[str, float]: Métricas de calidad de la imagen
        """
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                metrics = {}

                # 1. Resolución y calidad básica
                width, height = img.size
                total_pixels = width * height
                metrics['resolution_score'] = min(100, (total_pixels / (800 * 600)) * 100)
                metrics['aspect_ratio'] = width / height

                # 2. Estadísticas de color
                stat = ImageStat.Stat(img)

                # Diversidad de color (desviación estándar de cada canal)
                color_diversity = np.mean(stat.stddev)
                metrics['color_diversity'] = min(100, color_diversity / 255 * 100)

                # Brillo promedio
                brightness = np.mean(stat.mean)
                metrics['brightness'] = brightness / 255 * 100

                # 3. Contraste
                gray_img = img.convert('L')
                contrast = np.std(np.array(gray_img))
                metrics['contrast'] = min(100, contrast / 128 * 100)

                # 4. Entropía de la imagen
                entropy = BeautyAnalyzer._calculate_image_entropy(gray_img)
                metrics['entropy'] = min(100, entropy / 8 * 100)  # Max entropy ≈ 8 for 8-bit

                # 5. Detección de bordes (complejidad)
                edges = gray_img.filter(ImageFilter.FIND_EDGES)
                edge_strength = np.mean(np.array(edges))
                metrics['edge_complexity'] = min(100, edge_strength / 255 * 100)

                # 6. Saturación de color
                hsv_img = img.convert('HSV')
                hsv_array = np.array(hsv_img)
                saturation = np.mean(hsv_array[:, :, 1])
                metrics['saturation'] = saturation / 255 * 100

                # 7. Puntuación general de aptitud para mosaicos
                aptitude_score = BeautyAnalyzer._calculate_mosaic_aptitude(metrics)
                metrics['mosaic_aptitude'] = aptitude_score

                return metrics

        except Exception as e:
            print(f"Error analizando imagen {image_path}: {e}")
            return {'error': str(e)}

    @staticmethod
    def _calculate_image_entropy(image: Image.Image) -> float:
        """Calcula la entropía de una imagen."""
        img_array = np.array(image)
        hist, _ = np.histogram(img_array, bins=256, range=(0, 256))

        # Normalizar histograma
        hist = hist / hist.sum()

        # Calcular entropía
        entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Evitar log(0)
        return entropy

    @staticmethod
    def _calculate_mosaic_aptitude(metrics: Dict[str, float]) -> float:
        """
        Calcula la aptitud general de una imagen para ser usada en mosaicos.

        Args:
            metrics: Métricas calculadas de la imagen

        Returns:
            float: Puntuación de aptitud (0-100)
        """
        # Pesos para diferentes características
        weights = {
            'resolution_score': 0.20,    # Resolución adecuada
            'color_diversity': 0.20,     # Diversidad de colores
            'contrast': 0.15,            # Buen contraste
            'entropy': 0.15,             # Complejidad visual
            'saturation': 0.10,          # Saturación de color
            'edge_complexity': 0.10,     # Detalles visuales
            'brightness': 0.10           # Brillo balanceado
        }

        # Penalizar extremos en brillo
        brightness_penalty = 1.0
        if metrics.get('brightness', 50) < 10 or metrics.get('brightness', 50) > 90:
            brightness_penalty = 0.7

        # Penalizar aspect ratios extremos
        aspect_penalty = 1.0
        aspect_ratio = metrics.get('aspect_ratio', 1.0)
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            aspect_penalty = 0.8

        # Calcular puntuación ponderada
        score = 0
        for metric, weight in weights.items():
            score += metrics.get(metric, 0) * weight

        # Aplicar penalizaciones
        score *= brightness_penalty * aspect_penalty

        return max(0, min(100, score))

    @staticmethod
    def compare_mosaics(descriptions: List[Tuple[str, List[List[str]]]]) -> Dict:
        """
        Compara múltiples fotomosaicos y los ordena por calidad.

        Args:
            descriptions: Lista de tuplas (nombre, descripción_mosaico)

        Returns:
            Dict: Resultados de comparación ordenados por calidad
        """
        results = []

        for name, description in descriptions:
            beauty_index, metrics = BeautyAnalyzer.calculate_beauty_index(description)

            results.append({
                'name': name,
                'beauty_index': beauty_index,
                'metrics': metrics,
                'total_cells': sum(len(row) for row in description),
                'unique_images': len(set(img for row in description for img in row))
            })

        # Ordenar por índice de belleza (mayor = mejor)
        results.sort(key=lambda x: x['beauty_index'], reverse=True)

        return {
            'comparison': results,
            'best_mosaic': results[0] if results else None,
            'ranking': [r['name'] for r in results]
        }

    @staticmethod
    def generate_beauty_report(mosaic_description: List[List[str]],
                              source_image_path: str = None,
                              output_path: str = None) -> Dict:
        """
        Genera un reporte completo de belleza del fotomosaico.

        Args:
            mosaic_description: Descripción del mosaico
            source_image_path: Ruta de la imagen original
            output_path: Ruta donde guardar el reporte

        Returns:
            Dict: Reporte completo de belleza
        """
        beauty_index, metrics = BeautyAnalyzer.calculate_beauty_index(
            mosaic_description, source_image_path
        )

        # Análisis adicional
        repetitions = Counter()
        for row in mosaic_description:
            for img in row:
                repetitions[img] += 1

        total_cells = sum(len(row) for row in mosaic_description)

        report = {
            'beauty_index': beauty_index,
            'quality_level': BeautyAnalyzer._get_quality_level(beauty_index),
            'detailed_metrics': metrics,
            'statistics': {
                'total_cells': total_cells,
                'unique_images': len(repetitions),
                'most_repeated': max(repetitions.values()) if repetitions else 0,
                'least_repeated': min(repetitions.values()) if repetitions else 0,
                'average_repetitions': sum(repetitions.values()) / len(repetitions) if repetitions else 0
            },
            'recommendations': BeautyAnalyzer._generate_recommendations(beauty_index, metrics),
            'timestamp': time.time()
        }

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Reporte de belleza guardado en: {output_path}")

        return report

    @staticmethod
    def _get_quality_level(beauty_index: float) -> str:
        """Convierte índice numérico a nivel de calidad descriptivo."""
        if beauty_index >= 80:
            return "Excelente"
        elif beauty_index >= 65:
            return "Muy Bueno"
        elif beauty_index >= 50:
            return "Bueno"
        elif beauty_index >= 35:
            return "Regular"
        else:
            return "Necesita Mejora"

    @staticmethod
    def _generate_recommendations(beauty_index: float, metrics: Dict[str, float]) -> List[str]:
        """Genera recomendaciones basadas en las métricas."""
        recommendations = []

        if metrics.get('repetition', 0) < 50:
            recommendations.append("Aumentar el número de candidatos por celda (top_n)")
            recommendations.append("Considerar una colección de imágenes más grande")

        if metrics.get('entropy', 0) < 40:
            recommendations.append("Usar imágenes con mayor diversidad de colores")
            recommendations.append("Revisar la distribución de colores en la colección")

        if metrics.get('distribution', 0) < 50:
            recommendations.append("Mejorar el algoritmo de selección aleatoria")
            recommendations.append("Verificar el balance de colores en la colección")

        if metrics.get('coverage', 0) < 30:
            recommendations.append("Incrementar la diversidad de la colección de imágenes")

        if beauty_index < 50:
            recommendations.append("Considerar usar blending para mejorar el resultado visual")
            recommendations.append("Evaluar el uso de celdas más pequeñas")

        return recommendations

    @staticmethod
    def analyze_images_in_directory(directory: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Analiza la calidad de imágenes en un directorio.

        Args:
            directory: Directorio con imágenes a analizar
            limit: Límite de imágenes a procesar (None para todas)

        Returns:
            Lista de diccionarios con análisis de cada imagen
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"El directorio {directory} no existe")

        # Obtener lista de archivos de imagen
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        image_files = []

        for ext in image_extensions:
            image_files.extend(directory_path.glob(f"*{ext}"))
            image_files.extend(directory_path.glob(f"*{ext.upper()}"))

        if limit:
            image_files = image_files[:limit]

        results = []
        for image_file in image_files:
            try:
                metrics = BeautyAnalyzer.analyze_image_quality(str(image_file))
                quality_score = metrics.get('mosaic_aptitude', 0) / 100  # Normalizar a 0-1
                results.append({
                    'file': str(image_file),
                    'quality_score': quality_score,
                    'suitable': quality_score > 0.5,  # Umbral de calidad
                    'metrics': metrics
                })
            except Exception as e:
                print(f"Error analizando {image_file}: {e}")
                results.append({
                    'file': str(image_file),
                    'quality_score': 0.0,
                    'suitable': False,
                    'error': str(e)
                })

        return results