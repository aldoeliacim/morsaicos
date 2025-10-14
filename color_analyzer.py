"""
Advanced color analysis utilities for photomosaic generation.
Provides perceptual color metrics, texture descriptors, and helper
functions used across CPU and GPU pipelines.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set, Union

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError
from numba import jit

warnings.filterwarnings("ignore", category=UserWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Reference white for CIE Lab (D65, 2°)
_XYZ_REF_WHITE = np.array([95.047, 100.0, 108.883], dtype=np.float32)

# Conversion matrix from sRGB to XYZ (D65)
_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float32)

_DEFAULT_FEATURE_WEIGHTS = {
    "lab": 0.65,
    "riemersma": 0.0,
    "luma": 0.12,
    "saturation": 0.08,
    "texture": 0.08,
    "contrast": 0.07,
}


_INVALID_IMAGE_CACHE: Set[str] = set()


class ColorAnalyzer:
    """Utility collection for analyzing color statistics of images."""

    SIGNATURE_VERSION = 2

    # ------------------------------------------------------------------
    # Public helpers for average color computation
    # ------------------------------------------------------------------
    @staticmethod
    def get_average_color(image_path: Union[str, Path]) -> Tuple[float, float, float]:
        """Return the RGB average of an image stored on disk."""
        img_array = ColorAnalyzer._load_image_array(image_path)
        if img_array is None:
            return 0.0, 0.0, 0.0

        return ColorAnalyzer.get_average_color_from_array(img_array)

    @staticmethod
    def get_average_color_from_array(img_array: np.ndarray) -> Tuple[float, float, float]:
        """Return the RGB average of a numpy image array."""
        rgb_mean = np.mean(img_array.reshape(-1, 3), axis=0)
        return tuple(float(channel) for channel in rgb_mean)

    # ------------------------------------------------------------------
    # Color signature generation
    # ------------------------------------------------------------------
    @staticmethod
    def get_color_signature(image_path: Union[str, Path], compute_texture: bool = True) -> Dict[str, Any]:
        """Compute the advanced color signature for a file on disk."""
        img_array = ColorAnalyzer._load_image_array(image_path)
        if img_array is None:
            return ColorAnalyzer._signature_from_rgb((128.0, 128.0, 128.0))

        return ColorAnalyzer.compute_signature_from_array(img_array, compute_texture=compute_texture)

    @staticmethod
    def compute_signature_from_array(img_array: np.ndarray, compute_texture: bool = True) -> Dict[str, Any]:
        """Create a perceptual signature directly from an ndarray."""
        if img_array.size == 0:
            return ColorAnalyzer._signature_from_rgb((128.0, 128.0, 128.0))

        img_array = img_array.astype(np.float32, copy=False)
        flat = img_array.reshape(-1, 3)
        avg_rgb = np.mean(flat, axis=0)
        avg_lab = ColorAnalyzer.rgb_array_to_lab(avg_rgb[np.newaxis, :])[0]

        luma = ColorAnalyzer._calculate_luma(avg_rgb)
        saturation = ColorAnalyzer._estimate_saturation(img_array)
        contrast = np.std(flat)
        texture = ColorAnalyzer._estimate_texture(img_array) if compute_texture else 0.0

        return {
            "version": ColorAnalyzer.SIGNATURE_VERSION,
            "avg_color": tuple(float(c) for c in avg_rgb),
            "avg_lab": tuple(float(c) for c in avg_lab),
            "luma": float(luma),
            "saturation": float(saturation),
            "texture": float(texture),
            "contrast": float(contrast),
        }

    @staticmethod
    def ensure_signature(data: Any, image_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Normalise legacy index entries into full signatures."""
        if isinstance(data, dict) and data.get("version") == ColorAnalyzer.SIGNATURE_VERSION:
            needed = {"avg_color", "avg_lab", "luma", "saturation", "texture", "contrast"}
            if needed.issubset(data.keys()):
                return data

        if isinstance(data, dict):
            signature = dict(data)
            signature.setdefault("avg_color", (128.0, 128.0, 128.0))
            if "avg_lab" not in signature:
                lab = ColorAnalyzer.rgb_array_to_lab(np.array(signature["avg_color"], dtype=np.float32)[np.newaxis, :])[0]
                signature["avg_lab"] = tuple(float(c) for c in lab)
            signature.setdefault("luma", ColorAnalyzer._calculate_luma(signature["avg_color"]))
            signature.setdefault("saturation", 0.0)
            signature.setdefault("texture", 0.0)
            signature.setdefault("contrast", 0.0)

            if image_path and (signature.get("texture") == 0.0 or signature.get("saturation") == 0.0):
                arr = ColorAnalyzer._load_image_array(image_path)
                if arr is not None:
                    refreshed = ColorAnalyzer.compute_signature_from_array(arr)
                    signature.update(refreshed)

            signature["version"] = ColorAnalyzer.SIGNATURE_VERSION
            return signature

        if isinstance(data, (tuple, list)) and len(data) >= 3:
            rgb = tuple(float(c) for c in data[:3])
            if image_path:
                return ColorAnalyzer.get_color_signature(image_path)
            lab = ColorAnalyzer.rgb_array_to_lab(np.array(rgb, dtype=np.float32)[np.newaxis, :])[0]
            return {
                "version": ColorAnalyzer.SIGNATURE_VERSION,
                "avg_color": rgb,
                "avg_lab": tuple(float(c) for c in lab),
                "luma": ColorAnalyzer._calculate_luma(rgb),
                "saturation": 0.0,
                "texture": 0.0,
                "contrast": 0.0,
            }

        return ColorAnalyzer._signature_from_rgb((128.0, 128.0, 128.0))

    # ------------------------------------------------------------------
    # Helpers for safe image loading and invalid cache tracking
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_path(image_path: Union[str, Path]) -> str:
        try:
            return str(Path(image_path).resolve())
        except Exception:
            return str(image_path)

    @staticmethod
    def _mark_invalid(normalized_path: str, error: Any, raw_path: Optional[str] = None) -> None:
        if normalized_path in _INVALID_IMAGE_CACHE:
            return
        _INVALID_IMAGE_CACHE.add(normalized_path)
        message = str(error)
        target = raw_path or normalized_path
        print(f"Error procesando {target}: {message}")

    @staticmethod
    def _load_image_array(image_path: Union[str, Path]) -> Optional[np.ndarray]:
        normalized_path = ColorAnalyzer._normalize_path(image_path)
        if normalized_path in _INVALID_IMAGE_CACHE:
            return None

        path_obj = Path(image_path)
        raw_path = str(path_obj)

        try:
            if not path_obj.exists():
                ColorAnalyzer._mark_invalid(normalized_path, "archivo inexistente", raw_path)
                return None
            if path_obj.stat().st_size < 16:
                ColorAnalyzer._mark_invalid(normalized_path, "archivo vacío o corrupto", raw_path)
                return None
        except OSError as exc:
            ColorAnalyzer._mark_invalid(normalized_path, exc, raw_path)
            return None

        try:
            with Image.open(path_obj) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return np.asarray(img, dtype=np.float32)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            ColorAnalyzer._mark_invalid(normalized_path, exc, raw_path)
            return None

    @staticmethod
    def is_invalid_image(image_path: Union[str, Path]) -> bool:
        return ColorAnalyzer._normalize_path(image_path) in _INVALID_IMAGE_CACHE

    @staticmethod
    def _signature_from_rgb(rgb: Tuple[float, float, float]) -> Dict[str, Any]:
        lab = ColorAnalyzer.rgb_array_to_lab(np.array(rgb, dtype=np.float32)[np.newaxis, :])[0]
        return {
            "version": ColorAnalyzer.SIGNATURE_VERSION,
            "avg_color": tuple(float(c) for c in rgb),
            "avg_lab": tuple(float(c) for c in lab),
            "luma": ColorAnalyzer._calculate_luma(rgb),
            "saturation": 0.0,
            "texture": 0.0,
            "contrast": 0.0,
        }

    # ------------------------------------------------------------------
    # Feature matrix helpers
    # ------------------------------------------------------------------
    @staticmethod
    def riemersma_distance_vectorized(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
        """Compute Riemersma perceptual distance between lab1 and each row of lab2."""
        L1, a1, b1 = lab1[0], lab1[1], lab1[2]
        L2 = lab2[:, 0]
        a2 = lab2[:, 1]
        b2 = lab2[:, 2]

        delta_L = L1 - L2
        delta_a = a1 - a2
        delta_b = b1 - b2
        ave_r = (L1 + L2) / 2.0

        weight_r = 2 + (ave_r / 256.0)
        weight_b = 2 + ((255 - ave_r) / 256.0)

        return np.sqrt(weight_r * (delta_a ** 2) + 4.0 * (delta_L ** 2) + weight_b * (delta_b ** 2))

    @staticmethod
    def metric_coefficients(metric: str) -> Dict[str, float]:
        metric = (metric or "hybrid").lower()
        base = dict(_DEFAULT_FEATURE_WEIGHTS)
        if metric == "riemersma":
            base.update({"lab": 0.2, "riemersma": 0.6})
        elif metric == "ciede2000":
            base.update({"lab": 0.85, "riemersma": 0.0})
        else:  # hybrid
            base.update({"lab": 0.6, "riemersma": 0.2})
        return base

    @staticmethod
    def build_feature_matrix(color_index: Dict[str, Any], image_root: Optional[Path] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Convert a color index into vectorised feature arrays."""
        image_names: List[str] = []
        lab_list: List[Tuple[float, float, float]] = []
        luma_list: List[float] = []
        saturation_list: List[float] = []
        texture_list: List[float] = []
        contrast_list: List[float] = []
        signatures: List[Dict[str, Any]] = []

        for name, entry in color_index.items():
            if image_root and not str(name).lower().startswith(('http://', 'https://')):
                disk_path = str((image_root / name).resolve())
            else:
                disk_path = None
            signature = ColorAnalyzer.ensure_signature(entry, disk_path)
            color_index[name] = signature

            image_names.append(name)
            lab_list.append(signature["avg_lab"])
            luma_list.append(signature["luma"])
            saturation_list.append(signature["saturation"])
            texture_list.append(signature["texture"])
            contrast_list.append(signature["contrast"])
            signatures.append(signature)

        features = {
            "lab": np.array(lab_list, dtype=np.float32),
            "luma": np.array(luma_list, dtype=np.float32),
            "saturation": np.array(saturation_list, dtype=np.float32),
            "texture": np.array(texture_list, dtype=np.float32),
            "contrast": np.array(contrast_list, dtype=np.float32),
            "signatures": signatures,
        }

        return image_names, features

    @staticmethod
    def rank_candidates(target_signature: Dict[str, Any],
                        image_names: List[str],
                        feature_matrix: Dict[str, Any],
                        weights: Optional[Dict[str, float]] = None,
                        top_n: int = 30) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Rank candidate tiles using perceptual difference metrics."""
        if not image_names:
            return []

        selected_weights = weights if weights is not None else target_signature.get("distance_metric")
        weights = ColorAnalyzer._normalise_weights(selected_weights)

        target_lab = np.array(target_signature["avg_lab"], dtype=np.float32)
        delta_e_ciede = ColorAnalyzer.delta_e_ciede2000_vectorized(target_lab, feature_matrix["lab"])
        delta_e_riemersma = ColorAnalyzer.riemersma_distance_vectorized(target_lab, feature_matrix["lab"])
        diff_luma = np.abs(feature_matrix["luma"] - target_signature["luma"])
        diff_sat = np.abs(feature_matrix["saturation"] - target_signature["saturation"])
        diff_texture = np.abs(feature_matrix["texture"] - target_signature["texture"])
        diff_contrast = np.abs(feature_matrix["contrast"] - target_signature["contrast"])

        composite = (weights.get("lab", 0.0) * delta_e_ciede +
                     weights.get("riemersma", 0.0) * delta_e_riemersma +
                     weights["luma"] * diff_luma +
                     weights["saturation"] * diff_sat +
                     weights["texture"] * diff_texture +
                     weights["contrast"] * diff_contrast)

        effective_top = min(top_n, composite.shape[0])
        candidate_indices = np.argpartition(composite, effective_top - 1)[:effective_top]
        sorted_indices = candidate_indices[np.argsort(composite[candidate_indices])]

        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for idx in sorted_indices:
            results.append(
                (
                    image_names[idx],
                    float(composite[idx]),
                    feature_matrix["signatures"][idx],
                )
            )

        return results

    # ------------------------------------------------------------------
    # Colour space conversions
    # ------------------------------------------------------------------
    @staticmethod
    def rgb_array_to_lab(rgb: np.ndarray) -> np.ndarray:
        """Convert RGB values in range [0,255] to Lab."""
        if rgb.ndim == 1:
            rgb = rgb[np.newaxis, :]

        rgb_norm = rgb / 255.0
        mask = rgb_norm > 0.04045
        rgb_lin = np.where(mask, ((rgb_norm + 0.055) / 1.055) ** 2.4, rgb_norm / 12.92)

        xyz = rgb_lin @ _RGB_TO_XYZ.T * 100.0
        xyz_norm = xyz / _XYZ_REF_WHITE

        epsilon = 6 / 29
        mask = xyz_norm > epsilon ** 3
        f_xyz = np.where(mask, np.cbrt(xyz_norm), (xyz_norm / (3 * epsilon ** 2)) + (4 / 29))

        L = (116 * f_xyz[:, 1]) - 16
        a = 500 * (f_xyz[:, 0] - f_xyz[:, 1])
        b = 200 * (f_xyz[:, 1] - f_xyz[:, 2])

        return np.stack([L, a, b], axis=1)

    @staticmethod
    def delta_e_ciede2000_vectorized(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
        """Compute CIEDE2000 delta E between lab1 and each row of lab2."""
        L1, a1, b1 = lab1[0], lab1[1], lab1[2]
        L2 = lab2[:, 0]
        a2 = lab2[:, 1]
        b2 = lab2[:, 2]

        C1 = np.sqrt(a1 ** 2 + b1 ** 2)
        C2 = np.sqrt(a2 ** 2 + b2 ** 2)
        C_avg = (C1 + C2) / 2.0

        C_avg_p7 = C_avg ** 7
        G = 0.5 * (1 - np.sqrt(C_avg_p7 / (C_avg_p7 + 25.0 ** 7)))

        a1_prime = (1 + G) * a1
        a2_prime = (1 + G) * a2
        C1_prime = np.sqrt(a1_prime ** 2 + b1 ** 2)
        C2_prime = np.sqrt(a2_prime ** 2 + b2 ** 2)

        h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360.0
        h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360.0

        delta_L_prime = L2 - L1
        delta_C_prime = C2_prime - C1_prime

        h_diff = h2_prime - h1_prime
        delta_h_prime = np.where(
            C1_prime * C2_prime == 0,
            0.0,
            np.where(np.abs(h_diff) <= 180.0, h_diff, h_diff - np.sign(h_diff) * 360.0)
        )
        delta_H_prime = 2.0 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime / 2.0))

        L_avg_prime = (L1 + L2) / 2.0
        C_avg_prime = (C1_prime + C2_prime) / 2.0

        h_sum = h1_prime + h2_prime
        h_avg_prime = np.where(
            C1_prime * C2_prime == 0,
            h_sum,
            np.where(
                np.abs(h_diff) <= 180.0,
                h_sum / 2.0,
                np.where(h_sum < 360.0, (h_sum + 360.0) / 2.0, (h_sum - 360.0) / 2.0)
            )
        )

        T = (1 - 0.17 * np.cos(np.radians(h_avg_prime - 30)) +
             0.24 * np.cos(np.radians(2 * h_avg_prime)) +
             0.32 * np.cos(np.radians(3 * h_avg_prime + 6)) -
             0.20 * np.cos(np.radians(4 * h_avg_prime - 63)))

        delta_theta = 30.0 * np.exp(-((h_avg_prime - 275.0) / 25.0) ** 2)
        R_C = 2 * np.sqrt((C_avg_prime ** 7) / (C_avg_prime ** 7 + 25.0 ** 7))
        S_L = 1 + ((0.015 * (L_avg_prime - 50.0) ** 2) / np.sqrt(20 + (L_avg_prime - 50.0) ** 2))
        S_C = 1 + 0.045 * C_avg_prime
        S_H = 1 + 0.015 * C_avg_prime * T
        R_T = -np.sin(np.radians(2 * delta_theta)) * R_C

        delta_E = np.sqrt(
            (delta_L_prime / S_L) ** 2 +
            (delta_C_prime / S_C) ** 2 +
            (delta_H_prime / S_H) ** 2 +
            R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H)
        )

        return delta_E

    # ------------------------------------------------------------------
    # Legacy distance metrics (kept for compatibility)
    # ------------------------------------------------------------------
    @staticmethod
    def euclidean_distance(color1: Tuple[float, float, float],
                          color2: Tuple[float, float, float]) -> float:
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        return math.sqrt((r2 - r1) ** 2 + (g2 - g1) ** 2 + (b2 - b1) ** 2)

    @staticmethod
    def riemersma_distance(color1: Tuple[float, float, float],
                           color2: Tuple[float, float, float]) -> float:
        r1, g1, b1 = color1
        r2, g2, b2 = color2

        r_avg = (r1 + r2) / 2
        delta_r = r1 - r2
        delta_g = g1 - g2
        delta_b = b1 - b2

        return math.sqrt(
            (2 + r_avg / 256) * (delta_r ** 2) +
            4 * (delta_g ** 2) +
            (2 + (255 - r_avg) / 256) * (delta_b ** 2)
        )

    @staticmethod
    def find_best_matches(target_color: Tuple[float, float, float],
                         color_database: List[Tuple[str, Tuple[float, float, float]]],
                         metric: str = "euclidean",
                         top_n: int = 1) -> List[Tuple[str, float]]:
        distances: List[Tuple[str, float]] = []

        for image_path, color in color_database:
            if metric == "euclidean":
                dist = ColorAnalyzer.euclidean_distance(target_color, color)
            elif metric == "riemersma":
                dist = ColorAnalyzer.riemersma_distance(target_color, color)
            else:
                raise ValueError("Métrica debe ser 'euclidean' o 'riemersma'")
            distances.append((image_path, dist))

        distances.sort(key=lambda x: x[1])
        return distances[:top_n]

    @staticmethod
    @jit(nopython=True)
    def euclidean_distance_vectorized(color1: np.ndarray, color_array: np.ndarray) -> np.ndarray:
        diff = color_array - color1
        return np.sqrt(np.sum(diff * diff, axis=1))

    # ------------------------------------------------------------------
    # Texture / saturation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _calculate_luma(rgb: Iterable[float]) -> float:
        r, g, b = rgb
        return float(0.2126 * r + 0.7152 * g + 0.0722 * b)

    @staticmethod
    def _estimate_saturation(img_array: np.ndarray) -> float:
        rgb = np.clip(img_array / 255.0, 0.0, 1.0)
        max_vals = np.max(rgb, axis=2)
        min_vals = np.min(rgb, axis=2)
        delta = max_vals - min_vals
        saturation = np.divide(
            delta,
            max_vals,
            out=np.zeros_like(delta),
            where=max_vals > 0
        )
        return float(np.mean(saturation))

    @staticmethod
    def _estimate_texture(img_array: np.ndarray) -> float:
        gray = np.dot(img_array[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
        gx = np.gradient(gray, axis=0)
        gy = np.gradient(gray, axis=1)
        magnitude = np.sqrt(gx * gx + gy * gy)
        return float(np.mean(magnitude))

    # ------------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_entropy(image_path: str) -> float:
        try:
            with Image.open(image_path) as img:
                gray_img = img.convert('L')
                img_array = np.asarray(gray_img)
        except Exception as exc:  # pragma: no cover - defensive path
            print(f"Error calculando entropía para {image_path}: {exc}")
            return 0.0

        hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
        total = hist.sum()
        if total == 0:
            return 0.0
        probs = hist / total
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        if not weights:
            return dict(_DEFAULT_FEATURE_WEIGHTS)

        if isinstance(weights, str):
            return ColorAnalyzer._normalise_weights(ColorAnalyzer.metric_coefficients(weights))

        cleaned = {key: max(float(value), 0.0) for key, value in weights.items()}
        for key, default in _DEFAULT_FEATURE_WEIGHTS.items():
            cleaned.setdefault(key, default)

        total = sum(cleaned.values())
        if total == 0:
            return dict(_DEFAULT_FEATURE_WEIGHTS)

        for key in cleaned:
            cleaned[key] /= total
        return cleaned


__all__ = [
    "ColorAnalyzer",
]

