"""
Re-Identification Component
Prevents counting the same person twice using color histogram + texture matching
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List

from config import SIMILARITY_THRESHOLD


class ReIDDatabase:
    """Database for storing and matching person features."""

    def __init__(self, threshold: float = SIMILARITY_THRESHOLD):
        """
        Initialize Re-ID database.

        Args:
            threshold: Similarity threshold for matching
        """
        self.threshold = threshold
        self.features_db: List[dict] = []  # Changed to store dict with color and texture

    def _extract_lbp_histogram(self, gray_img: np.ndarray, num_points: int = 8, radius: int = 1) -> np.ndarray:
        """
        Extract Local Binary Pattern histogram for texture analysis.

        Args:
            gray_img: Grayscale image
            num_points: Number of points in LBP circle
            radius: Radius of LBP circle

        Returns:
            Normalized LBP histogram
        """
        rows, cols = gray_img.shape
        lbp = np.zeros((rows - 2 * radius, cols - 2 * radius), dtype=np.uint8)

        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = gray_img[i, j]
                binary = 0
                # Sample 8 points around center
                binary |= (1 << 7) if gray_img[i - 1, j - 1] >= center else 0
                binary |= (1 << 6) if gray_img[i - 1, j] >= center else 0
                binary |= (1 << 5) if gray_img[i - 1, j + 1] >= center else 0
                binary |= (1 << 4) if gray_img[i, j + 1] >= center else 0
                binary |= (1 << 3) if gray_img[i + 1, j + 1] >= center else 0
                binary |= (1 << 2) if gray_img[i + 1, j] >= center else 0
                binary |= (1 << 1) if gray_img[i + 1, j - 1] >= center else 0
                binary |= (1 << 0) if gray_img[i, j - 1] >= center else 0
                lbp[i - radius, j - radius] = binary

        # Create histogram with 59 bins (uniform LBP approximation)
        hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 256))
        hist = hist.astype(np.float32)
        cv2.normalize(hist, hist)
        return hist

    def extract_features(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[dict]:
        """
        Extract color, texture, and body shape features from a person crop.

        Args:
            frame: Full frame image
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Dict with 'color', 'texture', and 'aspect_ratio' features, or None if extraction fails
        """
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 10:
            return None

        # === BODY SHAPE FEATURES ===
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        aspect_ratio = bbox_height / bbox_width if bbox_width > 0 else 2.0

        # Resize to standard size
        crop_resized = cv2.resize(crop, (64, 128))

        # Split into upper and lower body
        upper = crop_resized[:64, :]
        lower = crop_resized[64:, :]

        # === COLOR FEATURES (HSV histograms) ===
        upper_hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
        lower_hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)

        hist_upper_h = cv2.calcHist([upper_hsv], [0], None, [30], [0, 180])
        hist_upper_s = cv2.calcHist([upper_hsv], [1], None, [32], [0, 256])
        hist_lower_h = cv2.calcHist([lower_hsv], [0], None, [30], [0, 180])
        hist_lower_s = cv2.calcHist([lower_hsv], [1], None, [32], [0, 256])

        cv2.normalize(hist_upper_h, hist_upper_h)
        cv2.normalize(hist_upper_s, hist_upper_s)
        cv2.normalize(hist_lower_h, hist_lower_h)
        cv2.normalize(hist_lower_s, hist_lower_s)

        color_features = np.concatenate([
            hist_upper_h.flatten(),
            hist_upper_s.flatten(),
            hist_lower_h.flatten(),
            hist_lower_s.flatten()
        ])

        # === TEXTURE FEATURES (LBP histograms) ===
        upper_gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
        lower_gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)

        lbp_upper = self._extract_lbp_histogram(upper_gray)
        lbp_lower = self._extract_lbp_histogram(lower_gray)

        texture_features = np.concatenate([lbp_upper, lbp_lower])

        return {
            'color': color_features,
            'texture': texture_features,
            'aspect_ratio': aspect_ratio
        }

    def _compute_color_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute color similarity using histogram correlation."""
        if feat1 is None or feat2 is None:
            return 0.0

        n = len(feat1) // 4
        similarities = []

        for i in range(4):
            h1 = feat1[i*n:(i+1)*n].astype(np.float32)
            h2 = feat2[i*n:(i+1)*n].astype(np.float32)
            sim = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
            similarities.append(sim)

        # Weighted average (upper body slightly more important)
        weights = [0.3, 0.2, 0.3, 0.2]
        return sum(s * w for s, w in zip(similarities, weights))

    def _compute_texture_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute texture similarity using histogram correlation."""
        if feat1 is None or feat2 is None:
            return 0.0

        n = len(feat1) // 2
        # Upper and lower body LBP
        h1_upper = feat1[:n].astype(np.float32)
        h1_lower = feat1[n:].astype(np.float32)
        h2_upper = feat2[:n].astype(np.float32)
        h2_lower = feat2[n:].astype(np.float32)

        sim_upper = cv2.compareHist(h1_upper, h2_upper, cv2.HISTCMP_CORREL)
        sim_lower = cv2.compareHist(h1_lower, h2_lower, cv2.HISTCMP_CORREL)

        # Equal weight for upper and lower
        return 0.5 * sim_upper + 0.5 * sim_lower

    def _compute_aspect_ratio_similarity(self, ar1: float, ar2: float) -> float:
        """Compute similarity based on body aspect ratio."""
        # Typical person aspect ratio is around 2.0-3.0 (height/width)
        # We use a gaussian-like similarity based on difference
        diff = abs(ar1 - ar2)
        # If difference > 0.5, significantly different body shape
        # Returns 1.0 for identical, ~0.6 for diff=0.3, ~0.1 for diff=0.7
        similarity = np.exp(-diff * diff / 0.18)
        return float(similarity)

    def _compute_similarity(self, feat1: Optional[dict], feat2: Optional[dict]) -> float:
        """Compute combined similarity using color, texture, and body shape."""
        if feat1 is None or feat2 is None:
            return 0.0

        color_sim = self._compute_color_similarity(feat1['color'], feat2['color'])
        texture_sim = self._compute_texture_similarity(feat1['texture'], feat2['texture'])
        aspect_sim = self._compute_aspect_ratio_similarity(feat1['aspect_ratio'], feat2['aspect_ratio'])

        # Combined weights: 50% color, 30% texture, 20% body shape
        # Body shape helps differentiate tall/thin vs short/wide people
        combined = 0.50 * color_sim + 0.30 * texture_sim + 0.20 * aspect_sim
        return combined

    def is_duplicate(self, features: Optional[dict]) -> Tuple[bool, float]:
        """
        Check if person already exists in database.

        Args:
            features: Feature dict with 'color' and 'texture' keys

        Returns:
            Tuple of (is_duplicate, max_similarity)
        """
        if features is None or len(self.features_db) == 0:
            return False, -1.0

        max_similarity = -1.0
        for stored_features in self.features_db:
            similarity = self._compute_similarity(features, stored_features)
            if similarity > max_similarity:
                max_similarity = similarity

        is_dup = max_similarity > self.threshold

        if is_dup:
            print(f"  [DUPLICATE] Similarity: {max_similarity:.3f} > {self.threshold}")
        else:
            print(f"  [NEW] Best similarity: {max_similarity:.3f} < {self.threshold}")

        return is_dup, max_similarity

    def add_person(self, features: Optional[dict]):
        """Add person's features to database."""
        if features is not None:
            self.features_db.append(features)

    def clear(self):
        """Clear the database."""
        self.features_db.clear()

    @property
    def count(self) -> int:
        """Number of people in database."""
        return len(self.features_db)
