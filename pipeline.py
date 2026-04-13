"""
pipeline.py — Feature extraction pipeline using OpenCV.

FeaturePipeline.extract(img) runs all four feature groups on the input image
and returns a dict that matches the shape expected by ProductDB.add().

Feature groups
--------------
1. SIFT descriptors    — (N, 128) float32, keypoints included
2. 2D HSV histogram    — (50, 60) H×S bins, normalized; flattened to 3000-dim embedding
3. Hu moments          — (7,) log-transformed for L2 stability
4. Harris corner density — scalar float, normalized by image area
"""

import cv2
import numpy as np


# HSV histogram bins
_H_BINS = 50
_S_BINS = 60

# SIFT: cap descriptors at this many to keep storage bounded
_MAX_SIFT_KP = 500


class FeaturePipeline:
    def __init__(self):
        self._sift = cv2.SIFT_create(nfeatures=_MAX_SIFT_KP)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def extract(self, img: np.ndarray, mask: np.ndarray | None = None) -> dict:
        """
        Extract all features from a BGR image.

        Parameters
        ----------
        img  : BGR uint8 image (full or already masked)
        mask : optional uint8 foreground mask (255 = fg). When provided, features
               are computed only within the masked region.

        Returns
        -------
        dict with keys:
            sift_desc      np.ndarray (N, 128) float32  or None if no kp found
            hist_hsv       np.ndarray (50, 60) float32
            hu_moments     np.ndarray (7,)     float64
            corner_density float
            embedding      np.ndarray (3000,)  float32  — L2-normed HSV histogram
        """
        sift_desc = self._compute_sift(img, mask)
        hist_hsv = self._compute_hsv_histogram(img, mask)
        hu_moments = self._compute_hu_moments(img, mask)
        corner_density = self._compute_corner_density(img, mask)
        embedding = _flatten_and_normalize(hist_hsv)

        return {
            "sift_desc": sift_desc,
            "hist_hsv": hist_hsv,
            "hu_moments": hu_moments,
            "corner_density": corner_density,
            "embedding": embedding,
        }

    # ------------------------------------------------------------------
    # Feature extractors
    # ------------------------------------------------------------------

    def _compute_sift(self, img: np.ndarray, mask: np.ndarray | None) -> np.ndarray | None:
        """Detect SIFT keypoints and compute descriptors."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = self._sift.detectAndCompute(gray, mask)
        if desc is None or len(kp) == 0:
            return None
        return desc.astype(np.float32)

    def _compute_hsv_histogram(self, img: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        """
        Compute a 2D H×S histogram in HSV space.

        Value channel is intentionally excluded — it is lighting-sensitive and
        hurts cross-condition matching.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1],                          # H and S channels
            mask,
            [_H_BINS, _S_BINS],
            [0, 180, 0, 256],                # H: 0–180, S: 0–255
        )
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.astype(np.float32)

    def _compute_hu_moments(self, img: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        """
        Compute the 7 Hu moments from the largest foreground contour.

        Log-transforms the moments to compress dynamic range:
            h_i = sign(h_i) * log10(|h_i| + 1e-7)
        """
        if mask is not None:
            src = mask
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, src = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Use the largest contour for shape description
            c = max(contours, key=cv2.contourArea)
            moments = cv2.moments(c)
        else:
            moments = cv2.moments(src)

        hu = cv2.HuMoments(moments).flatten()
        # Log-transform for numerical stability
        hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-7)
        return hu.astype(np.float64)

    def _compute_corner_density(self, img: np.ndarray, mask: np.ndarray | None) -> float:
        """
        Harris corner detection; return the density of strong corners.

        Density = corner_count / foreground_pixel_count (or total pixels if no mask).
        Clamped to [0, 1].
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

        # Threshold: keep corners > 1 % of the response maximum
        threshold = 0.01 * harris.max()
        corner_map = (harris > threshold).astype(np.uint8) * 255

        if mask is not None:
            corner_map = cv2.bitwise_and(corner_map, corner_map, mask=mask)
            area = float(np.count_nonzero(mask))
        else:
            area = float(gray.shape[0] * gray.shape[1])

        corner_count = float(np.count_nonzero(corner_map))
        density = corner_count / area if area > 0 else 0.0
        return float(np.clip(density, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_and_normalize(hist: np.ndarray) -> np.ndarray:
    """Flatten a 2D histogram and L2-normalize it for ANN search."""
    v = hist.ravel().astype(np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v
