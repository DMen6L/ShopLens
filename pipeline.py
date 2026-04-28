"""
pipeline.py — Feature extraction pipeline using OpenCV.

FeaturePipeline.extract(img) runs all feature groups on the input image
and returns a dict that matches the shape expected by ProductDB.add().

Feature groups
--------------
1. Fourier contour descriptor  — (32,) float32, shape-ANN embedding
   Invariant to translation, scale, rotation, and contour start-point.
   Captures whole-object silhouette; ignores interior texture / logos.

2. SIFT descriptors            — (N, 128) float32, optional structural evidence
3. 2D HSV histogram            — (50, 60) H×S bins, normalized; color signal
4. Hu moments                  — (7,) log-transformed for L2 stability
5. Harris corner density       — scalar float, texture density tie-breaker
"""

import cv2
import numpy as np


# HSV histogram bins
_H_BINS = 50
_S_BINS = 60

# SIFT: cap descriptors at this many to keep storage bounded
_MAX_SIFT_KP = 500

# Boundary ring: fraction of the foreground region's estimated radius used as
# ring thickness when masking SIFT / Harris to the product outline.
# Keeps detectors on the structural silhouette and away from interior logos.
_BOUNDARY_RING_FRACTION = 0.18

# Logo exclusion: fraction of the estimated foreground radius to blank out at
# the centroid before computing the HSV histogram.  Logos sit in the centre;
# excluding this circle means the histogram reflects the product's packaging
# colour rather than the logo's colours.
_LOGO_EXCLUSION_FRACTION = 0.50

# Fourier: resample contour to this many points before FFT
_CONTOUR_RESAMPLE_N = 256
# Number of Fourier coefficients kept as the shape embedding
_FOURIER_N_COEFFS = 32


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
            embedding      np.ndarray (32,)    float32  — L2-normed Fourier shape descriptor
                                                          (used for ANN search)
            sift_kp        list[cv2.KeyPoint]  — keypoints (needed by matcher RANSAC)
            sift_desc      np.ndarray (N, 128) float32  or None if no kp found
            hist_hsv       np.ndarray (50, 60) float32
            hu_moments     np.ndarray (7,)     float64
            corner_density float
            shape_geo      np.ndarray (3,)     float32  — L2-normed [solidity, elongation, extent]
        """
        embedding = self._compute_contour_fourier(img, mask)
        sift_kp, sift_desc = self._compute_sift(img, mask)
        hist_hsv = self._compute_hsv_histogram(img, mask)
        hu_moments = self._compute_hu_moments(img, mask)
        corner_density = self._compute_corner_density(img, mask)
        shape_geo = self._compute_shape_geometry(img, mask)

        return {
            "embedding":      embedding,      # 32-dim Fourier shape — ANN key
            "sift_kp":        sift_kp,        # list[KeyPoint] — used for RANSAC
            "sift_desc":      sift_desc,
            "hist_hsv":       hist_hsv,
            "hu_moments":     hu_moments,
            "corner_density": corner_density,
            "shape_geo":      shape_geo,      # [solidity, elongation, extent]
        }

    # ------------------------------------------------------------------
    # Feature extractors
    # ------------------------------------------------------------------

    def _compute_contour_fourier(
        self, img: np.ndarray, mask: np.ndarray | None
    ) -> np.ndarray:
        """
        Shape descriptor via Fourier analysis of the largest object contour.

        Pipeline
        --------
        1. Find the largest foreground contour.
        2. Resample to _CONTOUR_RESAMPLE_N equally-spaced (arc-length) points.
        3. Treat the (x, y) sequence as a complex signal z[k] = x[k] + j*y[k].
        4. Compute FFT(z).
        5. Normalise:
             - Translation invariance  → skip Z[0] (DC component).
             - Scale invariance        → divide all coefficients by |Z[1]|.
             - Rotation invariance     → take magnitudes (phase of Z[1] dropped).
             - Start-point invariance  → magnitudes are already phase-independent.
        6. Keep the first _FOURIER_N_COEFFS magnitude values (Z[1..N]).
        7. L2-normalise for cosine-compatible ANN search.

        Returns zeros if no usable contour is found.
        """
        src = mask if mask is not None else _otsu_binary(img)

        contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return np.zeros(_FOURIER_N_COEFFS, dtype=np.float32)

        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 4:
            return np.zeros(_FOURIER_N_COEFFS, dtype=np.float32)

        pts = _resample_contour(contour, _CONTOUR_RESAMPLE_N)
        z = pts[:, 0] + 1j * pts[:, 1]

        Z = np.fft.fft(z)

        first_mag = np.abs(Z[1])
        if first_mag < 1e-10:
            return np.zeros(_FOURIER_N_COEFFS, dtype=np.float32)

        # Skip Z[0], normalise by |Z[1]|, take magnitudes
        mags = (np.abs(Z[1: _FOURIER_N_COEFFS + 1]) / first_mag).astype(np.float32)

        # L2-normalise
        norm = np.linalg.norm(mags)
        if norm > 0:
            mags = mags / norm
        return mags

    def _compute_sift(
        self, img: np.ndarray, mask: np.ndarray | None
    ) -> tuple[list, np.ndarray | None]:
        """
        Detect SIFT keypoints and compute float32 descriptors (128-dim).

        When a foreground mask is available, detection is restricted to a
        boundary ring around the product silhouette.  This keeps keypoints
        on structural product edges rather than interior logo pixels, which
        are high-contrast and would otherwise dominate the descriptor set.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift_mask = _boundary_ring_mask(mask) if mask is not None else None
        kp, desc = self._sift.detectAndCompute(gray, sift_mask)
        if desc is None or len(kp) == 0:
            return [], None
        return kp, desc

    def _compute_hsv_histogram(self, img: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        """
        Compute a 2D H×S histogram in HSV space.

        Value channel excluded — lighting-sensitive and hurts cross-condition matching.

        The central region of the foreground (where the logo lives) is excluded
        from the histogram so that the colour signal reflects packaging/product
        colour rather than logo colours.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_mask = _logo_exclusion_mask(mask) if mask is not None else None
        hist = cv2.calcHist(
            [hsv],
            [0, 1],
            hist_mask,
            [_H_BINS, _S_BINS],
            [0, 180, 0, 256],
        )
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.astype(np.float32)

    def _compute_hu_moments(self, img: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        """
        Compute the 7 Hu moments from the largest foreground contour.

        Log-transforms the moments to compress dynamic range.
        """
        src = mask if mask is not None else _otsu_binary(img)

        contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        moments = cv2.moments(max(contours, key=cv2.contourArea)) if contours else cv2.moments(src)

        hu = cv2.HuMoments(moments).flatten()
        hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-7)
        return hu.astype(np.float64)

    def _compute_shape_geometry(
        self, img: np.ndarray, mask: np.ndarray | None
    ) -> np.ndarray:
        """
        Compute a compact shape geometry descriptor from the largest contour.

        Three values, each in [0, 1]:
            solidity   = contour_area / convex_hull_area
                         1.0 = perfectly convex; <1 = concavities / teeth present.
                         Directly separates jagged shapes (saw ~0.7) from smooth
                         shapes (screwdriver ~0.97).
            elongation = min(w, h) / max(w, h) from bounding rect
                         1.0 = square; near 0 = very elongated.
            extent     = contour_area / bounding_rect_area
                         Measures how much of the bounding box the shape fills.

        The vector is L2-normalised before storage so cosine / L2 distance
        are equivalent at query time.
        """
        src = mask if mask is not None else _otsu_binary(img)
        contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(3, dtype=np.float32)

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 1:
            return np.zeros(3, dtype=np.float32)

        # Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area / hull_area) if hull_area > 0 else 1.0

        # Elongation (min/max of bounding rect dims — always in [0, 1])
        _, _, w, h = cv2.boundingRect(contour)
        elongation = (float(min(w, h)) / float(max(w, h))) if max(w, h) > 0 else 1.0

        # Extent
        rect_area = float(w * h)
        extent = float(area / rect_area) if rect_area > 0 else 0.0

        geo = np.array([solidity, elongation, extent], dtype=np.float32)
        norm = np.linalg.norm(geo)
        if norm > 0:
            geo = geo / norm
        return geo

    def _compute_corner_density(self, img: np.ndarray, mask: np.ndarray | None) -> float:
        """
        Harris corner detection; return the density of strong corners.

        Density = corner_count / ring_pixel_count.  Clamped to [0, 1].

        Counting is restricted to the boundary ring of the foreground so that
        interior logo corners (sharp text strokes, icon edges) do not inflate
        the density score.  Two products that differ only in their logo will
        therefore produce similar corner-density scores.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

        threshold = 0.01 * harris.max()
        corner_map = (harris > threshold).astype(np.uint8) * 255

        if mask is not None:
            ring = _boundary_ring_mask(mask)
            corner_map = cv2.bitwise_and(corner_map, corner_map, mask=ring)
            area = float(np.count_nonzero(ring))
        else:
            area = float(gray.shape[0] * gray.shape[1])

        corner_count = float(np.count_nonzero(corner_map))
        density = corner_count / area if area > 0 else 0.0
        return float(np.clip(density, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _otsu_binary(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to binary mask via Otsu thresholding."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _logo_exclusion_mask(mask: np.ndarray) -> np.ndarray:
    """
    Return the foreground mask with the central logo region blanked out.

    Logos are typically printed on the centre face of a product.  Excluding a
    circle of radius (_LOGO_EXCLUSION_FRACTION * estimated_fg_radius) around
    the centroid means colour histograms reflect the product's packaging colour
    rather than the logo's colours.

    Falls back to the original mask if the foreground is too small or the
    exclusion would blank the entire mask.
    """
    area = float(np.count_nonzero(mask))
    if area < 1:
        return mask

    M = cv2.moments(mask)
    if M["m00"] < 1:
        return mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    radius_est = np.sqrt(area / np.pi)
    excl_r = max(1, int(_LOGO_EXCLUSION_FRACTION * radius_est))

    excl_circle = np.zeros_like(mask)
    cv2.circle(excl_circle, (cx, cy), excl_r, 255, -1)
    result = cv2.bitwise_and(mask, cv2.bitwise_not(excl_circle))

    # Safety: if exclusion removed everything, fall back to full mask
    if np.count_nonzero(result) == 0:
        return mask
    return result


def _boundary_ring_mask(mask: np.ndarray) -> np.ndarray:
    """
    Return a ring-shaped mask covering only the outer band of the foreground.

    The ring width is proportional to the estimated radius of the foreground
    blob (_BOUNDARY_RING_FRACTION * sqrt(area / π)).  This keeps SIFT
    keypoints and Harris corners on the structural silhouette of the product
    and away from the interior, where logos generate misleading high-contrast
    features.

    Falls back to the original mask when the foreground is too small or
    erosion would consume it entirely.
    """
    area = float(np.count_nonzero(mask))
    if area < 1:
        return mask

    radius_est = np.sqrt(area / np.pi)
    thickness = max(5, int(_BOUNDARY_RING_FRACTION * radius_est))

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * thickness + 1, 2 * thickness + 1)
    )
    eroded = cv2.erode(mask, kernel)

    # If erosion consumed everything, fall back to the full mask
    if np.count_nonzero(eroded) == 0:
        return mask

    ring = cv2.bitwise_and(mask, cv2.bitwise_not(eroded))
    return ring


def _resample_contour(contour: np.ndarray, n: int) -> np.ndarray:
    """
    Resample *contour* to exactly *n* equally-spaced (arc-length) 2D points.

    Closes the contour before interpolation so frequencies wrap correctly.
    """
    pts = contour.squeeze().astype(np.float64)
    if pts.ndim == 1:
        # Degenerate: single point or line
        pts = pts.reshape(-1, 2)

    # Close the loop
    pts_closed = np.vstack([pts, pts[:1]])
    diffs = np.diff(pts_closed, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cumlen[-1]

    if total < 1e-10:
        return np.zeros((n, 2), dtype=np.float64)

    t_query = np.linspace(0.0, total, n, endpoint=False)
    new_x = np.interp(t_query, cumlen, pts_closed[:, 0])
    new_y = np.interp(t_query, cumlen, pts_closed[:, 1])
    return np.column_stack([new_x, new_y])
