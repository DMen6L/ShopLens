"""
segmentation.py — Foreground isolation before feature extraction.

Two strategies:
  grabcut_foreground(img)  — primary path, uses GrabCut with an auto-estimated rect
  watershed_segment(img)   — fallback path, uses Otsu + morphological markers + Watershed

Both return a uint8 mask (same H×W as input, 255 = foreground, 0 = background)
and the masked BGR image with the background zeroed out.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grabcut_foreground(img: np.ndarray, iterations: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Isolate the product foreground using GrabCut.

    The bounding rect is estimated as a 10 % inset from each edge — no manual ROI needed.

    Parameters
    ----------
    img        : BGR uint8 image
    iterations : GrabCut EM iterations (default 5)

    Returns
    -------
    mask       : uint8 (H, W)  — 255 foreground, 0 background
    masked_img : BGR  (H, W, 3) — original pixels where mask==255, else 0
    """
    h, w = img.shape[:2]

    inset_y = max(1, int(h * 0.10))
    inset_x = max(1, int(w * 0.10))
    rect = (inset_x, inset_y, w - 2 * inset_x, h - 2 * inset_y)

    # GrabCut working arrays
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    gc_mask = np.zeros((h, w), np.uint8)

    cv2.grabCut(img, gc_mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

    # Both GC_FGD and GC_PR_FGD are treated as foreground
    mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Clean up small holes / specks
    mask = _morphological_cleanup(mask)

    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return mask, masked_img


def watershed_segment(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fallback segmentation using Otsu thresholding + distance transform + Watershed.

    Parameters
    ----------
    img        : BGR uint8 image

    Returns
    -------
    mask       : uint8 (H, W)  — 255 foreground, 0 background
    masked_img : BGR  (H, W, 3) — original pixels where mask==255, else 0
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu threshold on blurred gray
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological opening removes noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure-background: dilate the opened mask
    sure_bg = cv2.dilate(opened, kernel, iterations=3)

    # Sure-foreground: distance transform peak
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # Unknown region between sure-bg and sure-fg
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label markers for Watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1                   # background gets label 1
    markers[unknown == 255] = 0            # unknown → 0 (watershed will fill)

    markers = cv2.watershed(img, markers)

    # Everything not background (label 1) and not boundary (-1) is foreground
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    mask[(markers > 1)] = 255

    mask = _morphological_cleanup(mask)

    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return mask, masked_img


# ---------------------------------------------------------------------------
# Internal helpers ---------------------------------------------------------------------------

def _morphological_cleanup(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Remove small specks and fill small holes in a binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened
