"""
visualizer.py — Visualization utilities for ShopLens match results.

Two outputs (both returned as base64-encoded PNG strings):
  1. draw_matches()    — side-by-side query/candidate with ORB keypoint lines
  2. draw_score_bars() — horizontal confidence bar chart (hist / orb / hu / corner)
"""

from __future__ import annotations

import base64

import cv2
import numpy as np

from matcher import MatchResult


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_matches(
    query_img: np.ndarray,
    candidate_img: np.ndarray,
    max_matches: int = 40,
    target_height: int = 400,
) -> str:
    """
    Detect ORB keypoints on both images and draw good matches between them.

    Falls back to a plain side-by-side view when either image yields no
    descriptors (e.g. blank / very small images).

    Parameters
    ----------
    query_img      : BGR uint8 query image
    candidate_img  : BGR uint8 candidate image
    max_matches    : maximum number of match lines to draw (for readability)
    target_height  : both images are resized to this height before compositing

    Returns
    -------
    Base64-encoded PNG string.
    """
    q = _resize_to_height(query_img, target_height)
    c = _resize_to_height(candidate_img, target_height)

    orb = cv2.ORB_create(nfeatures=500)
    kp_q, desc_q = orb.detectAndCompute(cv2.cvtColor(q, cv2.COLOR_BGR2GRAY), None)
    kp_c, desc_c = orb.detectAndCompute(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), None)

    if desc_q is None or desc_c is None or len(kp_q) < 2 or len(kp_c) < 2:
        out = _side_by_side(q, c)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw = bf.knnMatch(desc_q, desc_c, k=2)
        good = [m for m, n in raw if m.distance < 0.75 * n.distance]
        good = sorted(good, key=lambda m: m.distance)[:max_matches]

        out = cv2.drawMatches(
            q, kp_q, c, kp_c, good, None,
            matchColor=(0, 200, 0),
            singlePointColor=(180, 180, 180),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    return _to_base64(out)


def draw_score_bars(result: MatchResult, title: str | None = None) -> str:
    """
    Render a horizontal bar chart showing the weighted score breakdown.

    Each bar is color-coded by component:
        hist   — blue
        orb    — green
        hu     — orange
        corner — purple

    Parameters
    ----------
    result : MatchResult from Matcher.match()
    title  : optional heading (defaults to product name + overall score)

    Returns
    -------
    Base64-encoded PNG string.
    """
    canvas = _build_score_canvas(result, title)
    return _to_base64(canvas)


# ---------------------------------------------------------------------------
# Score-bar rendering
# ---------------------------------------------------------------------------

# BGR colours for each feature component
_BAR_COLORS: dict[str, tuple[int, int, int]] = {
    "contour":   ( 20, 180,  80),   # teal    — primary shape channel
    "hist":      (200,  80,  20),   # blue-ish
    "orb":       ( 40, 160,  40),   # green
    "hu":        ( 20, 140, 220),   # orange
    "corner":    (160,  40, 160),   # purple
    "shape_geo": ( 30, 100, 220),   # amber   — solidity/elongation/extent
    "bow":       (180,  60,  60),   # steel-blue — Bag-of-Visual-Words
}

_LABELS = {
    "contour":   "Shape (Fourier)",
    "hist":      "Color (HSV)",
    "orb":       "ORB",
    "hu":        "Shape (Hu)",
    "corner":    "Corners",
    "shape_geo": "Shape Geo",
    "bow":       "BoVW",
}

_CANVAS_W    = 520
_PAD         = 20          # outer padding
_LABEL_W     = 100         # fixed width reserved for the row label
_BAR_MAX_W   = 300         # max bar width (= score 1.0)
_BAR_H       = 28          # height of each bar
_ROW_GAP     = 14          # vertical gap between rows
_HEADER_H    = 52          # space above the first row (title + overall score)
_FONT        = cv2.FONT_HERSHEY_SIMPLEX


def _build_score_canvas(result: MatchResult, title: str | None) -> np.ndarray:
    components = ["contour", "hist", "hu", "shape_geo", "bow", "orb", "corner"]
    n = len(components)
    canvas_h = _PAD + _HEADER_H + n * (_BAR_H + _ROW_GAP) + _PAD

    canvas = np.full((canvas_h, _CANVAS_W, 3), 245, dtype=np.uint8)  # off-white bg

    # --- Header ---
    heading = title or (result.name if result.name else f"Product {result.product_id}")
    cv2.putText(canvas, heading, (_PAD, _PAD + 18),
                _FONT, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    score_text = f"Overall score: {result.score:.1%}"
    cv2.putText(canvas, score_text, (_PAD, _PAD + 40),
                _FONT, 0.48, (60, 60, 60), 1, cv2.LINE_AA)

    # Separator line
    sep_y = _PAD + _HEADER_H - 6
    cv2.line(canvas, (_PAD, sep_y), (_CANVAS_W - _PAD, sep_y), (180, 180, 180), 1)

    # --- Bars ---
    bar_x0 = _PAD + _LABEL_W
    for i, key in enumerate(components):
        row_y = _PAD + _HEADER_H + i * (_BAR_H + _ROW_GAP)
        sim_val = result.breakdown.get(key, 0.0)
        bar_w = int(round(_BAR_MAX_W * sim_val))
        color = _BAR_COLORS[key]

        # Background track
        track_y = row_y + (_BAR_H - 14) // 2
        cv2.rectangle(
            canvas,
            (bar_x0, track_y),
            (bar_x0 + _BAR_MAX_W, track_y + 14),
            (210, 210, 210), -1,
        )
        # Filled bar
        if bar_w > 0:
            cv2.rectangle(
                canvas,
                (bar_x0, track_y),
                (bar_x0 + bar_w, track_y + 14),
                color, -1,
            )

        # Row label (vertically centred on bar)
        label = _LABELS[key]
        cv2.putText(canvas, label, (_PAD, track_y + 11),
                    _FONT, 0.40, (50, 50, 50), 1, cv2.LINE_AA)

        # Percentage text to the right of the bar
        pct_text = f"{sim_val:.0%}"
        cv2.putText(canvas, pct_text,
                    (bar_x0 + _BAR_MAX_W + 8, track_y + 11),
                    _FONT, 0.40, (50, 50, 50), 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
    """Proportionally resize *img* so its height equals *h*."""
    ih, iw = img.shape[:2]
    if ih == h:
        return img
    scale = h / ih
    new_w = max(1, int(round(iw * scale)))
    return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)


def _side_by_side(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Horizontally concatenate two same-height images with a 2-px divider."""
    divider = np.full((img_a.shape[0], 2, 3), 120, dtype=np.uint8)
    return np.concatenate([img_a, divider, img_b], axis=1)


def _to_base64(img: np.ndarray) -> str:
    """Encode a BGR numpy array as a base64 PNG string."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")
