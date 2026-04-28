from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matcher import (
    _bow_similarity,
    _fourier_similarity,
    _shape_geo_similarity,
    _sift_similarity,
)


def test_fourier_identical_scores_high() -> None:
    vec = np.zeros(32, dtype=np.float32)
    vec[0] = 1.0

    score = _fourier_similarity(vec, vec)

    assert score == 1.0


def test_fourier_moderate_difference_scores_strictly() -> None:
    query = np.zeros(32, dtype=np.float32)
    candidate = np.zeros(32, dtype=np.float32)
    query[0] = 1.0
    candidate[0] = 0.95
    candidate[1] = np.sqrt(1.0 - candidate[0] ** 2)

    score = _fourier_similarity(query, candidate)

    assert 0.40 < score < 0.50


def test_shape_geo_identical_scores_high() -> None:
    geo = np.array([0.8, 0.4, 0.4], dtype=np.float32)
    geo = geo / np.linalg.norm(geo)

    score = _shape_geo_similarity(geo, geo)

    assert score == 1.0


def test_shape_geo_distinct_scores_low() -> None:
    query = np.array([0.9, 0.1, 0.4], dtype=np.float32)
    candidate = np.array([0.5, 0.75, 0.43], dtype=np.float32)
    query = query / np.linalg.norm(query)
    candidate = candidate / np.linalg.norm(candidate)

    score = _shape_geo_similarity(query, candidate)

    assert score < 0.03


def test_bow_identical_histograms_score_high() -> None:
    hist = np.array([0.0, 0.6, 0.8], dtype=np.float32)

    score = _bow_similarity(hist, hist)

    assert score == 1.0


def test_bow_missing_histogram_scores_zero() -> None:
    hist = np.array([0.0, 0.6, 0.8], dtype=np.float32)

    score = _bow_similarity(hist, None)

    assert score == 0.0


def test_bow_different_histograms_score_below_identical() -> None:
    query = np.array([0.0, 0.6, 0.8], dtype=np.float32)
    candidate = np.array([0.8, 0.6, 0.0], dtype=np.float32)

    score = _bow_similarity(query, candidate)

    assert 0.0 < score < 1.0


def _grid_keypoints(count: int, dx: float = 0.0, dy: float = 0.0) -> list[cv2.KeyPoint]:
    return [
        cv2.KeyPoint(x=float((i % 8) * 10 + dx), y=float((i // 8) * 10 + dy), size=1.0)
        for i in range(count)
    ]


def _distinct_descriptors(count: int) -> np.ndarray:
    desc = np.zeros((count, 128), dtype=np.float32)
    for i in range(count):
        desc[i, i % 128] = 100.0
    return desc


def test_sift_strong_geometric_match_scores_high() -> None:
    desc = _distinct_descriptors(20)
    kp_q = _grid_keypoints(20)
    kp_c = _grid_keypoints(20, dx=3.0, dy=4.0)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    score = _sift_similarity(desc, desc.copy(), bf, kp_q=kp_q, kp_c=kp_c)

    assert score > 0.80


def test_sift_three_inlier_support_scores_zero() -> None:
    desc_q = _distinct_descriptors(40)
    desc_c = np.zeros((40, 128), dtype=np.float32)
    desc_c[:3] = desc_q[:3]
    kp_q = _grid_keypoints(40)
    kp_c = _grid_keypoints(40, dx=3.0, dy=4.0)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    score = _sift_similarity(desc_q, desc_c, bf, kp_q=kp_q, kp_c=kp_c)

    assert score == 0.0


if __name__ == "__main__":
    test_fourier_identical_scores_high()
    test_fourier_moderate_difference_scores_strictly()
    test_shape_geo_identical_scores_high()
    test_shape_geo_distinct_scores_low()
    test_bow_identical_histograms_score_high()
    test_bow_missing_histogram_scores_zero()
    test_bow_different_histograms_score_below_identical()
    test_sift_strong_geometric_match_scores_high()
    test_sift_three_inlier_support_scores_zero()
