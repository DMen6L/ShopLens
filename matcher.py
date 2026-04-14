"""
matcher.py — Multi-feature product matching with weighted score re-ranking.

Flow
----
1. VectorStore ANN search (HSV embedding) → candidate shortlist
2. Fetch full features for each candidate from ProductDB
3. Re-rank with a weighted combination of four similarity scores:
       w_hist  * hist_sim    (HSV color histogram — Bhattacharyya)
     + w_sift  * sift_sim    (SIFT keypoint matches — Lowe ratio test)
     + w_hu    * hu_sim      (Hu shape moments — L2 distance)
     + w_corner* corner_sim  (Harris corner density — absolute diff)

Weights (default)
-----------------
    hist   0.40  — color is the strongest discriminator for products
    sift   0.30  — structural/keypoint evidence
    hu     0.20  — silhouette shape
    corner 0.10  — texture density

All individual similarity scores are in [0, 1] (1 = identical).
The final `score` is also in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    product_id: int
    score: float                          # weighted final score [0, 1]
    hist_sim: float                       # HSV histogram similarity [0, 1]
    sift_sim: float                       # SIFT match ratio [0, 1]
    hu_sim: float                         # Hu-moment similarity [0, 1]
    corner_sim: float                     # corner-density similarity [0, 1]
    name: str = ""                        # filled in by Matcher if DB row available
    breakdown: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.breakdown = {
            "hist": self.hist_sim,
            "sift": self.sift_sim,
            "hu": self.hu_sim,
            "corner": self.corner_sim,
        }


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------

class Matcher:
    """
    Parameters
    ----------
    db          : ProductDB instance (used to fetch candidate features)
    vector_store: VectorStore instance (used for ANN shortlist)
    weights     : dict with keys hist/sift/hu/corner — must sum to 1.0
    ann_factor  : how many extra candidates to fetch before re-ranking
                  (top_k * ann_factor vectors are retrieved from ANN)
    """

    _DEFAULT_WEIGHTS = {
        "hist":   0.40,
        "sift":   0.30,
        "hu":     0.20,
        "corner": 0.10,
    }

    def __init__(
        self,
        db: Any,
        vector_store: Any,
        weights: dict[str, float] | None = None,
        ann_factor: int = 3,
    ):
        self._db = db
        self._vs = vector_store
        self._w = weights or self._DEFAULT_WEIGHTS
        self._ann_factor = ann_factor

        # BFMatcher for SIFT (L2 + kNN for Lowe ratio test)
        self._bf = cv2.BFMatcher(cv2.NORM_L2)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def match(
        self,
        query_features: dict[str, Any],
        top_k: int = 5,
    ) -> list[MatchResult]:
        """
        Find the top-k most similar products for a query feature dict.

        Parameters
        ----------
        query_features : output of FeaturePipeline.extract()
        top_k          : number of results to return

        Returns
        -------
        List of MatchResult sorted by score descending (best first).
        """
        # 1. ANN candidate shortlist
        candidates_ann = self._vs.search(
            query_features["embedding"],
            top_k=top_k * self._ann_factor,
        )
        if not candidates_ann:
            return []

        candidate_ids = [pid for pid, _ in candidates_ann]

        # 2. Fetch full features for each candidate
        candidate_rows = {}
        for pid in candidate_ids:
            row = self._db.get_by_id(pid)
            if row is not None:
                candidate_rows[pid] = row

        # 3. Score each candidate
        results: list[MatchResult] = []
        for pid, row in candidate_rows.items():
            r = self._score_pair(query_features, row)
            r.product_id = pid
            r.name = row.get("name", "")
            results.append(r)

        # 4. Sort by final score descending and return top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _score_pair(
        self,
        q: dict[str, Any],
        c: dict[str, Any],
    ) -> MatchResult:
        hist_sim   = _hist_similarity(q["hist_hsv"],      c["hist_hsv"])
        sift_sim   = _sift_similarity(q["sift_desc"],     c["sift_desc"], self._bf)
        hu_sim     = _hu_similarity(q["hu_moments"],      c["hu_moments"])
        corner_sim = _corner_similarity(q["corner_density"], c["corner_density"])

        score = (
            self._w["hist"]   * hist_sim
            + self._w["sift"]   * sift_sim
            + self._w["hu"]     * hu_sim
            + self._w["corner"] * corner_sim
        )

        return MatchResult(
            product_id=0,       # filled by caller
            score=float(np.clip(score, 0.0, 1.0)),
            hist_sim=hist_sim,
            sift_sim=sift_sim,
            hu_sim=hu_sim,
            corner_sim=corner_sim,
        )


# ---------------------------------------------------------------------------
# Per-feature similarity functions
# ---------------------------------------------------------------------------

def _hist_similarity(h_q: np.ndarray, h_c: np.ndarray) -> float:
    """
    Bhattacharyya-based histogram similarity.

    cv2.compareHist(HISTCMP_BHATTACHARYYA) returns 0 for identical histograms
    and approaches 1 as they diverge.  We invert to get a [0, 1] similarity.
    """
    # Ensure same shape and type
    hq = h_q.astype(np.float32)
    hc = h_c.astype(np.float32)
    # Normalize in case they were stored differently
    cv2.normalize(hq, hq, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hc, hc, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    dist = cv2.compareHist(hq, hc, cv2.HISTCMP_BHATTACHARYYA)
    return float(np.clip(1.0 - dist, 0.0, 1.0))


def _sift_similarity(
    desc_q: np.ndarray | None,
    desc_c: np.ndarray | None,
    bf: cv2.BFMatcher,
    ratio: float = 0.75,
) -> float:
    """
    Lowe ratio-test match count, normalized to [0, 1].

    Returns 0 if either descriptor set is None or too small.
    Score = good_matches / max(n_q, n_c)  — penalises mismatched descriptor counts.
    """
    if desc_q is None or desc_c is None:
        return 0.0
    if len(desc_q) < 2 or len(desc_c) < 2:
        return 0.0

    matches = bf.knnMatch(desc_q, desc_c, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]
    denom = max(len(desc_q), len(desc_c))
    return float(np.clip(len(good) / denom, 0.0, 1.0))


def _hu_similarity(hu_q: np.ndarray, hu_c: np.ndarray) -> float:
    """
    Shape similarity via L2 distance on log-transformed Hu moments.

    Converts distance → similarity with  sim = 1 / (1 + dist).
    """
    dist = float(np.linalg.norm(hu_q.astype(np.float64) - hu_c.astype(np.float64)))
    return 1.0 / (1.0 + dist)


def _corner_similarity(density_q: float, density_c: float) -> float:
    """
    Corner-density similarity: 1 minus the absolute normalised difference.
    Both densities are already in [0, 1].
    """
    return float(np.clip(1.0 - abs(density_q - density_c), 0.0, 1.0))
