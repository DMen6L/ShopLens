"""
matcher.py — Multi-feature product matching with weighted score re-ranking.

Flow
----
1. VectorStore ANN search (Fourier shape embedding) → candidate shortlist
   The ANN stage is now shape-driven: similar-shaped products are retrieved
   first, so logo/colour lookalikes that differ in silhouette are deprioritised
   before re-ranking even starts.

2. Fetch full features for each candidate from ProductDB.

3. Re-rank with a weighted combination of similarity scores:
       w_contour * contour_sim   (Fourier shape descriptor — primary)
     + w_hu      * hu_sim        (Hu moments — secondary shape)
     + w_shape_geo * shape_geo_sim (solidity/elongation/extent — primary shape)
     + w_hist    * hist_sim      (HSV colour histogram — secondary)
     + w_sift    * sift_sim      (SIFT + RANSAC — optional structural)
     + w_corner  * corner_sim    (Harris corner density — tie-breaker)

Weights (default)
-----------------
    contour   0.13  — whole-object Fourier silhouette; logo-blind
    hist      0.12  — colour; tertiary, must not dominate
    hu        0.18  — Hu moments; secondary shape descriptor
    shape_geo 0.38  — solidity/elongation/extent; primary shape discriminator
    bow       0.12  — Bag-of-Visual-Words; local texture patterns
    sift      0.04  — boundary-ring SIFT; logo-suppressed structural evidence
    corner    0.03  — boundary-ring Harris density (tie-breaker only)

All individual similarity scores are in [0, 1] (1 = identical).
The final `score` is also in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# Controls how sharply Fourier similarity falls off with shape distance.
# Higher = more discriminating (penalises small differences harder).
# At 3.0: cos_sim 0.95→0.74, 0.90→0.55, 0.85→0.41, 0.80→0.30
_FOURIER_SHARPNESS = 3.0

# Shape geometry: scaling factor k in  1 / (1 + k * dist).
# At k=3: L2 dist 0.0→1.0, 0.3→0.53, 0.5→0.40, 1.0→0.25
_GEO_K = 3.0

# BoVW: number of visual words for k-means vocabulary.
# Built on-the-fly from the ANN candidate pool — no persistent vocab file needed.
_BOW_K = 50


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    product_id: int
    score: float                          # weighted final score [0, 1]
    contour_sim: float                    # Fourier shape descriptor similarity [0, 1]
    hist_sim: float                       # HSV histogram similarity [0, 1]
    sift_sim: float                       # SIFT match ratio [0, 1]
    hu_sim: float                         # Hu-moment similarity [0, 1]
    corner_sim: float                     # corner-density similarity [0, 1]
    shape_geo_sim: float = 0.0            # shape geometry similarity [0, 1]
    bow_sim: float = 0.0                  # Bag-of-Visual-Words similarity [0, 1]
    view_id: int = 0                      # features.id of the best-matching view
    name: str = ""                        # filled in by Matcher if DB row available
    breakdown: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.breakdown = {
            "contour":   self.contour_sim,
            "hist":      self.hist_sim,
            "sift":      self.sift_sim,
            "hu":        self.hu_sim,
            "corner":    self.corner_sim,
            "shape_geo": self.shape_geo_sim,
            "bow":       self.bow_sim,
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
    weights     : dict with keys contour/hist/hu/shape_geo/bow/sift/corner — must sum to 1.0
    ann_factor  : how many extra candidates to fetch before re-ranking
                  (top_k * ann_factor vectors are retrieved from ANN)
    """

    _DEFAULT_WEIGHTS = {
        "contour":   0.13,  # Fourier shape — global silhouette; logo-blind
        "hist":      0.12,  # colour — tertiary; must not dominate
        "hu":        0.18,  # Hu moments — secondary shape descriptor
        "shape_geo": 0.38,  # solidity/elongation/extent — primary shape discriminator
        "bow":       0.12,  # Bag-of-Visual-Words — local texture patterns
        "sift":      0.04,  # SIFT boundary-ring RANSAC — structural evidence
        "corner":    0.03,  # corner density — tie-breaker only
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

        # BFMatcher for SIFT (L2 distance + kNN for Lowe ratio test)
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
        # 1. ANN candidate shortlist (shape-based, deduplicated by product_id)
        candidates_ann = self._vs.search(
            query_features["embedding"],
            top_k=top_k * self._ann_factor,
        )
        if not candidates_ann:
            return []

        candidate_ids = [pid for pid, _ in candidates_ann]

        # 2. Fetch ALL views for each candidate product
        candidate_rows: dict[int, list] = {}
        for pid in candidate_ids:
            views = self._db.get_views(pid)
            if views:
                candidate_rows[pid] = views

        # 3. Build BoVW vocabulary on-the-fly from query + all views of all candidates.
        all_descs_for_vocab = [query_features.get("sift_desc")]
        for views in candidate_rows.values():
            for v in views:
                all_descs_for_vocab.append(v.get("sift_desc"))
        vocab = _build_bow_vocab(all_descs_for_vocab, k=_BOW_K)

        query_bow = _encode_bow(query_features.get("sift_desc"), vocab)

        # 4. Score each view per product; keep the best-scoring view per product
        results: list[MatchResult] = []
        for pid, views in candidate_rows.items():
            best: MatchResult | None = None
            for view in views:
                view_bow = _encode_bow(view.get("sift_desc"), vocab)
                r = self._score_pair(query_features, view, query_bow=query_bow, cand_bow=view_bow)
                if best is None or r.score > best.score:
                    best = r
                    best.view_id = view["view_id"]
            best.product_id = pid
            best.name = views[0].get("name", "")
            results.append(best)

        # 5. Sort by final score descending and return top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _score_pair(
        self,
        q: dict[str, Any],
        c: dict[str, Any],
        query_bow: np.ndarray | None = None,
        cand_bow: np.ndarray | None = None,
    ) -> MatchResult:
        contour_sim   = _fourier_similarity(q["embedding"],        c["embedding"])
        hist_sim      = _hist_similarity(q["hist_hsv"],            c["hist_hsv"])
        sift_sim      = _sift_similarity(
            q["sift_desc"], c["sift_desc"], self._bf,
            kp_q=q.get("sift_kp"), kp_c=c.get("sift_kp"),
        )
        hu_sim        = _hu_similarity(q["hu_moments"],            c["hu_moments"])
        corner_sim    = _corner_similarity(q["corner_density"],    c["corner_density"])
        shape_geo_sim = _shape_geo_similarity(q.get("shape_geo"),  c.get("shape_geo"))
        bow_sim       = _bow_similarity(query_bow, cand_bow)

        score = (
            self._w["contour"]     * contour_sim
            + self._w["hist"]      * hist_sim
            + self._w["sift"]      * sift_sim
            + self._w["hu"]        * hu_sim
            + self._w["corner"]    * corner_sim
            + self._w["shape_geo"] * shape_geo_sim
            + self._w["bow"]       * bow_sim
        )

        return MatchResult(
            product_id=0,       # filled by caller
            score=float(np.clip(score, 0.0, 1.0)),
            contour_sim=contour_sim,
            hist_sim=hist_sim,
            sift_sim=sift_sim,
            hu_sim=hu_sim,
            corner_sim=corner_sim,
            shape_geo_sim=shape_geo_sim,
            bow_sim=bow_sim,
        )


# ---------------------------------------------------------------------------
# Per-feature similarity functions
# ---------------------------------------------------------------------------

def _fourier_similarity(fd_q: np.ndarray, fd_c: np.ndarray) -> float:
    """
    Gaussian kernel similarity between two L2-normalised Fourier shape descriptors.

    Cosine similarity (dot product) was used previously but clusters near 1.0 for
    all products of similar category (bottles, boxes, etc.) because their silhouettes
    are similar in the broad sense — scores were always 80%+ regardless of whether
    the products actually matched.

    Switching to an exponential kernel on the squared L2 distance spreads those
    scores into a useful range:
        same product  → dist_sq ≈ 0.00 → score ≈ 1.00
        similar shape → dist_sq ≈ 0.10 → score ≈ 0.74
        same category → dist_sq ≈ 0.20 → score ≈ 0.55
        different     → dist_sq ≈ 0.40 → score ≈ 0.30
        very different → dist_sq ≈ 0.70 → score ≈ 0.12

    _FOURIER_SHARPNESS controls how quickly scores fall off; higher = more peaky.
    """
    if fd_q is None or fd_c is None:
        return 0.0
    fq = fd_q.astype(np.float32).ravel()
    fc = fd_c.astype(np.float32).ravel()
    if len(fq) == 0 or len(fc) == 0:
        return 0.0
    dist_sq = float(np.sum((fq - fc) ** 2))   # both unit vectors → range [0, 4]
    return float(np.exp(-_FOURIER_SHARPNESS * dist_sq))


def _hist_similarity(h_q: np.ndarray, h_c: np.ndarray) -> float:
    """
    Bhattacharyya-based histogram similarity.

    cv2.compareHist(HISTCMP_BHATTACHARYYA) returns 0 for identical histograms
    and approaches 1 as they diverge.  We invert to get a [0, 1] similarity.
    """
    hq = h_q.astype(np.float32)
    hc = h_c.astype(np.float32)
    cv2.normalize(hq, hq, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hc, hc, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    dist = cv2.compareHist(hq, hc, cv2.HISTCMP_BHATTACHARYYA)
    return float(np.clip(1.0 - dist, 0.0, 1.0))


def _sift_similarity(
    desc_q: np.ndarray | None,
    desc_c: np.ndarray | None,
    bf: cv2.BFMatcher,
    ratio: float = 0.75,
    kp_q: list | None = None,
    kp_c: list | None = None,
) -> float:
    """
    SIFT similarity with optional RANSAC homography verification.

    Step 1 — Lowe ratio test on L2 distances to get candidate matches.
    Step 2 — If keypoints are available (≥4 matches), run findHomography(RANSAC)
             to keep only geometrically consistent inliers.

    Score = inliers / min(n_q, n_c)
    """
    if desc_q is None or desc_c is None:
        return 0.0
    if len(desc_q) < 2 or len(desc_c) < 2:
        return 0.0

    matches = bf.knnMatch(desc_q, desc_c, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]

    denom = min(len(desc_q), len(desc_c))

    if len(good) < 4 or kp_q is None or kp_c is None:
        return float(np.clip(len(good) / denom, 0.0, 1.0))

    pts_q = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_c = np.float32([kp_c[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    _, mask = cv2.findHomography(pts_q, pts_c, cv2.RANSAC, ransacReprojThreshold=5.0)
    if mask is None:
        return float(np.clip(len(good) / denom, 0.0, 1.0))

    inliers = int(mask.ravel().sum())
    return float(np.clip(inliers / denom, 0.0, 1.0))


def _hu_similarity(hu_q: np.ndarray, hu_c: np.ndarray) -> float:
    """Shape similarity via L2 distance on log-transformed Hu moments."""
    dist = float(np.linalg.norm(hu_q.astype(np.float64) - hu_c.astype(np.float64)))
    return 1.0 / (1.0 + dist)


def _corner_similarity(density_q: float, density_c: float) -> float:
    """Corner-density similarity: 1 minus the absolute normalised difference."""
    return float(np.clip(1.0 - abs(density_q - density_c), 0.0, 1.0))


def _shape_geo_similarity(geo_q: np.ndarray | None, geo_c: np.ndarray | None) -> float:
    """
    Shape geometry similarity (solidity, elongation, extent).

    Both vectors are L2-normalised unit vectors.  Uses the same
    1/(1 + k*dist) kernel as Hu moments; _GEO_K controls sharpness.

    Key discriminator: solidity separates jagged shapes (saw, ~0.7) from
    smooth ones (screwdriver, ~0.97) regardless of orientation.
    """
    if geo_q is None or geo_c is None:
        return 0.0
    gq = geo_q.astype(np.float32).ravel()
    gc = geo_c.astype(np.float32).ravel()
    if len(gq) == 0 or len(gc) == 0:
        return 0.0
    dist = float(np.linalg.norm(gq - gc))
    return 1.0 / (1.0 + _GEO_K * dist)


def _build_bow_vocab(desc_list: list, k: int = _BOW_K) -> np.ndarray | None:
    """
    Build a k-means visual vocabulary from a list of SIFT descriptor arrays.

    SIFT descriptors are already float32 (L2 space).

    Returns the (k, 128) float32 cluster centres, or None if there are too
    few descriptors to cluster.
    """
    all_descs = []
    for d in desc_list:
        if d is not None and len(d) > 0:
            all_descs.append(d.astype(np.float32))
    if not all_descs:
        return None
    combined = np.vstack(all_descs)
    # Need at least 2 × k samples for a meaningful vocabulary
    k = min(k, max(1, len(combined) // 2))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, vocab = cv2.kmeans(combined, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return vocab.astype(np.float32)


def _encode_bow(desc: np.ndarray | None, vocab: np.ndarray | None) -> np.ndarray | None:
    """
    Encode a descriptor array as an L2-normalised BoVW histogram.

    For each descriptor, find the nearest visual word (L2 in float32 space)
    and increment that bin.  The resulting histogram is L2-normalised.
    """
    if desc is None or len(desc) == 0 or vocab is None or len(vocab) == 0:
        return None
    desc_f = desc.astype(np.float32)          # (N, 128)
    # Squared L2 distances to every vocab word: (N, k)
    diffs = desc_f[:, None, :] - vocab[None, :, :]
    dists_sq = np.einsum("nkd,nkd->nk", diffs, diffs)
    assignments = np.argmin(dists_sq, axis=1)  # (N,)
    hist = np.bincount(assignments, minlength=len(vocab)).astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm
    return hist


def _bow_similarity(hist_q: np.ndarray | None, hist_c: np.ndarray | None) -> float:
    """
    Bag-of-Visual-Words similarity via chi-square distance.

    Chi-square distance is sensitive to the distribution shape, making it
    better than L2 for comparing normalised histograms.  We map it to [0,1]
    via 1/(1+chisq) so that identical histograms score 1.0.
    """
    if hist_q is None or hist_c is None:
        return 0.0
    hq = hist_q.astype(np.float64).ravel()
    hc = hist_c.astype(np.float64).ravel()
    if len(hq) != len(hc) or len(hq) == 0:
        return 0.0
    eps = 1e-10
    chisq = float(np.sum((hq - hc) ** 2 / (hq + hc + eps)))
    return float(np.clip(1.0 / (1.0 + chisq), 0.0, 1.0))
