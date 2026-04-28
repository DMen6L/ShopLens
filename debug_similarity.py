"""
Print local feature-similarity breakdown for two image files.

Usage:
    uv run python debug_similarity.py query.jpg candidate.jpg

This bypasses the database and vector store. It is meant for calibrating the
same per-feature scores that appear in the ShopLens query breakdown.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from matcher import Matcher, _build_bow_vocab, _encode_bow
from pipeline import FeaturePipeline
from segmentation import grabcut_foreground, watershed_segment


def _load_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not decode image: {path}")
    return img


def _segment(img, method: str):
    if method == "watershed":
        return watershed_segment(img)
    try:
        mask, masked = grabcut_foreground(img)
        if cv2.countNonZero(mask) == 0:
            return watershed_segment(img)
        return mask, masked
    except Exception:
        return watershed_segment(img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--seg-method",
        choices=("grabcut", "watershed"),
        default="grabcut",
        help="Segmentation method to use before feature extraction.",
    )
    args = parser.parse_args()

    pipeline = FeaturePipeline()
    matcher = Matcher(db=None, vector_store=None)

    query_img = _load_image(args.query)
    candidate_img = _load_image(args.candidate)
    query_mask, _ = _segment(query_img, args.seg_method)
    candidate_mask, _ = _segment(candidate_img, args.seg_method)

    query_features = pipeline.extract(query_img, query_mask)
    candidate_features = pipeline.extract(candidate_img, candidate_mask)
    vocab = _build_bow_vocab(
        [query_features.get("sift_desc"), candidate_features.get("sift_desc")]
    )
    query_bow = _encode_bow(query_features.get("sift_desc"), vocab)
    candidate_bow = _encode_bow(candidate_features.get("sift_desc"), vocab)
    result = matcher._score_pair(
        query_features,
        candidate_features,
        query_bow=query_bow,
        cand_bow=candidate_bow,
    )

    print(f"query:     {args.query}")
    print(f"candidate: {args.candidate}")
    print(f"overall:   {result.score:.4f}")
    for key, value in result.breakdown.items():
        print(f"{key:10s} {value:.4f}")


if __name__ == "__main__":
    main()
