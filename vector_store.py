"""
VectorStore — ANN search over product embeddings.

Supports two backends selected via the VECTOR_BACKEND env var:
  - "pgvector"  (default) — uses the embedding column in PostgreSQL
  - "qdrant"              — uses an external Qdrant service

Public API (same for both backends):
    upsert(product_id, embedding)  — add or replace a vector
    search(embedding, top_k)       — return [(product_id, score), ...]
    delete(product_id)             — remove a vector
"""

import os
from typing import Any

import numpy as np


BACKEND = os.environ.get("VECTOR_BACKEND", "pgvector").lower()
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "shoplens_products"
EMBEDDING_DIM = 32


# ---------------------------------------------------------------------------
# pgvector backend
# ---------------------------------------------------------------------------

class _PgVectorStore:
    def __init__(self, db: Any):
        """db is a ProductDB instance (shares the same connection)."""
        self._db = db

    def upsert(self, feature_id: int, product_id: int, embedding: np.ndarray) -> None:
        # pgvector: embedding is already stored in the features row by add_view().
        # No additional write is needed; this is intentionally a no-op.
        pass

    def search(self, embedding: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        """
        Return the top-k most similar *products* (deduplicated).

        When a product has multiple views, the best-matching view's score is
        used to represent the product (MIN distance = MAX similarity).
        """
        vec = _normalize(embedding).tolist()
        with self._db.conn.cursor() as cur:
            cur.execute(
                """
                SELECT product_id,
                       1 - MIN(embedding <-> %s::vector) AS score
                FROM   features
                GROUP  BY product_id
                ORDER  BY MIN(embedding <-> %s::vector)
                LIMIT  %s
                """,
                (vec, vec, top_k),
            )
            return [(row[0], float(row[1])) for row in cur.fetchall()]

    def delete(self, product_id: int) -> None:
        # Embedding is deleted via CASCADE when the product row is removed;
        # this is a no-op for pgvector (the features row carries the embedding).
        pass


# ---------------------------------------------------------------------------
# Qdrant backend
# ---------------------------------------------------------------------------

class _QdrantStore:
    def __init__(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = QdrantClient(url=QDRANT_URL)
        # Ensure collection exists
        existing = [c.name for c in self._client.get_collections().collections]
        if QDRANT_COLLECTION not in existing:
            self._client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )

    def upsert(self, feature_id: int, product_id: int, embedding: np.ndarray) -> None:
        from qdrant_client.models import PointStruct

        vec = _normalize(embedding).tolist()
        self._client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[PointStruct(id=feature_id, vector=vec, payload={"product_id": product_id})],
        )

    def search(self, embedding: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        """
        Return top-k products (deduplicated by product_id).

        Fetches top_k * 10 hits then keeps only the best score per product_id.
        """
        vec = _normalize(embedding).tolist()
        hits = self._client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vec,
            limit=top_k * 10,
            with_payload=True,
        )
        # Deduplicate: keep best score per product_id
        best: dict[int, float] = {}
        for hit in hits:
            pid = hit.payload["product_id"]
            if pid not in best or hit.score > best[pid]:
                best[pid] = hit.score
        return sorted(best.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def delete(self, product_id: int) -> None:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        self._client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="product_id", match=MatchValue(value=product_id))]
            ),
        )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def make_vector_store(db: Any = None):
    """
    Returns the appropriate VectorStore backend.

    Pass `db` (a ProductDB instance) when using pgvector.
    """
    if BACKEND == "qdrant":
        return _QdrantStore()
    return _PgVectorStore(db)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(vec: np.ndarray) -> np.ndarray:
    v = vec.astype(np.float32).ravel()
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v
