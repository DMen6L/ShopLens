"""
ProductDB — thin wrapper around PostgreSQL via psycopg2.

Responsibilities:
- Store / retrieve product metadata (products table)
- Store / retrieve serialized feature blobs (features table)
- Delete a product and its features (CASCADE handled by FK)
"""

import os
import pickle
from typing import Any

import cv2
import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector


def _unpack_orb_blob(
    blob: memoryview | bytes | None,
) -> tuple[list, np.ndarray | None]:
    """
    Deserialize the orb blob stored in the DB.

    New format  : pickle of (kp_coords: float32 (N,2), desc: uint8 (N,32))
    Legacy format: pickle of just the descriptor array

    Returns (keypoints: list[cv2.KeyPoint], descriptors: np.ndarray | None).
    """
    if blob is None:
        return [], None
    payload = pickle.loads(bytes(blob))
    if isinstance(payload, tuple):
        kp_coords, desc = payload
        kp = [cv2.KeyPoint(x=float(x), y=float(y), size=1.0) for x, y in kp_coords]
        return kp, desc
    # Legacy: payload is just the descriptor array
    return [], payload


class ProductDB:
    def __init__(self, dsn: str | None = None):
        self._dsn = dsn or os.environ["DATABASE_URL"]
        self._conn: psycopg2.extensions.connection | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._conn = psycopg2.connect(self._dsn)
        self._conn.autocommit = False
        register_vector(self._conn)

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    @property
    def conn(self) -> psycopg2.extensions.connection:
        if self._conn is None or self._conn.closed:
            self.connect()
        return self._conn

    # ------------------------------------------------------------------
    # Products
    # ------------------------------------------------------------------

    def add(self, name: str, features: dict[str, Any], image_data: bytes | None = None) -> int:
        """
        Insert a product and its features atomically.

        features keys expected:
            orb_desc        np.ndarray (N, 32) uint8  — may be None
            hist_hsv        np.ndarray (50, 60) float32
            hu_moments      np.ndarray (7,)     float64
            corner_density  float
            embedding       np.ndarray (32,)    float32  — L2-normed Fourier shape descriptor

        image_data : raw bytes of the original uploaded image (for visualization).

        Returns the new product id.
        """
        with self.conn:
            with self.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO products (name, image_data) VALUES (%s, %s) RETURNING id",
                    (name, psycopg2.Binary(image_data) if image_data else None),
                )
                product_id: int = cur.fetchone()[0]

                orb_desc = features.get("orb_desc")
                orb_kp   = features.get("orb_kp")  # list[cv2.KeyPoint] or None
                if orb_desc is not None:
                    # Store keypoint (x, y) coords alongside descriptors so the
                    # matcher can run RANSAC homography verification at query time.
                    kp_coords = (
                        np.float32([kp.pt for kp in orb_kp])
                        if orb_kp else np.empty((0, 2), dtype=np.float32)
                    )
                    sift_blob = pickle.dumps((kp_coords, orb_desc))
                else:
                    sift_blob = None
                hist_blob = pickle.dumps(features["hist_hsv"])
                hu = features["hu_moments"].tolist()
                corner = float(features["corner_density"])
                embedding = features["embedding"].tolist()

                shape_geo = features.get("shape_geo")
                shape_geo_blob = pickle.dumps(shape_geo) if shape_geo is not None else None

                cur.execute(
                    """
                    INSERT INTO features
                        (product_id, orb_desc, hist_hsv, hu_moments, corner_density, embedding, shape_geo)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (product_id, psycopg2.Binary(sift_blob) if sift_blob else None,
                     psycopg2.Binary(hist_blob), hu, corner, embedding,
                     psycopg2.Binary(shape_geo_blob) if shape_geo_blob else None),
                )
        return product_id

    def get_all(self) -> list[dict[str, Any]]:
        """
        Return all products with their deserialized features.
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT p.id, p.name, p.image_data, p.registered_at,
                       f.orb_desc, f.hist_hsv, f.hu_moments,
                       f.corner_density, f.embedding, f.shape_geo
                FROM   products p
                JOIN   features  f ON f.product_id = p.id
                ORDER  BY p.id
                """
            )
            rows = cur.fetchall()

        result = []
        for row in rows:
            d = dict(row)
            d["orb_kp"], d["orb_desc"] = _unpack_orb_blob(d.pop("orb_desc"))
            d["hist_hsv"] = pickle.loads(bytes(d["hist_hsv"]))
            d["hu_moments"] = np.array(d["hu_moments"], dtype=np.float64)
            d["corner_density"] = float(d["corner_density"])
            d["embedding"] = np.array(d["embedding"], dtype=np.float32)
            raw_geo = d.get("shape_geo")
            d["shape_geo"] = (
                pickle.loads(bytes(raw_geo)) if raw_geo else np.zeros(3, dtype=np.float32)
            )
            result.append(d)

        return result

    def get_by_id(self, product_id: int) -> dict[str, Any] | None:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT p.id, p.name, p.image_data, p.registered_at,
                       f.orb_desc, f.hist_hsv, f.hu_moments,
                       f.corner_density, f.embedding, f.shape_geo
                FROM   products p
                JOIN   features  f ON f.product_id = p.id
                WHERE  p.id = %s
                """,
                (product_id,),
            )
            row = cur.fetchone()

        if row is None:
            return None

        d = dict(row)
        d["orb_kp"], d["orb_desc"] = _unpack_orb_blob(d.pop("orb_desc"))
        d["hist_hsv"] = pickle.loads(bytes(d["hist_hsv"]))
        d["hu_moments"] = np.array(d["hu_moments"], dtype=np.float64)
        d["corner_density"] = float(d["corner_density"])
        d["embedding"] = np.array(d["embedding"], dtype=np.float32)
        raw_geo = d.get("shape_geo")
        d["shape_geo"] = (
            pickle.loads(bytes(raw_geo)) if raw_geo else np.zeros(3, dtype=np.float32)
        )
        return d

    def delete(self, product_id: int) -> bool:
        """Delete a product (features cascade). Returns True if a row was deleted."""
        with self.conn:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM products WHERE id = %s", (product_id,))
                return cur.rowcount > 0

    def list_products(self) -> list[dict[str, Any]]:
        """Lightweight listing — metadata only, no feature blobs."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, name, registered_at FROM products ORDER BY id")
            return [dict(r) for r in cur.fetchall()]
