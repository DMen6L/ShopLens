"""
ProductDB — thin wrapper around PostgreSQL via psycopg2.

Responsibilities:
- Store / retrieve product metadata (products table)
- Store / retrieve serialized feature blobs (features table)
- Each product can have multiple views (one features row per image/angle)
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


def _unpack_sift_blob(
    blob: memoryview | bytes | None,
) -> tuple[list, np.ndarray | None]:
    """
    Deserialize the sift blob stored in the DB.

    Format: pickle of (kp_coords: float32 (N,2), desc: float32 (N,128))

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

    def create_product(self, name: str, image_data: bytes | None = None) -> int:
        """
        Create a new product record (metadata only — no features).

        image_data : raw bytes of the primary / thumbnail image.

        Returns the new product id.
        """
        with self.conn:
            with self.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO products (name, image_data) VALUES (%s, %s) RETURNING id",
                    (name, psycopg2.Binary(image_data) if image_data else None),
                )
                return cur.fetchone()[0]

    def add_view(
        self,
        product_id: int,
        features: dict[str, Any],
        view_image_data: bytes | None = None,
    ) -> int:
        """
        Add one feature view (image + extracted features) to an existing product.

        Call once per registered angle; a product may have any number of views.

        features keys expected:
            sift_desc       np.ndarray (N, 128) float32  — may be None
            hist_hsv        np.ndarray (50, 60) float32
            hu_moments      np.ndarray (7,)     float64
            corner_density  float
            embedding       np.ndarray (32,)    float32  — L2-normed Fourier shape descriptor

        view_image_data : raw bytes of the image for this specific view/angle.

        Returns the new features row id (view_id), used later to retrieve the
        best-matching view's image during query visualization.
        """
        sift_desc = features.get("sift_desc")
        sift_kp   = features.get("sift_kp")
        if sift_desc is not None:
            kp_coords = (
                np.float32([kp.pt for kp in sift_kp])
                if sift_kp else np.empty((0, 2), dtype=np.float32)
            )
            sift_blob = pickle.dumps((kp_coords, sift_desc))
        else:
            sift_blob = None

        hist_blob = pickle.dumps(features["hist_hsv"])
        hu = features["hu_moments"].tolist()
        corner = float(features["corner_density"])
        embedding = features["embedding"].tolist()

        shape_geo = features.get("shape_geo")
        shape_geo_blob = pickle.dumps(shape_geo) if shape_geo is not None else None

        with self.conn:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO features
                        (product_id, sift_desc, hist_hsv, hu_moments, corner_density,
                         embedding, shape_geo, view_image_data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        product_id,
                        psycopg2.Binary(sift_blob) if sift_blob else None,
                        psycopg2.Binary(hist_blob),
                        hu, corner, embedding,
                        psycopg2.Binary(shape_geo_blob) if shape_geo_blob else None,
                        psycopg2.Binary(view_image_data) if view_image_data else None,
                    ),
                )
                return cur.fetchone()[0]

    def get_views(self, product_id: int) -> list[dict[str, Any]]:
        """
        Return all feature views for a product, with deserialized features.

        Each dict has:
            view_id        int           — features.id (use to fetch view image)
            name           str           — product name
            image_data     bytes | None  — raw image bytes for this view
            sift_kp        list[KeyPoint]
            sift_desc      np.ndarray | None
            hist_hsv       np.ndarray (50, 60)
            hu_moments     np.ndarray (7,)
            corner_density float
            embedding      np.ndarray (32,)
            shape_geo      np.ndarray (3,)
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT p.name,
                       f.id AS view_id,
                       f.sift_desc, f.hist_hsv, f.hu_moments,
                       f.corner_density, f.embedding, f.shape_geo,
                       f.view_image_data AS image_data
                FROM   products p
                JOIN   features  f ON f.product_id = p.id
                WHERE  p.id = %s
                ORDER  BY f.id
                """,
                (product_id,),
            )
            rows = cur.fetchall()

        result = []
        for row in rows:
            d = dict(row)
            d["sift_kp"], d["sift_desc"] = _unpack_sift_blob(d.pop("sift_desc"))
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

    def get_view_image(self, feature_id: int) -> bytes | None:
        """Return the raw image bytes stored for a specific feature view."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT view_image_data FROM features WHERE id = %s",
                (feature_id,),
            )
            row = cur.fetchone()
        if row is None or row[0] is None:
            return None
        return bytes(row[0])

    def get_by_id(self, product_id: int) -> dict[str, Any] | None:
        """Return product metadata (id, name, image_data, registered_at)."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, name, image_data, registered_at FROM products WHERE id = %s",
                (product_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return dict(row)

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
