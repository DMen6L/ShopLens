"""
ShopLens — FastAPI application entry point.

Phase 6: Full API endpoints for product registration and querying.

Endpoints
---------
GET  /health                  — liveness probe
GET  /products                — list all products (metadata only)
POST /products/register       — register a new product with image upload
POST /products/query          — find similar products for a query image
DELETE /products/{product_id} — remove a product
"""

import base64
from contextlib import asynccontextmanager

import cv2
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from database import ProductDB
from matcher import Matcher
from pipeline import FeaturePipeline
from segmentation import grabcut_foreground, watershed_segment
from vector_store import make_vector_store
from visualizer import draw_matches, draw_score_bars

# Shared state — populated during lifespan startup
db: ProductDB | None = None
vs = None
pipeline: FeaturePipeline | None = None
matcher: Matcher | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    global db, vs, pipeline, matcher
    db = ProductDB()
    db.connect()
    vs = make_vector_store(db)
    pipeline = FeaturePipeline()
    matcher = Matcher(db, vs)
    yield
    db.close()


app = FastAPI(title="ShopLens", lifespan=lifespan)

# Serve frontend static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image — unsupported format")
    return img


def _segment(img: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Run foreground segmentation.

    Falls back to Watershed if GrabCut produces an empty mask or raises.
    """
    if method == "watershed":
        return watershed_segment(img)
    try:
        mask, masked = grabcut_foreground(img)
        if np.count_nonzero(mask) == 0:
            return watershed_segment(img)
        return mask, masked
    except Exception:
        return watershed_segment(img)


def _encode_png(img: np.ndarray) -> str:
    """Encode a BGR numpy array as a base64 PNG string."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Products — list
# ---------------------------------------------------------------------------

@app.get("/products")
def list_products():
    """Return all registered products (id, name, registered_at)."""
    rows = db.list_products()
    # registered_at is a datetime; convert to ISO string for JSON
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "registered_at": r["registered_at"].isoformat() if r.get("registered_at") else None,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Products — register
# ---------------------------------------------------------------------------

@app.post("/products/register", status_code=201)
async def register_product(
    name: str = Form(...),
    file: UploadFile = File(...),
    seg_method: str = Form("grabcut"),
):
    """
    Register a new product.

    Multipart form fields:
      - name        : human-readable product name
      - file        : image file (JPEG / PNG / WebP / etc.)
      - seg_method  : "grabcut" (default) or "watershed"

    Returns:
      {
        "product_id": <int>,
        "masked_img": "<base64 PNG of the segmented foreground>"
      }
    """
    raw = await file.read()
    img = _decode_image(raw)
    mask, masked_img = _segment(img, seg_method)

    features = pipeline.extract(img, mask)
    product_id = db.add(name, features, image_data=raw)
    vs.upsert(product_id, features["embedding"])

    return {
        "product_id": product_id,
        "masked_img": _encode_png(masked_img),
    }


# ---------------------------------------------------------------------------
# Products — query
# ---------------------------------------------------------------------------

@app.post("/products/query")
async def query_products(
    file: UploadFile = File(...),
    top_k: int = Form(5),
    seg_method: str = Form("grabcut"),
):
    """
    Find the top-k most similar registered products for a query image.

    Multipart form fields:
      - file        : query image
      - top_k       : number of results to return (default 5)
      - seg_method  : "grabcut" (default) or "watershed"

    Returns:
      {
        "results": [
          {
            "product_id": <int>,
            "name": <str>,
            "score": <float 0–1>,
            "breakdown": { "hist": f, "sift": f, "hu": f, "corner": f },
            "match_img": "<base64 PNG — side-by-side with SIFT lines>",
            "score_bars_img": "<base64 PNG — confidence bar chart>"
          },
          ...
        ]
      }
    """
    raw = await file.read()
    query_img = _decode_image(raw)
    mask, _ = _segment(query_img, seg_method)
    query_features = pipeline.extract(query_img, mask)

    match_results = matcher.match(query_features, top_k=top_k)

    output = []
    for r in match_results:
        # Retrieve stored image for visualization
        candidate_row = db.get_by_id(r.product_id)
        candidate_img = None
        if candidate_row and candidate_row.get("image_data"):
            try:
                candidate_img = _decode_image(bytes(candidate_row["image_data"]))
            except HTTPException:
                pass

        if candidate_img is not None:
            match_img_b64 = draw_matches(query_img, candidate_img)
        else:
            # No stored image — fall back to encoded query image
            match_img_b64 = _encode_png(query_img)

        output.append({
            "product_id": r.product_id,
            "name": r.name,
            "score": round(r.score, 4),
            "breakdown": {k: round(v, 4) for k, v in r.breakdown.items()},
            "match_img": match_img_b64,
            "score_bars_img": draw_score_bars(r),
        })

    return {"results": output}


# ---------------------------------------------------------------------------
# Products — delete
# ---------------------------------------------------------------------------

@app.delete("/products/{product_id}")
def delete_product(product_id: int):
    """
    Delete a product and all its associated features.

    Returns { "deleted": <product_id> } on success, 404 if not found.
    """
    deleted = db.delete(product_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
    vs.delete(product_id)
    return {"deleted": product_id}


# ---------------------------------------------------------------------------
# Frontend root
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse("frontend/index.html")
