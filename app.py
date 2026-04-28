"""
ShopLens — FastAPI application entry point.

Phase 6: Full API endpoints for product registration and querying.

Endpoints
---------
GET  /health                       — liveness probe
GET  /products                     — list all products (metadata only)
POST /products/register            — register a new product (first view)
POST /products/{product_id}/views  — add an additional view to an existing product
POST /products/query               — find similar products for a query image
DELETE /products/{product_id}      — remove a product and all its views
"""

import base64
from contextlib import asynccontextmanager

import cv2
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
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
    try:
        with db.conn.cursor() as cur:
            cur.execute("SELECT 1")
        db_ok = True
    except Exception:
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "db": db_ok}


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

@app.post("/products/preview-segmentation")
async def preview_segmentation(
    file: UploadFile = File(...),
    seg_method: str = Form("grabcut"),
):
    """
    Run segmentation on an uploaded image and return the result for interactive editing.

    Returns:
      {
        "original_img": "<base64 PNG of the original image>",
        "mask_b64":     "<base64 grayscale PNG — 255 foreground, 0 background>",
        "masked_img":   "<base64 PNG of the initial masked result>"
      }
    """
    raw = await file.read()
    img = _decode_image(raw)
    mask, masked_img = _segment(img, seg_method)
    ok, mask_buf = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError("cv2.imencode failed for mask")
    mask_b64 = base64.b64encode(mask_buf.tobytes()).decode("ascii")
    return {
        "original_img": _encode_png(img),
        "mask_b64": mask_b64,
        "masked_img": _encode_png(masked_img),
    }


@app.post("/products/register", status_code=201)
async def register_product(
    name: str = Form(...),
    file: UploadFile = File(...),
    seg_method: str = Form("grabcut"),
    mask_data: str | None = Form(None),
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

    if mask_data:
        mask_bytes = base64.b64decode(mask_data)
        mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise HTTPException(status_code=400, detail="Could not decode provided mask image")
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        if np.count_nonzero(mask) == 0:
            raise HTTPException(status_code=400, detail="Mask is empty — mark at least some foreground area")
        masked_img = cv2.bitwise_and(img, img, mask=mask)
    else:
        mask, masked_img = _segment(img, seg_method)

    features = pipeline.extract(img, mask)
    product_id = db.create_product(name, image_data=raw)
    feature_id = db.add_view(product_id, features, view_image_data=raw)
    vs.upsert(feature_id, product_id, features["embedding"])

    return {
        "product_id": product_id,
        "masked_img": _encode_png(masked_img),
    }


# ---------------------------------------------------------------------------
# Products — add view to existing product
# ---------------------------------------------------------------------------

@app.post("/products/{product_id}/views", status_code=201)
async def add_product_view(
    product_id: int,
    file: UploadFile = File(...),
    seg_method: str = Form("grabcut"),
    mask_data: str | None = Form(None),
):
    """
    Add an additional view (angle/lighting condition) to an existing product.

    Multipart form fields:
      - file        : image file for this view
      - seg_method  : "grabcut" (default) or "watershed"
      - mask_data   : optional base64 PNG mask

    Returns:
      {
        "product_id": <int>,
        "view_id":    <int>,
        "masked_img": "<base64 PNG of the segmented foreground>"
      }
    """
    # Verify the product exists
    if db.get_by_id(product_id) is None:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")

    raw = await file.read()
    img = _decode_image(raw)

    if mask_data:
        mask_bytes = base64.b64decode(mask_data)
        mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise HTTPException(status_code=400, detail="Could not decode provided mask image")
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        if np.count_nonzero(mask) == 0:
            raise HTTPException(status_code=400, detail="Mask is empty — mark at least some foreground area")
        masked_img = cv2.bitwise_and(img, img, mask=mask)
    else:
        mask, masked_img = _segment(img, seg_method)

    features = pipeline.extract(img, mask)
    feature_id = db.add_view(product_id, features, view_image_data=raw)
    vs.upsert(feature_id, product_id, features["embedding"])

    return {
        "product_id": product_id,
        "view_id":    feature_id,
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
    mask_data: str | None = Form(None),
):
    """
    Find the top-k most similar registered products for a query image.

    Multipart form fields:
      - file        : query image
      - top_k       : number of results to return (default 5)
      - seg_method  : "grabcut" (default) or "watershed"
      - mask_data   : optional base64 PNG mask edited by the user (255=fg, 0=bg).
                      When provided, auto-segmentation is skipped entirely.

    Returns:
      {
        "results": [
          {
            "product_id": <int>,
            "name": <str>,
            "score": <float 0–1>,
            "breakdown": { "contour": f, "hist": f, "sift": f, "hu": f, "corner": f },
            "match_img": "<base64 PNG — side-by-side with SIFT lines>",
            "score_bars_img": "<base64 PNG — confidence bar chart>"
          },
          ...
        ]
      }
    """
    raw = await file.read()
    query_img = _decode_image(raw)

    if mask_data:
        mask_bytes = base64.b64decode(mask_data)
        mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise HTTPException(status_code=400, detail="Could not decode provided mask image")
        if mask.shape[:2] != query_img.shape[:2]:
            mask = cv2.resize(mask, (query_img.shape[1], query_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        if np.count_nonzero(mask) == 0:
            raise HTTPException(status_code=400, detail="Mask is empty — mark at least some foreground area")
    else:
        mask, _ = _segment(query_img, seg_method)

    query_features = pipeline.extract(query_img, mask)

    match_results = matcher.match(query_features, top_k=top_k)

    output = []
    for r in match_results:
        # Retrieve the best-matching view's image for visualization
        candidate_img = None
        view_img_bytes = db.get_view_image(r.view_id) if r.view_id else None
        if view_img_bytes:
            try:
                candidate_img = _decode_image(view_img_bytes)
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
# Products — image
# ---------------------------------------------------------------------------

@app.get("/products/{product_id}/image")
def get_product_image(product_id: int):
    """Return the raw stored image for a product (JPEG or PNG, whatever was uploaded)."""
    row = db.get_by_id(product_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
    image_data = row.get("image_data")
    if not image_data:
        raise HTTPException(status_code=404, detail="No image stored for this product")
    raw = bytes(image_data)
    # Detect content type by magic bytes
    content_type = "image/jpeg" if raw[:2] == b"\xff\xd8" else "image/png"
    return Response(content=raw, media_type=content_type)


# ---------------------------------------------------------------------------
# Frontend — must be last so API routes take precedence
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
