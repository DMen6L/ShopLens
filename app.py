"""
ShopLens — FastAPI application entry point.

Phase 1: skeleton with lifespan (DB + VectorStore init) and a health endpoint.
Full API endpoints are added in Phase 6.
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from database import ProductDB
from vector_store import make_vector_store

# Shared state — populated during lifespan startup
db: ProductDB | None = None
vs = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    global db, vs
    db = ProductDB()
    db.connect()
    vs = make_vector_store(db)
    yield
    db.close()


app = FastAPI(title="ShopLens", lifespan=lifespan)

# Serve frontend static files at /
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}
