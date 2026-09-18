"""
EDiTH Backend — FastAPI Application Entry Point.

Initializes the FastAPI app, registers CORS middleware, mounts routes,
and loads DL models into memory on startup.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.model_registry import models
from app.routes import analyze, health

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("edith")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load DL models once on startup, release on shutdown."""
    settings = get_settings()
    logger.info("EDiTH starting up — loading models...")

    # --- DR Classifier (EfficientNet-B3) ---
    try:
        from app.models.classifier import load_classifier

        models["classifier"] = load_classifier(settings)
        logger.info("✓ DR classifier loaded")
    except Exception as e:
        logger.warning(f"⚠ DR classifier not loaded (weights missing?): {e}")
        models["classifier"] = None

    # --- Image Quality Assessment (MobileNetV2) ---
    try:
        from app.models.quality import load_quality_model

        models["quality"] = load_quality_model(settings)
        logger.info("✓ IQA model loaded")
    except Exception as e:
        logger.warning(f"⚠ IQA model not loaded (weights missing?): {e}")
        models["quality"] = None

    # --- RAG Knowledge Base ---
    try:
        from app.rag.retriever import load_retriever

        models["retriever"] = load_retriever(settings)
        logger.info("✓ RAG retriever loaded")
    except Exception as e:
        logger.warning(f"⚠ RAG retriever not loaded: {e}")
        models["retriever"] = None

    logger.info("EDiTH startup complete.")
    yield

    # Cleanup
    models.clear()
    logger.info("EDiTH shutdown — models released.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="EDiTH",
    description=(
        "AI-Powered Diabetic Retinopathy Screening "
        "& Clinical Decision-Support System"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# --- CORS ---
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pathlib import Path

# --- Routes ---
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(analyze.router, prefix="/api", tags=["Analysis"])

# --- Serve Frontend directly at root URL (http://127.0.0.1:8000/) ---
frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

