"""
EDiTH Configuration — Pydantic Settings for environment variables and app config.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # --- API Keys ---
    GROQ_API_KEY: str = ""

    # --- Paths ---
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_DIR: Path = BASE_DIR / "models" / "weights"
    DATA_DIR: Path = BASE_DIR.parent / "data"
    CHROMA_DB_DIR: Path = DATA_DIR / "chroma_db"

    # --- Model Config ---
    CLASSIFIER_MODEL_NAME: str = "efficientnet_b0"
    CLASSIFIER_WEIGHTS: str = "efficientnet_b0_dr.pth"
    IQA_WEIGHTS: str = "mobilenetv2_iqa.pth"
    CLASSIFIER_INPUT_SIZE: int = 224
    IQA_INPUT_SIZE: int = 224
    NUM_DR_CLASSES: int = 5
    DR_LABELS: list[str] = [
        "No DR",
        "Mild",
        "Moderate",
        "Severe",
        "Proliferative DR",
    ]

    # --- IQA Thresholds ---
    IQA_CONFIDENCE_THRESHOLD: float = 0.5
    BLUR_THRESHOLD: float = 100.0  # Laplacian variance below this = blurry

    # --- RAG Config ---
    RAG_COLLECTION_NAME: str = "dr_cases"
    RAG_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RAG_TOP_K: int = 3

    # --- LLM Config ---
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_TEMPERATURE: float = 0.2
    GROQ_MAX_TOKENS: int = 2048

    # --- Server ---
    CORS_ORIGINS: list[str] = ["*"]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
