"""
Health check endpoint for EDiTH backend.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Returns server health status and loaded model availability."""
    from app.model_registry import models

    return {
        "status": "healthy",
        "service": "EDiTH",
        "models": {
            "classifier": models.get("classifier") is not None,
            "quality": models.get("quality") is not None,
            "retriever": models.get("retriever") is not None,
        },
    }
