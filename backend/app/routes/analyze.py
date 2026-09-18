"""
Main analysis endpoint — accepts a fundus image and returns the full
EDiTH pipeline result: quality check → classification → Grad-CAM →
RAG retrieval → LLM report.
"""

import base64
import io
import logging
import time

from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image

from app.config import get_settings
from app.model_registry import models

logger = logging.getLogger("edith.analyze")
router = APIRouter()


@router.post("/analyze")
async def analyze_fundus(file: UploadFile = File(...)):
    """
    Full analysis pipeline for a single fundus image.

    Returns:
        JSON with quality assessment, DR classification, Grad-CAM heatmap,
        similar cases, and LLM-generated report.
    """
    settings = get_settings()
    start_time = time.time()

    # ------------------------------------------------------------------
    # 1. Read & validate uploaded image
    # ------------------------------------------------------------------
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    logger.info(f"Received image: {file.filename} ({image.size[0]}x{image.size[1]})")

    # ------------------------------------------------------------------
    # 2. Image Quality Assessment
    # ------------------------------------------------------------------
    quality_result = {"gradable": True, "confidence": 1.0, "issues": []}

    if models.get("quality") is not None:
        from app.models.quality import assess_quality

        quality_result = assess_quality(image, models["quality"], settings)
        logger.info(f"Quality: gradable={quality_result['gradable']}, "
                     f"confidence={quality_result['confidence']:.3f}")

        if not quality_result["gradable"]:
            return {
                "quality": quality_result,
                "classification": None,
                "gradcam_image": None,
                "similar_cases": [],
                "report": (
                    "Image quality is insufficient for reliable analysis. "
                    "Please recapture the fundus image with better focus "
                    "and illumination."
                ),
                "processing_time_s": round(time.time() - start_time, 2),
            }

    # ------------------------------------------------------------------
    # 3. DR Classification
    # ------------------------------------------------------------------
    classification_result = None
    gradcam_b64 = None

    if models.get("classifier") is not None:
        from app.models.classifier import classify
        from app.gradcam.gradcam import generate_gradcam

        classification_result = classify(image, models["classifier"], settings)
        logger.info(
            f"Classification: {classification_result['label']} "
            f"(grade {classification_result['grade']}, "
            f"confidence {classification_result['confidence']:.3f})"
        )

        # ------------------------------------------------------------------
        # 4. Grad-CAM
        # ------------------------------------------------------------------
        try:
            gradcam_img = generate_gradcam(
                image, models["classifier"], classification_result["grade"], settings
            )
            # Encode heatmap overlay as base64 PNG
            buf = io.BytesIO()
            gradcam_img.save(buf, format="PNG")
            gradcam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            logger.warning(f"Grad-CAM generation failed: {e}")
            gradcam_b64 = None
    else:
        # Fallback when no classifier is loaded (development mode)
        classification_result = {
            "grade": -1,
            "label": "Model not loaded",
            "confidence": 0.0,
            "probabilities": [],
        }

    # ------------------------------------------------------------------
    # 5. RAG Case Retrieval
    # ------------------------------------------------------------------
    similar_cases = []

    if models.get("retriever") is not None and classification_result["grade"] >= 0:
        from app.rag.retriever import retrieve_similar_cases

        similar_cases = retrieve_similar_cases(
            models["retriever"],
            dr_grade=classification_result["grade"],
            top_k=settings.RAG_TOP_K,
        )
        logger.info(f"Retrieved {len(similar_cases)} similar cases")

    # ------------------------------------------------------------------
    # 6. LLM Report Generation
    # ------------------------------------------------------------------
    report = ""

    if settings.GROQ_API_KEY and classification_result["grade"] >= 0:
        try:
            from app.llm.report_generator import generate_report

            report = generate_report(
                classification=classification_result,
                quality=quality_result,
                similar_cases=similar_cases,
                settings=settings,
            )
            logger.info("LLM report generated successfully")
        except Exception as e:
            logger.error(f"LLM report generation failed: {e}")
            report = f"Report generation unavailable: {e}"
    elif not settings.GROQ_API_KEY:
        report = "Report generation unavailable — GROQ_API_KEY not configured."

    # ------------------------------------------------------------------
    # Response
    # ------------------------------------------------------------------
    processing_time = round(time.time() - start_time, 2)
    logger.info(f"Analysis complete in {processing_time}s")

    return {
        "quality": quality_result,
        "classification": classification_result,
        "gradcam_image": gradcam_b64,
        "similar_cases": similar_cases,
        "report": report,
        "processing_time_s": processing_time,
    }
