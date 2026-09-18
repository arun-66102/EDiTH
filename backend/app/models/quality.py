"""
Image Quality Assessment (IQA) Module.

Uses a MobileNetV2-based binary classifier to determine whether a fundus
image is gradable (suitable for DR classification) or ungradable (too
blurry, poorly illuminated, or has artifacts).

Also includes heuristic checks (blur score, illumination) as fallbacks.
"""

import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

from app.config import Settings
from app.models.preprocessing import preprocess_for_quality, compute_blur_score

logger = logging.getLogger("edith.quality")

# ---------------------------------------------------------------------------
# Image transforms for MobileNetV2
# ---------------------------------------------------------------------------
quality_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def build_quality_model(num_classes: int = 2) -> nn.Module:
    """
    Build MobileNetV2 for binary IQA classification (Gradable / Ungradable).
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # Replace the final classifier layer
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def load_quality_model(settings: Settings) -> dict:
    """
    Load the IQA model weights. Returns a dict with model and device.

    If weights are not found, returns the model with pretrained ImageNet
    weights (won't be accurate for IQA but allows the pipeline to run).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_quality_model(num_classes=2)

    weights_path = settings.MODEL_DIR / settings.IQA_WEIGHTS
    has_trained_weights = weights_path.exists()
    if has_trained_weights:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded IQA weights from {weights_path}")
    else:
        logger.warning(
            f"IQA weights not found at {weights_path}. "
            "Using heuristic checks (blur & illumination) until IQA model is trained."
        )

    model.to(device)
    model.eval()

    return {"model": model, "device": device, "has_trained_weights": has_trained_weights}


def assess_quality(
    pil_image: Image.Image, quality_model: dict, settings: Settings
) -> dict:
    """
    Assess fundus image quality using both DL model (if trained) and heuristic checks.

    Args:
        pil_image: Input fundus image.
        quality_model: Dict with 'model', 'device', and 'has_trained_weights' keys.
        settings: Application settings.

    Returns:
        Dict with:
            - gradable (bool): Whether the image is suitable for diagnosis.
            - confidence (float): Model confidence in its prediction.
            - issues (list[str]): Detected quality issues.
    """
    issues = []
    model = quality_model["model"]
    device = quality_model["device"]
    has_trained_weights = quality_model.get("has_trained_weights", False)

    # ------------------------------------------------------------------
    # Heuristic check: Blur (Laplacian variance)
    # ------------------------------------------------------------------
    blur_score = compute_blur_score(pil_image)
    if blur_score < 15.0:
        issues.append(f"Image is severely blurry (sharpness score: {blur_score:.1f})")
    elif blur_score < settings.BLUR_THRESHOLD:
        issues.append(f"Mild blur detected (sharpness score: {blur_score:.1f})")

    # ------------------------------------------------------------------
    # Heuristic check: Illumination (mean brightness)
    # ------------------------------------------------------------------
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    mean_brightness = float(gray.mean())
    if mean_brightness < 15:
        issues.append(f"Image is completely dark (brightness: {mean_brightness:.1f})")
    elif mean_brightness < 30:
        issues.append(f"Suboptimal low lighting (brightness: {mean_brightness:.1f})")
    elif mean_brightness > 240:
        issues.append(f"Image is severely overexposed (brightness: {mean_brightness:.1f})")

    # ------------------------------------------------------------------
    # DL-based quality prediction (only when trained weights are loaded)
    # ------------------------------------------------------------------
    dl_gradable = True
    gradable_prob = 0.95
    ungradable_prob = 0.05

    if has_trained_weights:
        input_tensor = quality_transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            gradable_prob = probs[0, 0].item()
            ungradable_prob = probs[0, 1].item()

        dl_gradable = gradable_prob >= settings.IQA_CONFIDENCE_THRESHOLD
        if not dl_gradable:
            issues.append(
                f"DL quality model flagged image as ungradable "
                f"(confidence: {ungradable_prob:.3f})"
            )

    # ------------------------------------------------------------------
    # Final decision: gradable unless critical issues or trained DL model flags it
    # ------------------------------------------------------------------
    has_critical_issues = any(
        "severely blurry" in issue.lower()
        or "completely dark" in issue.lower()
        or "severely overexposed" in issue.lower()
        for issue in issues
    )
    gradable = dl_gradable and not has_critical_issues

    return {
        "gradable": gradable,
        "confidence": round(gradable_prob if gradable else ungradable_prob, 3),
        "issues": issues,
        "blur_score": round(blur_score, 2),
        "mean_brightness": round(mean_brightness, 2),
    }
