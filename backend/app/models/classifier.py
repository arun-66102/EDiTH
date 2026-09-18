"""
DR Severity Classifier Module.

Uses EfficientNet-B3 (pretrained on ImageNet, fine-tuned on APTOS/EyePACS)
to classify fundus images into one of five DR severity levels:
    0 — No DR
    1 — Mild
    2 — Moderate
    3 — Severe
    4 — Proliferative DR
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms

from app.config import Settings
from app.models.preprocessing import preprocess_for_classification

logger = logging.getLogger("edith.classifier")

# ---------------------------------------------------------------------------
# Image transforms for EfficientNet-B3
# ---------------------------------------------------------------------------
classifier_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def build_classifier(model_name: str = "efficientnet_b0", num_classes: int = 5) -> nn.Module:
    """
    Build EfficientNet model for DR severity classification.

    Uses the `timm` library which provides clean access to EfficientNet
    variants with configurable classifier heads.
    """
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
    )
    return model


def load_classifier(settings: Settings) -> dict:
    """
    Load the DR classification model and weights.

    Returns a dict with model, device, and input_size for use in inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = getattr(settings, "CLASSIFIER_MODEL_NAME", "efficientnet_b0")
    input_size = settings.CLASSIFIER_INPUT_SIZE
    weights_path = settings.MODEL_DIR / settings.CLASSIFIER_WEIGHTS

    # Fallback check: if default weights don't exist, check alternative
    if not weights_path.exists():
        fallback_b0 = settings.MODEL_DIR / "efficientnet_b0_dr.pth"
        fallback_b3 = settings.MODEL_DIR / "efficientnet_b3_dr.pth"
        if fallback_b0.exists():
            weights_path = fallback_b0
            model_name = "efficientnet_b0"
            input_size = 224
        elif fallback_b3.exists():
            weights_path = fallback_b3
            model_name = "efficientnet_b3"
            input_size = 300

    model = build_classifier(model_name=model_name, num_classes=settings.NUM_DR_CLASSES)

    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded classifier weights ({model_name}) from {weights_path}")
    else:
        logger.warning(
            f"Classifier weights not found at {weights_path}. "
            f"Using pretrained ImageNet weights ({model_name}) — predictions will be random. "
            "Train the classifier on APTOS data first."
        )

    model.to(device)
    model.eval()

    return {"model": model, "device": device, "input_size": input_size, "model_name": model_name}


def classify(
    pil_image: Image.Image, classifier_model: dict, settings: Settings
) -> dict:
    """
    Classify a fundus image into one of five DR severity grades.

    Args:
        pil_image: Input fundus image (PIL).
        classifier_model: Dict with 'model' and 'device' keys.
        settings: Application settings.

    Returns:
        Dict with:
            - grade (int): 0–4 severity grade.
            - label (str): Human-readable label.
            - confidence (float): Softmax confidence of predicted class.
            - probabilities (list[float]): Per-class softmax probabilities.
    """
    model = classifier_model["model"]
    device = classifier_model["device"]
    input_size = classifier_model.get("input_size", settings.CLASSIFIER_INPUT_SIZE)

    # Preprocess: Ben Graham + CLAHE
    preprocessed = preprocess_for_classification(
        pil_image, target_size=input_size
    )

    # OpenCV BGR → PIL RGB → tensor
    from PIL import Image as PILImage
    import cv2

    rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
    pil_preprocessed = PILImage.fromarray(rgb)

    # Resize to exact model input size
    pil_preprocessed = pil_preprocessed.resize(
        (input_size, input_size),
        PILImage.BILINEAR,
    )

    input_tensor = classifier_transform(pil_preprocessed).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)

    probs_np = probs.cpu().numpy()[0]
    grade = int(np.argmax(probs_np))

    return {
        "grade": grade,
        "label": settings.DR_LABELS[grade],
        "confidence": round(float(probs_np[grade]), 4),
        "probabilities": [round(float(p), 4) for p in probs_np],
    }
