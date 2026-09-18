"""
Fundus Image Preprocessing Pipeline.

Implements Ben Graham's preprocessing (from the Kaggle DR competition),
CLAHE enhancement, and circular fundus cropping for consistent input
to the DL models.
"""

import cv2
import numpy as np
from PIL import Image


def crop_fundus_circle(image: np.ndarray) -> np.ndarray:
    """
    Crop the circular fundus region from the image, removing the black
    background borders common in fundus photographs.

    Args:
        image: BGR image array.

    Returns:
        Cropped BGR image containing only the fundus circle.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to find the bright fundus region
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find contours and pick the largest one (the fundus)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Add small padding
    pad = 10
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(image.shape[1] - x, w + 2 * pad)
    h = min(image.shape[0] - y, h + 2 * pad)

    cropped = image[y : y + h, x : x + w]
    return cropped


def ben_graham_preprocessing(image: np.ndarray, target_size: int = 300) -> np.ndarray:
    """
    Ben Graham's preprocessing for fundus images (Kaggle DR competition winner).

    Steps:
        1. Crop to fundus circle
        2. Resize to target size
        3. Subtract local average color (Gaussian blur) to normalize illumination
        4. Clip and rescale to [0, 255]

    Args:
        image: BGR image array.
        target_size: Output image dimension (square).

    Returns:
        Preprocessed BGR image array.
    """
    # Step 1: Crop fundus
    image = crop_fundus_circle(image)

    # Step 2: Resize
    image = cv2.resize(image, (target_size, target_size))

    # Step 3: Subtract local average color
    # This normalizes illumination variations across different cameras/clinics
    image = image.astype(np.float32)
    blur = cv2.GaussianBlur(image, (0, 0), target_size / 30.0)
    image = cv2.addWeighted(image, 4, blur, -4, 128)

    # Step 4: Clip to valid range
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to
    enhance the visibility of retinal features like microaneurysms and
    hemorrhages.

    Args:
        image: BGR image array.
        clip_limit: CLAHE clip limit.
        tile_grid_size: Grid size for histogram equalization.

    Returns:
        CLAHE-enhanced BGR image array.
    """
    # Convert to LAB color space — apply CLAHE only to the L channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return result


def preprocess_for_classification(
    pil_image: Image.Image, target_size: int = 300
) -> np.ndarray:
    """
    Full preprocessing pipeline for the DR classification model.

    Args:
        pil_image: Input PIL Image.
        target_size: Model input size.

    Returns:
        Preprocessed BGR image array ready for model input.
    """
    # PIL → OpenCV (BGR)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Ben Graham preprocessing (crop + normalize illumination)
    image = ben_graham_preprocessing(image, target_size)

    # CLAHE enhancement
    image = apply_clahe(image)

    return image


def preprocess_for_quality(
    pil_image: Image.Image, target_size: int = 224
) -> np.ndarray:
    """
    Preprocessing for the IQA model. Lighter than classification preprocessing
    because the quality model needs to see the original image artifacts.

    Args:
        pil_image: Input PIL Image.
        target_size: Model input size.

    Returns:
        Preprocessed BGR image array.
    """
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    image = crop_fundus_circle(image)
    image = cv2.resize(image, (target_size, target_size))
    return image


def compute_blur_score(pil_image: Image.Image) -> float:
    """
    Compute a blur score using Laplacian variance.
    Higher values = sharper image. Below ~100 is typically blurry.

    Args:
        pil_image: Input PIL Image.

    Returns:
        Laplacian variance (float). Higher = sharper.
    """
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(laplacian_var)
