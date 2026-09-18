"""
Grad-CAM Explainability Module.

Generates class activation heatmaps from the final convolutional layer
of EfficientNet-B3 to visually highlight the retinal regions that
influenced the model's DR severity prediction.

This supports clinician trust by showing *where* the model is looking,
making it possible to verify whether the model focused on actual lesions
(microaneurysms, hemorrhages, exudates) or irrelevant artifacts.
"""

import logging

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.config import Settings
from app.models.preprocessing import preprocess_for_classification

logger = logging.getLogger("edith.gradcam")

# ---------------------------------------------------------------------------
# Grad-CAM transforms (same as classifier)
# ---------------------------------------------------------------------------
gradcam_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class GradCAM:
    """
    Grad-CAM implementation for convolutional neural networks.

    Hooks into a target convolutional layer to capture activations and
    gradients, then computes a weighted combination to produce a heatmap.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, input, output):
        """Forward hook to capture activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook to capture gradients."""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the given input and target class.

        Args:
            input_tensor: Preprocessed input tensor [1, C, H, W].
            target_class: Target class index for gradient computation.

        Returns:
            Heatmap as numpy array [H, W] with values in [0, 1].
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)

        # Zero all gradients
        self.model.zero_grad()

        # Backward pass for target class
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)

        # Compute Grad-CAM
        if self.gradients is None or self.activations is None:
            logger.warning("Grad-CAM: No gradients/activations captured")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        # Global average pooling of gradients → channel weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]

        # ReLU — keep only positive contributions
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def remove_hooks(self):
        """Remove registered hooks to free memory."""
        self._forward_hook.remove()
        self._backward_hook.remove()


def get_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Get the target convolutional layer for Grad-CAM from EfficientNet-B3.

    For EfficientNet models in timm, the last convolutional block is
    typically at model.conv_head or the last block in model.blocks.
    """
    # timm EfficientNet: the last conv layer before the classifier
    if hasattr(model, "conv_head"):
        return model.conv_head
    # Fallback: last block's last convolution
    elif hasattr(model, "blocks"):
        last_block = model.blocks[-1]
        # Find the last Conv2d in the block
        last_conv = None
        for module in last_block.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            return last_conv

    raise ValueError("Could not find target layer for Grad-CAM in the model")


def generate_gradcam(
    pil_image: Image.Image,
    classifier_model: dict,
    target_class: int,
    settings: Settings,
) -> Image.Image:
    """
    Generate a Grad-CAM heatmap overlaid on the original fundus image.

    Args:
        pil_image: Original fundus image (PIL).
        classifier_model: Dict with 'model' and 'device'.
        target_class: DR grade (0–4) to explain.
        settings: Application settings.

    Returns:
        PIL Image with Grad-CAM heatmap overlaid on the original image.
    """
    model = classifier_model["model"]
    device = classifier_model["device"]
    input_size = classifier_model.get("input_size", settings.CLASSIFIER_INPUT_SIZE)

    # Enable gradients for Grad-CAM (model was in eval + no_grad for inference)
    model.eval()

    # Preprocess image
    preprocessed = preprocess_for_classification(
        pil_image, target_size=input_size
    )
    rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
    pil_preprocessed = Image.fromarray(rgb)
    pil_preprocessed = pil_preprocessed.resize(
        (input_size, input_size),
        Image.BILINEAR,
    )

    input_tensor = gradcam_transform(pil_preprocessed).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    # Get target layer and generate heatmap
    target_layer = get_target_layer(model)
    grad_cam = GradCAM(model, target_layer)

    try:
        heatmap = grad_cam.generate(input_tensor, target_class)
    finally:
        grad_cam.remove_hooks()

    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(
        heatmap,
        (input_size, input_size),
    )

    # Apply colormap (JET — blue=cold/low, red=hot/high)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay on original image
    original_resized = np.array(pil_preprocessed)
    overlay = cv2.addWeighted(original_resized, 0.6, heatmap_colored, 0.4, 0)

    return Image.fromarray(overlay)
