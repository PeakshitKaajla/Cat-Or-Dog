"""
app/model_utils.py
------------------
Runtime utilities for the inference layer:
  • load the saved ResNet18 checkpoint into an eval-mode model
  • pre-process raw image bytes into the tensor format the model expects
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WEIGHTS_PATH: Path = Path(__file__).resolve().parent.parent / "weights" / "baseline_model.pth"
CLASS_LABELS: List[str] = ["cat", "dog"]
NUM_CLASSES: int = len(CLASS_LABELS)

# ImageNet channel-wise statistics used during pre-training
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Image transform pipeline
# ---------------------------------------------------------------------------
inference_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_model(weights_path: Path = WEIGHTS_PATH,
               device: torch.device | None = None) -> nn.Module:
    """Reconstruct the ResNet18 binary classifier and load saved weights.

    Parameters
    ----------
    weights_path : Path
        Absolute or relative path to the ``.pth`` state-dict file.
    device : torch.device, optional
        Target device.  Falls back to CUDA when available, else CPU.

    Returns
    -------
    nn.Module
        The model in **eval** mode, ready for inference.

    Raises
    ------
    FileNotFoundError
        If *weights_path* does not exist on disk.
    """
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at '{weights_path}'. "
            "Run `python -m model.train` first to generate them."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Re-create the architecture (must match model/train.py)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------
def preprocess_image(image_bytes: bytes,
                     device: torch.device | None = None) -> torch.Tensor:
    """Convert raw image bytes to a batched tensor ready for the model.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of a JPEG / PNG / etc. image.
    device : torch.device, optional
        Target device (defaults to CPU).

    Returns
    -------
    torch.Tensor
        Shape ``(1, 3, 224, 224)`` – a single-image batch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = inference_transforms(image)           # (3, 224, 224)
    return tensor.unsqueeze(0).to(device)           # (1, 3, 224, 224)
