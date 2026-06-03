"""
app/main.py
-----------
Asynchronous FastAPI application serving Cat-vs-Dog predictions from a
ResNet18 backbone.

Endpoints
~~~~~~~~~
GET  /health   – liveness / readiness probe
POST /predict  – accepts an image file, returns class label + confidence
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.model_utils import CLASS_LABELS, load_model, load_gatekeeper, screen_image, preprocess_image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("cat_or_dog")

# ---------------------------------------------------------------------------
# Pydantic response schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Schema returned by the /health endpoint."""
    status: str = Field(..., examples=["healthy"])
    model_loaded: bool = Field(
        ..., description="Whether the model weights were loaded successfully."
    )


class PredictionResponse(BaseModel):
    """Schema returned by the /predict endpoint."""
    label: str = Field(..., examples=["cat", "dog", "none"])
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Softmax probability of the predicted class.",
    )
    class_probabilities: dict[str, float] = Field(
        ..., description="Per-class softmax probabilities.",
    )


class ErrorResponse(BaseModel):
    """Generic error envelope."""
    detail: str


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
_model: nn.Module | None = None
_gatekeeper: nn.Module | None = None
_device: torch.device = torch.device("cpu")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models once at startup; clean up on shutdown."""
    global _model, _gatekeeper, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        _model = load_model(device=_device)
        logger.info("Binary classifier loaded successfully on %s.", _device)
    except FileNotFoundError as exc:
        logger.error("Model loading failed: %s", exc)
        _model = None  # the app stays up but /predict will 503

    try:
        _gatekeeper = load_gatekeeper(device=_device)
        logger.info("ImageNet gatekeeper loaded successfully on %s.", _device)
    except Exception as exc:
        logger.error("Gatekeeper loading failed: %s", exc)
        _gatekeeper = None

    yield  # ── app is running ──

    _model = None
    _gatekeeper = None
    logger.info("Shutdown complete – model references released.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Cat or Dog – Inference API",
    version="0.1.0",
    description=(
        "Production-grade MLOps inference layer serving a baseline "
        "ResNet18 model for binary Cat-vs-Dog classification."
    ),
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Frontend Mount
# ---------------------------------------------------------------------------
from pathlib import Path
_STATIC_DIR = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

@app.get("/", include_in_schema=False)
async def root_frontend():
    """Serve the web frontend index.html."""
    return FileResponse(_STATIC_DIR / "index.html")


# Allowed MIME prefixes for uploaded images
_ALLOWED_CONTENT_TYPES = frozenset({
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/bmp",
    "image/tiff",
    "image/webp",
})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["ops"],
)
async def health() -> HealthResponse:
    """Liveness / readiness probe.

    Returns the current health status of the API and whether the model
    weights were loaded at startup.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type"},
        503: {"model": ErrorResponse, "description": "Model unavailable"},
    },
    summary="Classify an image as Cat or Dog",
    tags=["inference"],
)
async def predict(file: UploadFile = File(..., description="Image to classify")) -> PredictionResponse:
    """Accept an uploaded image and return the predicted class with
    confidence scores.

    Raises
    ------
    HTTPException 400
        If the uploaded file is not a recognised image MIME type.
    HTTPException 503
        If the model is not available (e.g. weights file was missing at startup).
    """
    # ── Guard: model availability ──
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Ensure baseline_model.pth exists and restart the server.",
        )

    # ── Guard: content type ──
    content_type = file.content_type or ""
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{content_type}'. "
                f"Accepted types: {', '.join(sorted(_ALLOWED_CONTENT_TYPES))}."
            ),
        )

    # ── Read & preprocess ──
    image_bytes = await file.read()
    try:
        tensor = preprocess_image(image_bytes, device=_device)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process image: {exc}",
        )

    # ── Gatekeeper: what does this image contain? ──
    screening = "unknown"
    if _gatekeeper is not None:
        screening = screen_image(tensor, _gatekeeper)

        if screening == "both":
            return PredictionResponse(
                label="both",
                confidence=1.0,
                class_probabilities={label: 0.5 for label in CLASS_LABELS},
            )

    # ── Inference ──
    with torch.no_grad():
        logits = _model(tensor)                         # (1, NUM_CLASSES)
        probs  = torch.softmax(logits, dim=1)[0]        # (NUM_CLASSES,)

    predicted_idx = int(probs.argmax())
    top_confidence = float(probs[predicted_idx])
    class_probs = {
        label: round(float(probs[i]), 4)
        for i, label in enumerate(CLASS_LABELS)
    }

    # If gatekeeper says it's not a real cat/dog, return as a fun "resembles" result
    if screening == "none":
        predicted_label = f"resembles-{CLASS_LABELS[predicted_idx]}"
    else:
        predicted_label = CLASS_LABELS[predicted_idx]

    return PredictionResponse(
        label=predicted_label,
        confidence=round(top_confidence, 4),
        class_probabilities=class_probs,
    )
