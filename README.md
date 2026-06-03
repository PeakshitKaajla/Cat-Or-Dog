# Cat or Dog -- MLOps Inference Engine

A production-grade, end-to-end machine learning system that classifies images as cats or dogs using a fine-tuned ResNet18 backbone. Built with a two-stage inference pipeline, a minimalist web frontend, and containerized deployment.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Two-Stage Inference Pipeline](#two-stage-inference-pipeline)
- [Model Details](#model-details)
- [API Reference](#api-reference)
- [Frontend](#frontend)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Prepare the Dataset](#prepare-the-dataset)
  - [Train the Model](#train-the-model)
  - [Run the Server](#run-the-server)
- [Docker](#docker)
- [Deployment](#deployment)
- [Tech Stack](#tech-stack)

---

## Overview

This project demonstrates a complete MLOps workflow for binary image classification:

1. **Offline training** -- Fine-tune a pre-trained ResNet18 on the Microsoft PetImages dataset with frozen base layers, reducing trainable parameters from 11.1M to 1,026.
2. **Production inference** -- Serve predictions through an async FastAPI backend with structured Pydantic response schemas, input validation, and error handling.
3. **Out-of-distribution detection** -- A gatekeeper model (full ImageNet ResNet18) pre-screens every image to reject inputs that are not cats or dogs, and detects when both animals are present.
4. **Resemblance mode** -- When a non-cat/dog image is uploaded (e.g., a person), the system runs the classifier anyway and returns a fun "You resemble a..." result.
5. **Web interface** -- A minimalist drag-and-drop frontend with animated confidence bars, served directly by the API.
6. **Containerized deployment** -- Dockerized for one-command deployment to Hugging Face Spaces or Google Cloud Run.

---

## Architecture

```
                         +------------------+
                         |   Web Frontend   |
                         |  (HTML/JS/CSS)   |
                         +--------+---------+
                                  |
                            POST /predict
                                  |
                         +--------v---------+
                         |   FastAPI Server  |
                         |   (app/main.py)   |
                         +--------+---------+
                                  |
                    +-------------+-------------+
                    |                           |
           +--------v---------+       +--------v---------+
           |    Gatekeeper     |       |  Binary Classifier|
           | (ImageNet 1000)   |       |  (Fine-tuned FC)  |
           +--------+---------+       +--------+---------+
                    |                           |
              Screens image              Cat / Dog / Both
           (cat / dog / both /           / Resembles-X
               none)
```

---

## Project Structure

```
Cat-Or-Dog/
|
|-- app/
|   |-- __init__.py
|   |-- main.py              # FastAPI application, endpoints, lifespan
|   |-- model_utils.py       # Model loaders, gatekeeper, preprocessing
|   |-- static/
|       |-- index.html        # Web frontend
|       |-- style.css         # Minimalist dark/light theme
|       |-- script.js         # Drag-and-drop, fetch, UI animations
|
|-- model/
|   |-- __init__.py
|   |-- prepare_data.py      # Dataset splitting (PetImages / Kaggle)
|   |-- train.py             # Fine-tuning loop with validation
|
|-- weights/
|   |-- baseline_model.pth   # Trained model weights (git-ignored)
|
|-- data/                    # Train/val splits (git-ignored)
|   |-- train/
|   |   |-- cat/
|   |   |-- dog/
|   |-- val/
|       |-- cat/
|       |-- dog/
|
|-- Dockerfile               # Production container image
|-- requirements.txt         # Pinned Python dependencies
|-- deploy.py                # One-command Hugging Face uploader
|-- run_server.bat           # Windows one-click launcher
|-- .gitignore
|-- .dockerignore
```

---

## Two-Stage Inference Pipeline

Every uploaded image passes through two models before a result is returned.

### Stage 1: ImageNet Gatekeeper

A full 1000-class ImageNet ResNet18 examines the image and checks whether any of its top-5 predictions fall within known cat or dog breed class indices (ImageNet classes 151-295).

| Gatekeeper Result | Behavior |
|---|---|
| Only cat breeds detected | Proceeds to Stage 2 |
| Only dog breeds detected | Proceeds to Stage 2 |
| Both cat and dog breeds detected | Returns `"both"` immediately |
| Neither detected (person, object, etc.) | Proceeds to Stage 2 in **resemblance mode** |

### Stage 2: Binary Classifier

The fine-tuned ResNet18 head outputs softmax probabilities for `[cat, dog]`. If the gatekeeper flagged the image as out-of-distribution, the label is prefixed with `resembles-` to distinguish fun results from real classifications.

---

## Model Details

| Property | Value |
|---|---|
| Architecture | ResNet18 |
| Pre-trained Weights | ImageNet1K V1 |
| Training Strategy | Transfer learning -- base layers frozen |
| Total Parameters | 11,177,538 |
| Trainable Parameters | 1,026 (final FC layer only) |
| Output Classes | 2 (cat, dog) |
| Training Dataset | Microsoft PetImages (~20,000 train, ~5,000 val) |
| Validation Accuracy | 96.96% (1 epoch) |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam (lr=0.001) |

The frozen-base approach reduces trainable parameters by 99.99%, enabling training on CPU in under 10 minutes and keeping inference lightweight for free-tier cloud deployments.

---

## API Reference

### GET /health

Liveness and readiness probe.

**Response (200):**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### POST /predict

Classify an uploaded image.

**Request:** `multipart/form-data` with a `file` field containing a JPEG, PNG, GIF, BMP, TIFF, or WebP image.

**Response (200):**
```json
{
  "label": "dog",
  "confidence": 0.9782,
  "class_probabilities": {
    "cat": 0.0218,
    "dog": 0.9782
  }
}
```

**Possible label values:**

| Label | Meaning |
|---|---|
| `cat` | Image classified as a cat |
| `dog` | Image classified as a dog |
| `both` | Gatekeeper detected both animals in the image |
| `resembles-cat` | Not a real cat/dog, but the model leans cat |
| `resembles-dog` | Not a real cat/dog, but the model leans dog |

**Error responses:**

| Code | Condition |
|---|---|
| 400 | Unsupported file type or corrupt/unreadable image |
| 503 | Model weights were not found at server startup |

### GET /

Serves the web frontend (`index.html`).

### Interactive Docs

FastAPI auto-generates interactive API documentation at `/docs` (Swagger UI) and `/redoc` (ReDoc).

---

## Frontend

The web interface is a self-contained single-page application built with vanilla HTML, CSS, and JavaScript. No build step or framework required.

**Features:**
- Drag-and-drop or click-to-browse file upload
- Instant local image preview via the FileReader API
- Animated confidence progress bars for cat and dog probabilities
- Distinct visual states for each result type (cat, dog, both, resemblance)
- Fully responsive layout

The frontend is served directly by FastAPI from the `app/static/` directory.

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Cat-Or-Dog.git
cd Cat-Or-Dog

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Prepare the Dataset

Download the [Microsoft PetImages](https://www.microsoft.com/en-us/download/details.aspx?id=54765) dataset, then run the data preparation script to split it into training and validation sets:

```bash
python -m model.prepare_data --source /path/to/PetImages
```

This creates `data/train/` and `data/val/` directories with `cat/` and `dog/` subdirectories.

### Train the Model

```bash
# Default: 5 epochs, lr=0.001, batch size 32
python -m model.train

# Custom configuration
python -m model.train --epochs 10 --lr 0.0005 --batch-size 64
```

The script automatically:
- Scans for and removes corrupt images
- Freezes the ResNet18 base layers
- Tracks validation accuracy per epoch
- Saves the best checkpoint to `weights/baseline_model.pth`

### Run the Server

```bash
# Development mode with hot-reload
uvicorn app.main:app --reload
```

On Windows, you can also double-click `run_server.bat` to start the server and open the browser automatically.

Open `http://127.0.0.1:8000` in your browser to use the web interface, or `http://127.0.0.1:8000/docs` for the interactive API documentation.

**Quick test with curl:**
```bash
curl http://localhost:8000/health

curl -X POST -F "file=@path/to/image.jpg" http://localhost:8000/predict
```

---

## Docker

```bash
# Build the image
docker build -t cat-or-dog .

# Run the container
docker run -p 8000:7860 cat-or-dog
```

The Dockerfile uses `python:3.10-slim`, layer-cached dependency installation, and runs under a non-root user for security compliance.

---

## Deployment

### Hugging Face Spaces (Free)

The included `deploy.py` script uploads the project directly to a Hugging Face Docker Space without requiring Git:

```bash
python deploy.py
```

It will prompt for your Hugging Face access token and Space repository ID, then upload only the necessary files (`app/`, `model/`, `weights/`, `Dockerfile`, `requirements.txt`).

### Google Cloud Run

```bash
gcloud run deploy cat-or-dog-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1
```

The Dockerfile reads the `$PORT` environment variable at runtime, so it is compatible with any container platform that injects a dynamic port.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| Deep Learning | PyTorch, Torchvision, Pillow |
| API Framework | FastAPI, Uvicorn, Pydantic |
| Frontend | HTML, CSS (vanilla), JavaScript (vanilla) |
| Containerization | Docker |
| Deployment | Hugging Face Spaces, Google Cloud Run |
