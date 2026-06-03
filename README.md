<div align="center">

# 🐱 Cat or Dog? 🐶

**A production-grade MLOps inference engine powered by PyTorch & FastAPI**

Built to demonstrate end-to-end machine learning engineering — from model training to cloud deployment.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [API Reference](#-api-reference) · [Training](#-training-your-own-model) · [Deployment](#-cloud-deployment)

</div>

---

## ✨ Features

- **Transfer Learning** — Fine-tuned ResNet18 backbone with a frozen feature extractor. Only 1,026 of 11.1M parameters are trained, enabling CPU-only training in minutes.
- **Two-Stage Inference Pipeline** — A full ImageNet gatekeeper pre-screens every image before the binary classifier runs, preventing misclassification of humans, objects, and other non-animal inputs.
- **Fun Resemblance Mode** — Upload a photo of a person and the engine tells you whether you resemble a cat or a dog.
- **Real-Time REST API** — Async FastAPI backend with Pydantic-validated schemas, structured error handling (400/503), and health probes.
- **Drag & Drop Web UI** — Minimalist frontend with animated confidence bars, served directly by the API at the root URL.
- **One-Command Cloud Deploy** — Dockerized with a Python deployment script for Hugging Face Spaces. No Git required.

---

## 🏗 Architecture

```
                    ┌──────────────────────────────────┐
                    │          Web Frontend             │
                    │   (HTML / CSS / JavaScript)       │
                    └──────────────┬───────────────────┘
                                   │  POST /predict
                                   ▼
                    ┌──────────────────────────────────┐
                    │        FastAPI Server             │
                    │   (Pydantic schemas, async I/O)   │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────────┐
                    │     Stage 1: ImageNet Gatekeeper  │
                    │  Full ResNet18 (1000 classes)     │
                    │                                    │
                    │  Top-5 has cat breeds? dog breeds? │
                    │  ┌─────┬──────┬────────┐          │
                    │  │ cat │ dog  │ both   │ none     │
                    │  └──┬──┴──┬───┴───┬────┘──┬──     │
                    └─────┼─────┼───────┼───────┼──────┘
                          │     │       │       │
                          ▼     ▼       │       ▼
                    ┌─────────────────┐ │  Resemblance
                    │ Stage 2: Binary │ │     Mode
                    │   Classifier    │ │  "You resemble
                    │ ResNet18 (2-cls)│ │    a dog!"
                    │ Fine-tuned head │ │
                    └────────┬────────┘ │
                             │          │
                             ▼          ▼
                    ┌──────────────────────────────────┐
                    │    JSON Response                  │
                    │  { label, confidence,             │
                    │    class_probabilities }          │
                    └──────────────────────────────────┘
```

---

## 📂 Project Structure

```
Cat-Or-Dog/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application, endpoints, lifespan
│   ├── model_utils.py       # Model loaders, gatekeeper, preprocessing
│   └── static/
│       ├── index.html        # Drag & drop web interface
│       ├── style.css         # Minimalist dark/light theme
│       └── script.js         # Client-side upload & animation logic
├── model/
│   ├── __init__.py
│   ├── train.py              # Fine-tuning pipeline with validation
│   └── prepare_data.py       # Dataset splitting & organization
├── weights/
│   └── baseline_model.pth    # Trained model checkpoint
├── Dockerfile                # Production container (HF Spaces ready)
├── deploy.py                 # One-click Hugging Face deployment
├── requirements.txt          # Pinned dependencies
├── run_server.bat            # Windows one-click launcher
└── .gitignore
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/Cat-Or-Dog.git
cd Cat-Or-Dog

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Train the Model (Optional)

A pre-trained checkpoint is included in `weights/`. To retrain from scratch:

```bash
# Prepare the dataset (expects PetImages/ or Kaggle flat layout)
python -m model.prepare_data --source /path/to/PetImages

# Fine-tune (runs on CPU in ~7 minutes)
python -m model.train --epochs 5 --batch-size 32
```

### 3. Run the Server

```bash
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

> **Windows users:** Double-click `run_server.bat` to launch everything with one click.

---

## 📡 API Reference

### `GET /health`

Liveness / readiness probe.

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `POST /predict`

Upload an image to classify.

```bash
curl -X POST -F "file=@cat.jpg" http://localhost:8000/predict
```

**Response:**

```json
{
  "label": "cat",
  "confidence": 0.9782,
  "class_probabilities": {
    "cat": 0.9782,
    "dog": 0.0218
  }
}
```

**Possible `label` values:**

| Label | Meaning |
|-------|---------|
| `cat` | Image is a cat |
| `dog` | Image is a dog |
| `both` | Image contains both a cat and a dog |
| `resembles-cat` | Not a cat/dog, but resembles a cat (fun mode) |
| `resembles-dog` | Not a cat/dog, but resembles a dog (fun mode) |

**Error codes:** `400` (bad file type / corrupt image), `503` (model not loaded)

Full interactive docs available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 🧠 Training Your Own Model

The training pipeline uses **transfer learning** on a ResNet18 backbone pre-trained on ImageNet.

### Strategy

| Parameter | Value |
|---|---|
| Architecture | ResNet18 |
| Pre-trained weights | ImageNet1K_V1 |
| Frozen layers | All convolutional blocks |
| Trainable layers | Final FC head only |
| Trainable parameters | 1,026 / 11,177,538 (0.01%) |
| Optimizer | Adam (lr=1e-3) |
| Loss function | CrossEntropyLoss |
| Data augmentation | RandomCrop, RandomHorizontalFlip |

### Results (1 epoch, CPU)

| Metric | Score |
|---|---|
| Training accuracy | 95.23% |
| Validation accuracy | **96.96%** |
| Training time | ~7 minutes |

### CLI Options

```bash
python -m model.train --help

  --epochs       Number of epochs (default: 5)
  --batch-size   Batch size (default: 32)
  --lr           Learning rate (default: 0.001)
```

---

## ☁️ Cloud Deployment

### Hugging Face Spaces (Recommended, Free)

The `Dockerfile` is pre-configured for Hugging Face Spaces (port 7860, non-root user).

**Option A — No Git required:**

```bash
pip install huggingface_hub
python deploy.py
```

The script prompts for your HF token and Space ID, then uploads everything automatically.

**Option B — Via Git:**

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/cat-or-dog
git push hf main
```

### Docker (Self-hosted)

```bash
docker build -t cat-or-dog .
docker run -p 8000:8000 -e PORT=8000 cat-or-dog
```

---

## 🛡 Out-of-Distribution Safety

A common failure mode with binary classifiers is that they **force a prediction on any input** — even images that aren't cats or dogs. This project solves it with a **two-stage gatekeeper architecture**:

1. **Stage 1** loads the full 1000-class ImageNet ResNet18 and checks if the top-5 predictions contain cat or dog breed classes (ImageNet indices 151–295).
2. **Stage 2** (the fine-tuned binary head) only runs if Stage 1 confirms the image is relevant.
3. If the image is **not** a cat or dog (e.g., a person, car, or food), it enters **Resemblance Mode** — the binary classifier still runs, but the result is framed as a fun *"You resemble a cat!"* instead of a definitive classification.

This prevents embarrassing misclassifications while adding a playful feature.

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| ML Framework | PyTorch, Torchvision |
| API Framework | FastAPI, Pydantic, Uvicorn |
| Frontend | Vanilla HTML, CSS, JavaScript |
| Containerization | Docker |
| Cloud | Hugging Face Spaces |
| Image Processing | Pillow |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
