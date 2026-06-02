"""
model/train.py
--------------
Fine-tunes a pre-trained ResNet18 for binary Cat-vs-Dog classification.

The feature-extractor layers stay frozen; only the final FC head is trained.
Because only ~1K parameters are trainable, this runs in minutes even on CPU.

Prerequisites:
    1. Download the Kaggle Dogs-vs-Cats dataset.
    2. Run ``python -m model.prepare_data --source path/to/kaggle/train``
       to organize images into ``data/train/{cat,dog}`` and ``data/val/{cat,dog}``.

Usage:
    python -m model.train                          # defaults (5 epochs, lr=1e-3)
    python -m model.train --epochs 10 --lr 0.0005  # custom hyper-params
    python -m model.train --batch-size 64           # bigger batches if RAM allows
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES: int = 2  # Cat, Dog
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
WEIGHTS_DIR: Path = PROJECT_ROOT / "weights"
MODEL_FILENAME: str = "baseline_model.pth"

# ImageNet channel-wise statistics (must match app/model_utils.py)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model(num_classes: int = NUM_CLASSES, freeze_base: bool = True) -> nn.Module:
    """Return a ResNet18 adapted for *num_classes*-way classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 2 - Cat vs Dog).
    freeze_base : bool
        If ``True`` every parameter *except* the final FC layer is frozen so
        only the classifier head is trainable.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    in_features: int = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Progress indicator every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"      batch {batch_idx + 1}/{len(dataloader)} "
                  f"| loss: {loss.item():.4f}")

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on the validation set. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Corrupt image cleanup
# ---------------------------------------------------------------------------
def _remove_corrupt_images(root: Path) -> int:
    """Walk *root* and delete files that PIL cannot open. Returns count removed."""
    removed = 0
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            fpath = Path(dirpath) / fname
            try:
                with Image.open(fpath) as img:
                    img.verify()
            except Exception:
                fpath.unlink()
                removed += 1
    if removed:
        print(f"[*] Removed {removed} corrupt image(s) from {root}")
    return removed


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------
def save_model(model: nn.Module, save_dir: Path = WEIGHTS_DIR,
               filename: str = MODEL_FILENAME) -> Path:
    """Persist model weights (state_dict) to *save_dir/filename*."""
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    torch.save(model.state_dict(), save_path)
    return save_path


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def run_training(
    data_dir: Path = DATA_DIR,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_workers: int = 0,
) -> None:
    """End-to-end training pipeline."""

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    # ── Datasets ──
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training data not found at '{train_dir}'.\n"
            "    Run:  python -m model.prepare_data --source path/to/kaggle/train"
        )

    # Remove corrupt images before training (PetImages has a few broken JPEGs)
    _remove_corrupt_images(train_dir)
    _remove_corrupt_images(val_dir)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    print(f"[*] Training samples : {len(train_dataset):,}")
    print(f"    Validation samples: {len(val_dataset):,}")
    print(f"    Classes: {train_dataset.classes}")
    print(f"    Class-to-idx: {train_dataset.class_to_idx}")
    print()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    # ── Model ──
    model = build_model().to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[*] Model: ResNet18 (trainable: {trainable:,} / {total:,} params)")

    # ── Loss & Optimizer (only FC layer params) ──
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    print(f"[*] Optimizer: Adam (lr={lr})")
    print(f"[*] Batch size: {batch_size}")
    print(f"[*] Epochs: {epochs}")
    print("-" * 60)

    # ── Training loop ──
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch}/{epochs} ({elapsed:.1f}s) "
            f"| train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} "
            f"| val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}",
            end="",
        )

        # Save if best validation accuracy so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_model(model)
            print("  << saved (best)")
        else:
            print()

    print("-" * 60)
    print(f"[OK] Training complete!")
    print(f"     Best val accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"     Weights saved to:  {WEIGHTS_DIR / MODEL_FILENAME}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune ResNet18 for Cat-vs-Dog classification."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help=f"Root of the organized dataset (default: {DATA_DIR}).",
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Number of training epochs (default: 5).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate for Adam optimizer (default: 1e-3).",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="DataLoader worker processes (default: 0 = main process).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Cat vs Dog -- ResNet18 Fine-Tuning")
    print("=" * 60)

    run_training(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
