"""
model/prepare_data.py
---------------------
Organizes a pet-images dataset into the train/val folder structure that
torchvision.datasets.ImageFolder expects:

    data/
    +-- train/
    |   +-- cat/
    |   +-- dog/
    +-- val/
        +-- cat/
        +-- dog/

Supports two source formats:
  1. Microsoft PetImages layout:   PetImages/Cat/*.jpg, PetImages/Dog/*.jpg
  2. Kaggle flat layout:           train/cat.0.jpg, dog.0.jpg, ...

Also filters out corrupt / unreadable images that would crash training.

Usage:
    python -m model.prepare_data --source PetImages
    python -m model.prepare_data --source PetImages --val-split 0.15
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_DEST: Path = PROJECT_ROOT / "data"
DEFAULT_VAL_SPLIT: float = 0.20  # 80% train, 20% val
SEED: int = 42
VALID_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif"})


def _is_valid_image(path: Path) -> bool:
    """Return True if *path* can be opened as an RGB image by Pillow."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _detect_format(source_dir: Path) -> str:
    """Detect whether *source_dir* is PetImages-style or Kaggle-flat-style."""
    # PetImages style: subdirs named Cat / Dog (case-insensitive)
    subdirs = {d.name.lower() for d in source_dir.iterdir() if d.is_dir()}
    if {"cat", "dog"} <= subdirs:
        return "petimages"

    # Kaggle flat style: files named cat.0.jpg / dog.0.jpg
    sample_files = [f.name.lower() for f in source_dir.iterdir() if f.is_file()][:50]
    if any(f.startswith("cat") for f in sample_files) and any(f.startswith("dog") for f in sample_files):
        return "kaggle"

    raise ValueError(
        f"Could not detect dataset format in '{source_dir}'.\n"
        "Expected either:\n"
        "  - PetImages style: subdirectories named 'Cat' and 'Dog'\n"
        "  - Kaggle style: files named 'cat.0.jpg', 'dog.0.jpg', etc."
    )


def _collect_petimages(source_dir: Path) -> tuple[list[Path], list[Path]]:
    """Collect image paths from PetImages/Cat and PetImages/Dog."""
    cat_dir = next(d for d in source_dir.iterdir() if d.is_dir() and d.name.lower() == "cat")
    dog_dir = next(d for d in source_dir.iterdir() if d.is_dir() and d.name.lower() == "dog")

    cats = sorted([f for f in cat_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS])
    dogs = sorted([f for f in dog_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS])
    return cats, dogs


def _collect_kaggle(source_dir: Path) -> tuple[list[Path], list[Path]]:
    """Collect image paths from a flat Kaggle-style directory."""
    cats = sorted([f for f in source_dir.iterdir() if f.name.lower().startswith("cat") and f.suffix.lower() in VALID_EXTENSIONS])
    dogs = sorted([f for f in source_dir.iterdir() if f.name.lower().startswith("dog") and f.suffix.lower() in VALID_EXTENSIONS])
    return cats, dogs


def organise_dataset(
    source_dir: Path,
    dest_dir: Path = DEFAULT_DEST,
    val_split: float = DEFAULT_VAL_SPLIT,
) -> dict[str, int]:
    """Copy valid images from *source_dir* into an ImageFolder layout.

    Corrupt images are detected and skipped automatically.

    Returns
    -------
    dict
        Counts per split, e.g. ``{"train_cat": 8000, ...}``.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    fmt = _detect_format(source_dir)
    print(f"[*] Detected format: {fmt}")

    if fmt == "petimages":
        cats, dogs = _collect_petimages(source_dir)
    else:
        cats, dogs = _collect_kaggle(source_dir)

    print(f"[*] Found {len(cats):,} cat files, {len(dogs):,} dog files")

    # ── Filter corrupt images ──
    print("[*] Validating images (filtering corrupt files)...")
    clean_cats: list[Path] = []
    clean_dogs: list[Path] = []
    corrupt_count = 0

    for f in cats:
        if _is_valid_image(f):
            clean_cats.append(f)
        else:
            corrupt_count += 1

    for f in dogs:
        if _is_valid_image(f):
            clean_dogs.append(f)
        else:
            corrupt_count += 1

    if corrupt_count:
        print(f"    Skipped {corrupt_count} corrupt/unreadable files")
    print(f"    Valid: {len(clean_cats):,} cats, {len(clean_dogs):,} dogs")

    # ── Shuffle & split ──
    random.seed(SEED)
    random.shuffle(clean_cats)
    random.shuffle(clean_dogs)

    def _split(files: list[Path]) -> tuple[list[Path], list[Path]]:
        n_val = int(len(files) * val_split)
        return files[n_val:], files[:n_val]

    train_cats, val_cats = _split(clean_cats)
    train_dogs, val_dogs = _split(clean_dogs)

    # ── Copy into ImageFolder layout ──
    splits: dict[str, list[tuple[str, list[Path]]]] = {
        "train": [("cat", train_cats), ("dog", train_dogs)],
        "val":   [("cat", val_cats),   ("dog", val_dogs)],
    }

    counts: dict[str, int] = {}

    for split_name, classes in splits.items():
        for class_name, files in classes:
            out_dir = dest_dir / split_name / class_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for f in files:
                shutil.copy2(f, out_dir / f.name)

            key = f"{split_name}_{class_name}"
            counts[key] = len(files)
            print(f"    {split_name}/{class_name}: {len(files):,} images")

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize a pet-images dataset into ImageFolder layout."
    )
    parser.add_argument(
        "--source", type=Path, required=True,
        help="Path to the dataset folder (PetImages/ or flat Kaggle train/).",
    )
    parser.add_argument(
        "--dest", type=Path, default=DEFAULT_DEST,
        help=f"Destination root (default: {DEFAULT_DEST}).",
    )
    parser.add_argument(
        "--val-split", type=float, default=DEFAULT_VAL_SPLIT,
        help=f"Fraction of data for validation (default: {DEFAULT_VAL_SPLIT}).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Dataset Preparation")
    print("=" * 60)
    print(f"  Source:    {args.source}")
    print(f"  Dest:      {args.dest}")
    print(f"  Val split: {args.val_split:.0%}")
    print()

    counts = organise_dataset(args.source, args.dest, args.val_split)

    total = sum(counts.values())
    print(f"\n[OK] Done! {total:,} images organized into {args.dest}")
    print("     Run next:  python -m model.train")


if __name__ == "__main__":
    main()
