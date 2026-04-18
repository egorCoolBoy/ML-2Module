from __future__ import annotations

import random
import shutil
from pathlib import Path

import yaml

from .logging_utils import get_logger

LOGGER = get_logger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _safe_copy_pairs(samples: list[tuple[Path, Path]], split_dir: Path) -> None:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for image_path, label_path in samples:
        shutil.copy2(image_path, images_dir / image_path.name)
        shutil.copy2(label_path, labels_dir / label_path.name)


def prepare_dataset(
    source_dir: str | Path,
    output_dir: str | Path = "data/dataset",
    train_ratio: float = 0.7,
    valid_ratio: float = 0.2,
    seed: int = 42,
) -> Path:
    """Prepare YOLO dataset with train/valid/test split from a train-only export."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    train_images = source_dir / "train" / "images"
    train_labels = source_dir / "train" / "labels"

    if not train_images.exists() or not train_labels.exists():
        raise FileNotFoundError(
            f"Expected source dataset in {train_images} and {train_labels}"
        )

    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(train_images.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = train_labels / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))

    if not pairs:
        raise RuntimeError("No image/label pairs found in dataset")

    random.seed(seed)
    random.shuffle(pairs)

    n_total = len(pairs)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)

    train_samples = pairs[:n_train]
    valid_samples = pairs[n_train : n_train + n_valid]
    test_samples = pairs[n_train + n_valid :]

    if output_dir.exists():
        shutil.rmtree(output_dir)

    _safe_copy_pairs(train_samples, output_dir / "train")
    _safe_copy_pairs(valid_samples, output_dir / "valid")
    _safe_copy_pairs(test_samples, output_dir / "test")

    yaml_path = output_dir / "data.yaml"
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["number"],
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    LOGGER.info(
        "Dataset prepared at %s | train=%d valid=%d test=%d",
        output_dir,
        len(train_samples),
        len(valid_samples),
        len(test_samples),
    )
    return yaml_path
