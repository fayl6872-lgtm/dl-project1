"""
data_utils.py
=============
Data loading, augmentation, and mixing utilities for the CIFAR-10 project.

Public API
----------
  CIFAR10Dataset   – PyTorch Dataset backed by on-disk PNG files
  get_transforms   – factory for train / val / test transform pipelines
  read_csv         – parse a CSV file into a list of row-dicts
  stratified_split – reproducible train / val split preserving class ratios
  cutmix_collate   – collate function applying CutMix augmentation
  mixup_collate    – collate function applying Mixup augmentation

CIFAR-10 statistics (pre-computed on the full training set):
  mean = (0.4914, 0.4822, 0.4465)
  std  = (0.2470, 0.2435, 0.2616)
"""

import csv
import math
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def read_csv(path: str) -> List[Dict[str, str]]:
    """Read a CSV file and return a list of row dictionaries.

    Args:
        path: Absolute or relative path to the CSV file.

    Returns:
        List of dicts, one per data row (header row is consumed as keys).
    """
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def stratified_split(
    rows: List[Dict[str, str]],
    val_frac: float = 0.1,
    seed: int = 6953,
    fold: int = 0,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Split *rows* into (train_rows, val_rows) while preserving class ratios.

    Args:
        rows:     List of row-dicts, each must have a ``"label"`` key.
        val_frac: Fraction of samples to use for validation (default 0.1).
        seed:     Random seed for reproducibility.

    Returns:
        Tuple (train_rows, val_rows).
    """
    rng = random.Random(seed)
    if fold < 0:
        raise ValueError('Validation fold must be nonnegative')

    # Group indices by class label
    class_indices: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        label = row["label"]
        class_indices.setdefault(label, []).append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []

    for label, indices in class_indices.items():
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n_val = max(1, math.ceil(len(shuffled) * val_frac))
        start, end = fold * n_val, (fold + 1) * n_val
        if end > len(shuffled):
            raise ValueError('Validation fold exceeds the available class samples')
        val_indices.extend(shuffled[start:end])
        train_indices.extend(shuffled[:start] + shuffled[end:])

    train_rows = [rows[i] for i in sorted(train_indices)]
    val_rows   = [rows[i] for i in sorted(val_indices)]
    return train_rows, val_rows


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CIFAR10Dataset(Dataset):
    """PyTorch Dataset that loads CIFAR-10 images from PNG files on disk.

    Args:
        root:      Path to the data directory (e.g. ``../data``). Images are
                   looked up under ``root/train_images/`` or ``root/test_images/``
                   as determined by the ``filename`` field in *rows*.
        rows:      List of dicts from :func:`read_csv`. Each dict must have
                   ``"filename"`` (e.g. ``"train_00000.png"``).  Train rows
                   also have a ``"label"`` key; test rows may omit it.
        transform: Optional torchvision transform applied to the PIL image.
    """

    def __init__(
        self,
        root: str,
        rows: List[Dict[str, str]],
        transform: Optional[Callable] = None,
        cache_images: bool = False,
    ):
        self.root      = Path(root)
        self.rows      = rows
        self.transform = transform

        # Determine image sub-directory from first filename
        first_fn = rows[0]["filename"] if rows else ""
        if first_fn.startswith("test"):
            self._img_dir = self.root / "test_images"
        else:
            self._img_dir = self.root / "train_images"
        self._images = None
        if cache_images:
            self._images = np.empty((len(rows), 32, 32, 3), dtype=np.uint8)
            for idx, row in enumerate(rows):
                folder = "test_images" if row["filename"].startswith("test") else "train_images"
                with Image.open(self.root / folder / row["filename"]) as image:
                    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
                    if pixels.shape != (32, 32, 3):
                        raise ValueError("Cached image must have shape 32x32x3")
                    self._images[idx] = pixels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row  = self.rows[idx]
        fn   = row["filename"]

        # Support both train_images/ and test_images/ filenames in same dataset
        if fn.startswith("test"):
            img_path = self.root / "test_images" / fn
        else:
            img_path = self.root / "train_images" / fn

        if self._images is None:
            with Image.open(img_path) as source:
                image = source.convert("RGB")
        else:
            image = Image.fromarray(self._images[idx])

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label"]) if "label" in row and row["label"] != "" else -1
        return image, label


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(mode: str, augment_policy: str = "baseline", randaugment_magnitude: int = 9) -> T.Compose:
    """Return a torchvision transform pipeline.

    Args:
        mode:            ``"train"``, ``"val"``, or ``"test"``.
        augment_policy:  One of ``"baseline"``, ``"randaugment"``, or
                         ``"strong"`` (only relevant when *mode* is ``"train"``).

    Returns:
        A :class:`torchvision.transforms.Compose` pipeline.

    Policies (train only)
    ~~~~~~~~~~~~~~~~~~~~~
    ``baseline``
        RandomCrop(32, pad=4) → RandomHorizontalFlip → ToTensor → Normalize
    ``randaugment``
        baseline  +  RandAugment(num_ops=2, magnitude=9)
    ``strong``
        randaugment  +  RandomErasing(p=0.5)
    """
    normalize = T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)

    if mode in ("val", "test"):
        return T.Compose([
            T.ToTensor(),
            normalize,
        ])

    # ---- train ----
    base = [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
    ]

    if augment_policy == "randaugment":
        base.append(T.RandAugment(num_ops=2, magnitude=randaugment_magnitude))
    elif augment_policy == "strong":
        base.append(T.RandAugment(num_ops=2, magnitude=randaugment_magnitude))

    base.append(T.ToTensor())
    base.append(normalize)

    if augment_policy == "strong":
        base.append(T.RandomErasing(p=0.5))

    return T.Compose(base)


# ---------------------------------------------------------------------------
# CutMix / Mixup collate functions
# ---------------------------------------------------------------------------

def _rand_bbox(size: Tuple[int, ...], lam: float) -> Tuple[int, int, int, int]:
    """Sample a random bounding box for CutMix.

    Args:
        size: Tensor shape (N, C, H, W).
        lam:  Lambda drawn from Beta distribution.

    Returns:
        (bbx1, bby1, bbx2, bby2) corner coordinates.
    """
    H, W = size[2], size[3]
    cut_ratio = math.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cx = random.randint(0, W)
    cy = random.randint(0, H)

    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)
    return bbx1, bby1, bbx2, bby2


def cutmix_collate(
    batch: List[Tuple[torch.Tensor, int]],
    alpha: float = 1.0,
    probability: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """CutMix data-augmentation collate function (Yun et al., ICCV 2019).

    Args:
        batch: List of (image_tensor, label) pairs from the Dataset.
        alpha: Beta distribution concentration parameter.

    Returns:
        (mixed_images, labels_a, labels_b, lam)

    Usage in DataLoader::

        from functools import partial
        loader = DataLoader(ds, collate_fn=partial(cutmix_collate, alpha=1.0))

    Training loss::

        loss = lam * criterion(output, labels_a) + (1 - lam) * criterion(output, labels_b)
    """
    images, labels = zip(*batch)
    images = torch.stack(images)             # (N, C, H, W)
    labels = torch.tensor(labels, dtype=torch.long)

    if probability < 1.0 and random.random() >= probability:
        return images, labels, labels, 1.0

    lam = float(np.random.beta(alpha, alpha))
    n   = images.size(0)
    idx = torch.randperm(n)

    bbx1, bby1, bbx2, bby2 = _rand_bbox(images.size(), lam)
    mixed = images.clone()
    mixed[:, :, bby1:bby2, bbx1:bbx2] = images[idx, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda to actual pixel ratio
    lam = 1.0 - (bby2 - bby1) * (bbx2 - bbx1) / (images.size(2) * images.size(3))

    return mixed, labels, labels[idx], lam


def mixup_collate(
    batch: List[Tuple[torch.Tensor, int]],
    alpha: float = 0.4,
    probability: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup data-augmentation collate function (Zhang et al., ICLR 2018).

    Args:
        batch: List of (image_tensor, label) pairs from the Dataset.
        alpha: Beta distribution concentration parameter.

    Returns:
        (mixed_images, labels_a, labels_b, lam)
    """
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)

    if probability < 1.0 and random.random() >= probability:
        return images, labels, labels, 1.0

    lam = float(np.random.beta(alpha, alpha))
    n   = images.size(0)
    idx = torch.randperm(n)

    mixed = lam * images + (1.0 - lam) * images[idx]
    return mixed, labels, labels[idx], lam
