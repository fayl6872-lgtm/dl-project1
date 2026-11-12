"""
train.py
========
Full training script for the CIFAR-10 deep learning project.

Usage examples
--------------
  # Quick baseline
  python train.py --model preact --epochs 100

  # SE-ResNet with strong augmentation, CutMix, and label smoothing
  python train.py --model se --augment strong --cutmix --label-smoothing 0.1 \\
                  --run-name se_cutmix_strong

  # WideResNet with Mixup on Apple Silicon
  python train.py --model wide --augment randaugment --mixup --device mps

All outputs are written to  outputs/<run_name>_<timestamp>/
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Local imports
from data_utils import (
    CIFAR10Dataset,
    cutmix_collate,
    get_transforms,
    mixup_collate,
    read_csv,
    stratified_split,
)
from models.resnet import BaselineResNet, PreActResNet, SEResNet, WideResNet


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a ResNet variant on CIFAR-10 (from scratch, ≤5M params)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",         type=str,   default="../data",
                   help="Path to data directory containing train_images/, test_images/, etc.")
    p.add_argument("--model",            type=str,   default="preact",
                   choices=["baseline", "preact", "se", "wide"],
                   help="Model architecture to train.")
    p.add_argument("--widths", type=int, nargs=3, default=None,
                   help="Three stage widths for preact/se/wide; parameter limit still applies.")
    p.add_argument("--blocks-per-stage", type=int, nargs=3, default=None,
                   help="Residual block counts in the three stages for preact/se/wide.")
    p.add_argument("--augment",          type=str,   default="baseline",
                   choices=["baseline", "randaugment", "strong"],
                   help="Training augmentation policy.")
    p.add_argument("--mixup",            action="store_true",
                   help="Enable Mixup augmentation (mutually exclusive with CutMix).")
    p.add_argument("--cutmix",           action="store_true",
                   help="Enable CutMix augmentation (mutually exclusive with Mixup).")
    p.add_argument("--label-smoothing",  type=float, default=0.0,
                   help="Label smoothing epsilon (0 = disabled).")
    p.add_argument("--stochastic-depth", type=float, default=0.0,
                   help="DropPath drop probability (0 = disabled).")
    p.add_argument("--epochs",           type=int,   default=100)
    p.add_argument("--batch-size",       type=int,   default=128)
    p.add_argument("--lr",               type=float, default=0.1,
                   help="Peak / initial learning rate.")
    p.add_argument("--weight-decay",     type=float, default=5e-4)
    p.add_argument("--optimizer",        type=str,   default="sgd",
                   choices=["sgd", "adamw"])
    p.add_argument("--scheduler",        type=str,   default="cosine",
                   choices=["cosine", "onecycle"])
    p.add_argument("--warmup-epochs",    type=int,   default=5,
                   help="Number of warm-up epochs (linear LR ramp).")
    p.add_argument("--tta",              action="store_true",
                   help="Use test-time augmentation (original + H-flip) during evaluation.")
    p.add_argument("--seed",             type=int,   default=6953)
    p.add_argument("--validation-fold", type=int, default=0)
    p.add_argument("--randaugment-magnitude", type=int, default=9)
    p.add_argument("--mix-probability", type=float, default=1.0)
    p.add_argument("--cutmix-alpha", type=float, default=1.0)
    p.add_argument("--mixup-alpha", type=float, default=0.4)
    p.add_argument("--split-seed",       type=int,   default=None,
                   help="Fixed train/validation split seed, independent of model initialization.")
    p.add_argument("--run-name",         type=str,   default="exp")
    p.add_argument("--output-dir",       type=str,   default="outputs",
                   help="Parent directory for timestamped run outputs.")
    p.add_argument("--num-workers",      type=int,   default=None,
                   help="DataLoader workers; 0 avoids multiprocessing/file-descriptor issues.")
    p.add_argument("--amp", action="store_true",
                   help="Use CUDA float16 mixed precision for training; validation stays float32.")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="Optional parameter EMA decay in (0, 1); 0 disables it. BN buffers copy the training model.")
    p.add_argument("--cache-images", action="store_true",
                   help="Cache decoded uint8 images in RAM; random augmentation remains online.")
    p.add_argument("--skip-submission", action="store_true",
                   help="Train/select on the validation split without predicting the hidden test set.")
    p.add_argument("--device",           type=str,   default="auto",
                   help="Device: 'auto', 'cpu', 'cuda', 'mps', etc.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "baseline": BaselineResNet,
    "preact":   PreActResNet,
    "se":       SEResNet,
    "wide":     WideResNet,
}


def build_model(args: argparse.Namespace) -> nn.Module:
    cls = MODEL_REGISTRY[args.model]
    kwargs: Dict = {"num_classes": 10}
    for name in ["widths", "blocks_per_stage"]:
        values = getattr(args, name, None)
        if values is not None:
            if args.model == "baseline" or any(value <= 0 for value in values):
                raise ValueError("Custom widths/blocks require a supported model and positive values")
            kwargs[name] = tuple(values)
    if args.stochastic_depth > 0:
        kwargs["drop_path_prob"] = args.stochastic_depth
    model = cls(**kwargs)
    return model


class ModelEMA:
    """Frozen single-model moving average, updated only after optimizer steps.

    The optimizer post-step hook is not invoked when GradScaler skips an
    optimizer step. No per-batch GPU-to-CPU scalar reads are needed here.
    Warm-up averages the first updates before using the configured decay.
    BatchNorm running buffers are copied from the current training model.
    """

    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be strictly between 0 and 1")
        self.source = model
        self.module = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, optimizer=None, args=None, kwargs=None):
        decay = min(self.decay, self.updates / (self.updates + 1.0))
        for shadow, value in zip(self.module.parameters(), self.source.parameters()):
            shadow.lerp_(value.detach(), 1.0 - decay)
        for shadow, value in zip(self.module.buffers(), self.source.buffers()):
            shadow.copy_(value)
        self.updates += 1


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def warmup_lr(epoch: int, warmup_epochs: int, base_lr: float, min_lr: float) -> float:
    """Linear warm-up from min_lr → base_lr over warmup_epochs."""
    if warmup_epochs <= 0:
        return base_lr
    frac = min(epoch / warmup_epochs, 1.0)
    return min_lr + frac * (base_lr - min_lr)


def cosine_lr(
    epoch: int,
    warmup_epochs: int,
    total_epochs: int,
    base_lr: float,
    min_lr: float = 1e-6,
) -> float:
    """Cosine decay after warm-up."""
    if epoch < warmup_epochs:
        return warmup_lr(epoch, warmup_epochs, base_lr, base_lr * 0.01)
    t = epoch - warmup_epochs
    T = max(total_epochs - warmup_epochs, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t / T))


def set_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ---------------------------------------------------------------------------
# Loss with optional label smoothing
# ---------------------------------------------------------------------------

def build_criterion(label_smoothing: float) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


# ---------------------------------------------------------------------------
# Single train epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_mixup: bool,
    use_cutmix: bool,
    scaler=None,
    step_scheduler=None,
) -> Tuple[float, float]:
    """Run one training epoch.

    Returns:
        (avg_loss, accuracy_fraction)
    """
    model.train()
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), dtype=torch.int64, device=device)
    total_samples = 0
    amp_enabled = scaler is not None and scaler.is_enabled()

    for batch in loader:
        if use_mixup or use_cutmix:
            # Batch is (mixed_images, labels_a, labels_b, lam)
            images, labels_a, labels_b, lam = batch
            images   = images.to(device, non_blocking=True)
            labels_a = labels_a.to(device, non_blocking=True)
            labels_b = labels_b.to(device, non_blocking=True)

            dominant_labels = labels_a if lam >= 0.5 else labels_b
        else:
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            dominant_labels = labels

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            outputs = model(images)
            if use_mixup or use_cutmix:
                loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)
        advanced = True
        if amp_enabled:
            old_scale = scaler.get_scale() if step_scheduler is not None else None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if old_scale is not None:
                advanced = scaler.get_scale() >= old_scale
        else:
            loss.backward()
            optimizer.step()
        if step_scheduler is not None and advanced:
            step_scheduler.step()

        total_loss += loss.detach() * images.size(0)
        total_correct += (outputs.argmax(dim=1) == dominant_labels).sum()
        total_samples += images.size(0)

    return total_loss.item() / total_samples, total_correct.item() / total_samples


# ---------------------------------------------------------------------------
# Evaluation (val or test)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tta: bool = False,
) -> Tuple[float, float]:
    """Evaluate model on a DataLoader.

    Args:
        model:    Trained model.
        loader:   DataLoader (labels must be valid integers).
        criterion: Loss function.
        device:   Torch device.
        tta:      If True, average logits over (original, h-flip).

    Returns:
        (avg_loss, accuracy_fraction)
    """
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), dtype=torch.int64, device=device)
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        if tta:
            flipped = torch.flip(images, dims=[3])
            logits  = (logits + model(flipped)) * 0.5

        loss = criterion(logits, labels)
        total_loss    += loss * images.size(0)
        preds          = logits.argmax(dim=1)
        total_correct  += (preds == labels).sum()
        total_samples  += images.size(0)

    return total_loss.item() / total_samples, total_correct.item() / total_samples


# ---------------------------------------------------------------------------
# Generate test-set submission
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_submission(
    model: nn.Module,
    test_rows: List[Dict],
    data_dir: str,
    out_path: str,
    device: torch.device,
    batch_size: int = 256,
    tta: bool = False,
) -> None:
    """Generate submission.csv from the best model checkpoint.

    Args:
        model:      Loaded best model (already on *device*).
        test_rows:  Row-dicts from test.csv.
        data_dir:   Data root directory.
        out_path:   Path where submission.csv will be written.
        device:     Torch device.
        batch_size: Inference batch size.
        tta:        Enable test-time augmentation.
    """
    from data_utils import CIFAR10Dataset, get_transforms

    transform = get_transforms("test")
    dataset   = CIFAR10Dataset(data_dir, test_rows, transform=transform)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=False)

    model.eval()
    all_preds: List[int] = []

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if tta:
            flipped = torch.flip(images, dims=[3])
            logits  = (logits + model(flipped)) * 0.5
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for row, pred in zip(test_rows, all_preds):
            writer.writerow([row["id"], pred])

    print(f"[Submission] Saved {len(all_preds)} predictions → {out_path}")


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Validation: mixup and cutmix are mutually exclusive
    if args.mixup and args.cutmix:
        raise ValueError("--mixup and --cutmix are mutually exclusive. Choose one.")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in [0, 1)")
    if not 0 <= args.validation_fold < 10 or not 0 <= args.randaugment_magnitude <= 30:
        raise ValueError('Invalid validation fold or RandAugment magnitude')
    if not 0 <= args.mix_probability <= 1 or min(args.cutmix_alpha, args.mixup_alpha) <= 0:
        raise ValueError('Invalid mixing probability or alpha')

    set_seed(args.seed)
    device = select_device(args.device)
    print(f"[Device] Using {device}")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp currently requires CUDA")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    # ------------------------------------------------------------------ #
    # Output directory
    # ------------------------------------------------------------------ #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = Path(args.output_dir) / f"{args.run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Run] Output directory: {run_dir}")

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    train_csv_path = os.path.join(args.data_dir, "train_labels.csv")
    test_csv_path  = os.path.join(args.data_dir, "test.csv")

    all_train_rows = read_csv(train_csv_path)
    test_rows      = read_csv(test_csv_path)

    split_seed = args.seed if args.split_seed is None else args.split_seed
    train_rows, val_rows = stratified_split(all_train_rows, val_frac=0.1, seed=split_seed, fold=args.validation_fold)
    print(f"[Data] Train: {len(train_rows)}, Val: {len(val_rows)}, Test: {len(test_rows)}")

    train_transform = get_transforms("train", augment_policy=args.augment, randaugment_magnitude=args.randaugment_magnitude)
    val_transform   = get_transforms("val")

    train_ds = CIFAR10Dataset(args.data_dir, train_rows, transform=train_transform, cache_images=args.cache_images)
    val_ds   = CIFAR10Dataset(args.data_dir, val_rows, transform=val_transform, cache_images=args.cache_images)

    # Collate function for mixing augmentations
    collate_fn = None
    if args.cutmix:
        collate_fn = partial(cutmix_collate, alpha=args.cutmix_alpha, probability=args.mix_probability)
        print(f"[Augment] CutMix alpha={args.cutmix_alpha}; probability={args.mix_probability}")
    elif args.mixup:
        collate_fn = partial(mixup_collate, alpha=args.mixup_alpha, probability=args.mix_probability)
        print(f"[Augment] Mixup alpha={args.mixup_alpha}; probability={args.mix_probability}")

    num_workers = args.num_workers if args.num_workers is not None else min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model = build_model(args).to(device)
    n_params = model.count_params()
    print(f"[Model] {args.model} | {n_params:,} parameters")
    assert n_params <= 5_000_000, f"Model has {n_params:,} params – exceeds 5 M limit!"

    # ------------------------------------------------------------------ #
    # Optimizer & scheduler
    # ------------------------------------------------------------------ #
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    else:  # adamw
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    # Scheduler is handled manually per epoch (warmup + cosine / onecycle)
    if args.scheduler == "onecycle":
        # PyTorch's OneCycleLR works at the step level
        scheduler_obj = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=min(0.99, max(0.01, args.warmup_epochs / args.epochs)),
            anneal_strategy="cos",
            div_factor=100.0,
            final_div_factor=1e4,
        )
        use_step_scheduler = True
    else:
        scheduler_obj       = None
        use_step_scheduler  = False

    criterion = build_criterion(args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    ema_hook = optimizer.register_step_post_hook(ema.update) if ema is not None else None
    if ema is not None:
        assert sum(p.numel() for p in ema.module.parameters() if p.requires_grad) == 0
        print(f"[EMA] decay={args.ema_decay}; frozen shadow; final prediction uses one model")

    # ------------------------------------------------------------------ #
    # Save config
    # ------------------------------------------------------------------ #
    config = vars(args)
    config["param_count"]  = n_params
    config["device"]       = str(device)
    config["run_dir"]      = str(run_dir)
    config["split_seed"] = split_seed
    config["training_precision"] = "amp_float16" if args.amp else "float32"
    config["validation_precision"] = "float32"
    config["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    config["deterministic_fp32"] = True
    if ema is not None:
        config["ema_bn_policy"] = "copy training-model running buffers after successful optimizer steps"
        config["ema_shadow_trainable_params"] = 0
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    history_path = run_dir / "history.csv"
    history_header = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_seconds"]
    if ema is not None:
        history_header += ["raw_val_loss", "raw_val_acc", "ema_val_loss", "ema_val_acc", "selected_weights", "ema_updates"]

    best_val_acc = 0.0
    best_ckpt    = run_dir / "best.pt"          # better of raw / EMA at its best epoch
    # With EMA, the best raw and best EMA weights are also kept separately, so a
    # submission can name exactly which weights it used.
    best_by_source = {"raw": 0.0, "ema": 0.0}
    latest_ckpt  = run_dir / "latest.pt"

    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(history_header)

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8} | {'Time':>7}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- Learning rate ---
        if use_step_scheduler:
            current_lr = scheduler_obj.get_last_lr()[0]
        else:
            current_lr = cosine_lr(epoch - 1, args.warmup_epochs, args.epochs, args.lr)
            set_lr(optimizer, current_lr)

        # --- Train ---
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            use_mixup=args.mixup, use_cutmix=args.cutmix,
            scaler=scaler,
            step_scheduler=scheduler_obj if use_step_scheduler else None,
        )

        # --- Validate ---
        raw_val_loss, raw_val_acc = evaluate(model, val_loader, criterion, device, tta=args.tta)
        val_loss, val_acc = raw_val_loss, raw_val_acc
        selected_model, selected_weights = model, "raw"
        metrics = [train_loss, train_acc, raw_val_loss, raw_val_acc]
        if ema is not None:
            ema_val_loss, ema_val_acc = evaluate(ema.module, val_loader, criterion, device, tta=args.tta)
            metrics += [ema_val_loss, ema_val_acc]
            if ema_val_acc > raw_val_acc:
                val_loss, val_acc = ema_val_loss, ema_val_acc
                selected_model, selected_weights = ema.module, "ema"
        if not all(math.isfinite(value) for value in metrics):
            raise FloatingPointError("Non-finite training or validation metric")

        elapsed = time.time() - t0

        # --- Checkpoints ---
        latest_state = {"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "val_acc": raw_val_acc}
        if ema is not None:
            latest_state.update({"weights_source": "raw", "ema_model": ema.module.state_dict(),
                                 "ema_updates": ema.updates, "ema_val_acc": ema_val_acc})
        torch.save(latest_state, latest_ckpt)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {"epoch": epoch, "model": selected_model.state_dict(), "val_acc": val_acc}
            if ema is not None:
                best_state.update({"weights_source": selected_weights, "ema_updates": ema.updates})
            torch.save(best_state, best_ckpt)

        if ema is not None:
            for source, acc, module in (("raw", raw_val_acc, model), ("ema", ema_val_acc, ema.module)):
                if acc > best_by_source[source]:
                    best_by_source[source] = acc
                    torch.save({"epoch": epoch, "model": module.state_dict(), "val_acc": acc,
                                "weights_source": source, "ema_updates": ema.updates},
                               run_dir / f"best_{source}.pt")

        # --- Logging ---
        history_values = [
                epoch,
                f"{train_loss:.4f}",
                f"{train_acc:.4f}",
                f"{val_loss:.4f}",
                f"{val_acc:.4f}",
                f"{current_lr:.6f}",
                f"{elapsed:.1f}",
            ]
        if ema is not None:
            history_values += [f"{raw_val_loss:.4f}", f"{raw_val_acc:.4f}",
                               f"{ema_val_loss:.4f}", f"{ema_val_acc:.4f}", selected_weights, ema.updates]
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow(history_values)

        print(
            f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.2%} | "
            f"{val_loss:>8.4f} | {val_acc:>6.2%} | {current_lr:>8.6f} | {elapsed:>5.1f}s"
        )
        if ema is not None:
            print(f"      [EMA check] raw={raw_val_acc:.2%}; ema={ema_val_acc:.2%}; selected={selected_weights}; updates={ema.updates}")

    if ema_hook is not None:
        ema_hook.remove()
    print(f"\n[Done] Best val acc: {best_val_acc:.2%}  |  checkpoint: {best_ckpt}")
    if args.skip_submission:
        print(f"[Complete] Validation-only run saved in {run_dir}; no hidden-test inference.")
        return

    # ------------------------------------------------------------------ #
    # Generate submission from best checkpoint
    # ------------------------------------------------------------------ #
    best_state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(best_state["model"])
    submission_path = str(run_dir / "submission.csv")
    generate_submission(
        model, test_rows, args.data_dir, submission_path,
        device=device, batch_size=args.batch_size * 2, tta=args.tta,
    )

    print(f"[Complete] Results saved in {run_dir}")


if __name__ == "__main__":
    main()
