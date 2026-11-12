"""All-data refits (B1, B2): train on all 50,000 images (no validation set), keep one fixed epoch,
and write a flip-TTA submission.

Each recipe matches its 45k/5k development recipe; the saved epoch is the median best epoch
of the three validation folds.

    python train_full_data.py --recipe b1_deep_narrow  --output-dir runs/b1_deep_narrow
    python train_full_data.py --recipe b2_wide_shallow --output-dir runs/b2_wide_shallow
    python train_full_data.py --recipe b3_large        --output-dir runs/b3_large_full
"""

import argparse
import time
from functools import partial
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data_utils import CIFAR10Dataset, cutmix_collate, get_transforms, read_csv
from models.resnet import WideResNet
from train import build_criterion, cosine_lr, generate_submission, set_lr, set_seed, train_epoch

# Shared by all recipes (identical to the fold experiments, minus the validation split).
EPOCHS = 300          # cosine schedule length
SEED = 6953
LABEL_SMOOTHING = 0.05
CUTMIX_PROBABILITY = 0.5
RANDAUGMENT_MAGNITUDE = 9

RECIPES = {
    # B1: deep/narrow
    "b1_deep_narrow":  dict(widths=(56, 112, 224), blocks=(4, 4, 4), augment="randaugment", drop_path=0.05, save_epoch=295),
    # B2: wide/shallow
    "b2_wide_shallow": dict(widths=(80, 160, 320), blocks=(2, 2, 2), augment="randaugment", drop_path=0.0,  save_epoch=298),
    # B3 recipe (larger, + RandomErasing) refit on all data
    "b3_large":        dict(widths=(64, 128, 256), blocks=(4, 4, 3), augment="strong",      drop_path=0.05, save_epoch=294),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    parser.add_argument("--data-dir", default="../data", help="Dataset directory")
    parser.add_argument("--output-dir", required=True, help="Where final.pt, history.csv and submission.csv go")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    recipe = RECIPES[args.recipe]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    device = torch.device(args.device)

    # All 50,000 images are used for training; there is no validation set.
    rows = read_csv(f"{args.data_dir}/train_labels.csv")
    tests = read_csv(f"{args.data_dir}/test.csv")
    print(f"[Data] Train: {len(rows)}, Test: {len(tests)}")

    dataset = CIFAR10Dataset(args.data_dir, rows,
                             transform=get_transforms("train", recipe["augment"], RANDAUGMENT_MAGNITUDE),
                             cache_images=True)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=True,
                        collate_fn=partial(cutmix_collate, alpha=1.0, probability=CUTMIX_PROBABILITY))

    model = WideResNet(num_classes=10, widths=recipe["widths"], blocks_per_stage=recipe["blocks"],
                       drop_path_prob=recipe["drop_path"]).to(device)
    n_params = model.count_params()
    print(f"[Model] WideResNet {args.recipe} | {n_params:,} parameters")
    assert n_params <= 5_000_000

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    criterion = build_criterion(LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda")

    history = out / "history.csv"
    with history.open("w", newline="") as f:
        f.write("epoch,train_loss,train_acc,lr,epoch_seconds\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        lr = cosine_lr(epoch - 1, 5, EPOCHS, 0.1)
        set_lr(optimizer, lr)

        loss, acc = train_epoch(model, loader, optimizer, criterion, device,
                                use_mixup=False, use_cutmix=True,
                                scaler=scaler, step_scheduler=None)
        print(f"[Epoch {epoch:>3}/{EPOCHS}] loss={loss:.4f} acc={acc:.4f} lr={lr:.5f} ({time.time() - t0:.0f}s)",
              flush=True)

        with history.open("a", newline="") as f:
            f.write(f"{epoch},{loss:.4f},{acc:.4f},{lr:.6f},{time.time() - t0:.1f}\n")

        if epoch == recipe["save_epoch"]:
            torch.save({"epoch": epoch, "model": model.state_dict()}, out / "final.pt")
            print(f"[Save] epoch {epoch} weights -> final.pt")

    # Predict with the fixed-epoch raw weights, averaging logits of original + horizontal flip.
    ckpt = torch.load(out / "final.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    generate_submission(model, tests, args.data_dir, str(out / "submission.csv"),
                        device, batch_size=256, tta=True)
    print("[Done] submission.csv written")


if __name__ == "__main__":
    main()
