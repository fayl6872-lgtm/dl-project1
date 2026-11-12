"""Pick the epoch for an all-data refit from fold runs of the same recipe.

For each run directory, read history.csv, take the epoch with the highest raw-weight validation
accuracy, and report the median of those epochs (the epoch train_full_data.py saves).

    python select_full_data_epoch.py outputs/b1_fold0_* outputs/b1_fold1_* outputs/b1_fold2_*
"""
import argparse
import csv
import statistics
from pathlib import Path


def best_raw_epoch(run_dir):
    with (Path(run_dir) / "history.csv").open() as f:
        rows = list(csv.DictReader(f))
    column = "raw_val_acc" if "raw_val_acc" in rows[0] else "val_acc"   # raw column exists when EMA is on
    best = max(rows, key=lambda r: (float(r[column]), -int(r["epoch"])))
    return int(best["epoch"]), float(best[column]), len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Fold run directories containing history.csv")
    runs = parser.parse_args().runs
    epochs = []
    print(f"{'run':<50} {'best epoch':>10} {'raw val acc':>12} {'epochs run':>11}")
    for run in runs:
        epoch, acc, total = best_raw_epoch(run)
        epochs.append(epoch)
        print(f"{Path(run).name:<50} {epoch:>10} {acc:>12.2%} {total:>11}")
    print(f"\nmedian best epoch -> save epoch {int(statistics.median(epochs))}")


if __name__ == "__main__":
    main()
