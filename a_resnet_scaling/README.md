# A · Scaling a ResNet: A0 → A3

Each step has a 45k/5k **development** run (selects the recipe and inference policy, reports
validation accuracy) and an **all-data** refit on 50k images (the submitted model).

| Step | Change | Dev. val. |
|---|---|---:|
| A0 | ResNet-18 layout, widths 32–256, crop + flip, 100 ep | 94.54 |
| A1 | widths 40–320; Mixup (α 0.2) / CutMix (α 1.0) on 80% of batches, off for the last 20 ep; EMA; 200 ep | 96.44 |
| A2 | 3-stage preact WRN 56/112/224 ×4 (8×8 final map); RandAugment (2, 9) + Cutout 16; EMA 0.995; BN recalibration; 300 ep | 97.30 |
| A3 | A2 continued 80 ep with SAM (ρ 0.05, lr 0.03) | 97.58 |

Dev. val. uses flip TTA, except A0 (no TTA).

## Files

```text
prepare_data.py             validate dataset files, cache images as uint8 .npy, fixed seed-42 45k/5k split
common.py                   cached image bank, GPU crop/flip, evaluation, atomic checkpointing
resnet.py                   A0/A1: 4-stage CIFAR ResNet with configurable width
wide_resnet.py              A2/A3: 3-stage preactivation Wide ResNet (16-channel stem)
augment_mixing.py           Cutout, Mixup, CutMix
randaugment_data.py         per-image RandAugment with deterministic per-epoch seeds
augment_degradation.py      noise / blur / resampling / brightness (robustness probe)
sam.py                      two-pass SAM update
train_a0_baseline.py        A0
train_a1_wider_resnet.py    A1
train_a2_wide_resnet.py     A2
train_a3_sam.py             A3
train_degradation_probe.py  robustness probe
extract_final_weights.py    split a finished run's last.pt into final_raw/ and final_ema/ checkpoints
recalibrate_bn.py           recompute BatchNorm statistics on augmented training images
evaluate_and_export.py      validation under TTA policies, or write a submission CSV
evaluate_degradations.py    validation accuracy under fixed synthetic degradations
configs/                    one JSON per run
tests/                      python -m unittest discover -s tests
```

## Setup

Python 3.12, PyTorch 2.7.1, torchvision 0.22.1 (original runs: one RTX 4090). Run everything from
this folder; outputs go to `runs/<name>/`.

```bash
pip install -r requirements.txt
python prepare_data.py          # reads ../data, writes cache/ and splits/
```

## A0
```bash
python train_a0_baseline.py
python evaluate_and_export.py --checkpoint runs/a0_baseline/best.pt
python evaluate_and_export.py --checkpoint runs/a0_baseline/best.pt --policy none --export submissions/a0.csv
```

## A1
```bash
python train_a1_wider_resnet.py --config configs/a1_dev_cutout.json --output runs/a1_dev_cutout    # 96.24
python train_a1_wider_resnet.py --config configs/a1_dev_mixing.json --output runs/a1_dev_mixing    # 96.44, selected
python evaluate_and_export.py --checkpoint runs/a1_dev_mixing/best.pt
python train_a1_wider_resnet.py --config configs/a1_full_mixing.json --output runs/a1_full_mixing --all-data
python evaluate_and_export.py --checkpoint runs/a1_full_mixing/final.pt --policy flip --export submissions/a1.csv
```

## A2
```bash
python train_a2_wide_resnet.py --config configs/a2_dev_wrn.json --output runs/a2_dev_wrn
python extract_final_weights.py --run runs/a2_dev_wrn
python recalibrate_bn.py --checkpoint runs/a2_dev_wrn/final_ema/best.pt --output runs/a2_dev_wrn/final_ema/calibrated/best.pt
python evaluate_and_export.py --checkpoint runs/a2_dev_wrn/final_ema/calibrated/best.pt              # 97.30, selected
python train_a2_wide_resnet.py --config configs/a2_full_wrn.json --output runs/a2_full_wrn --all-data
python recalibrate_bn.py --checkpoint runs/a2_full_wrn/final.pt --output runs/a2_full_wrn/calibrated/final.pt
python evaluate_and_export.py --checkpoint runs/a2_full_wrn/calibrated/final.pt --policy flip --export submissions/a2.csv
```
Selection compared best vs final epoch, raw vs EMA weights, with and without BN recalibration.

## A3
```bash
python train_a3_sam.py --config configs/a3_dev_sam.json --output runs/a3_dev_sam     # parent: runs/a2_dev_wrn/final_ema
python extract_final_weights.py --run runs/a3_dev_sam
python evaluate_and_export.py --checkpoint runs/a3_dev_sam/final_ema/best.pt          # 97.58, selected
python train_a3_sam.py --config configs/a3_full_sam.json --output runs/a3_full_sam --all-data
python evaluate_and_export.py --checkpoint runs/a3_full_sam/final.pt --policy flip --export submissions/a3.csv
```
The all-data SAM run continues `runs/a2_full_wrn/final.pt` (before BN recalibration).

## Robustness probe

Continue the A3 model with random degradations on 35% of images, and measure accuracy on validation
images under fixed noise, blur, down/up-sampling and brightness changes.

```bash
python evaluate_degradations.py --checkpoint runs/a3_dev_sam/final_ema/best.pt --output runs/a3_dev_sam/degradation_eval.json
python train_degradation_probe.py --config configs/probe_dev_degradation.json --output runs/probe_dev_degradation
python extract_final_weights.py --run runs/probe_dev_degradation
python evaluate_degradations.py --checkpoint runs/probe_dev_degradation/best.pt --output runs/probe_dev_degradation/degradation_eval.json
python evaluate_degradations.py --checkpoint runs/probe_dev_degradation/final_ema/best.pt --output runs/probe_dev_degradation/final_ema/degradation_eval.json
```

| Checkpoint | Clean | Degraded (mean of 4) |
|---|---:|---:|
| A3 | 97.58 | 94.65 |
| + 1 epoch of degradation training | 97.32 | 95.93 |
| + 100 epochs | 96.88 | 96.30 |

## Notes

- Mixed precision is bfloat16; cuDNN autotuning is on, so reruns match statistically, not bit-for-bit.
- Flip TTA averages softmax probabilities of the original and flipped image.
- Training scripts copy their own sources into `runs/<name>/source/` and refuse `--resume` if the code changed.
