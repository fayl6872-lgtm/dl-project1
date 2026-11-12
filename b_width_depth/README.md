# B · Width vs depth at a fixed budget: B1 → B7

All development runs use one script, `train.py`, configured by flags. `train_full_data.py` refits a
selected recipe on all 50,000 images.

```text
models/resnet.py      BaselineResNet, PreActResNet, SEResNet, WideResNet (configurable widths / blocks)
data_utils.py         PNG dataset, stratified 10% validation folds (seed 6953), RandAugment,
                      RandomErasing, CutMix / Mixup collate functions
train.py              45k/5k training with warmup + cosine, AMP, EMA, stochastic depth, flip TTA
train_full_data.py    50k refits: --recipe b1_deep_narrow | b2_wide_shallow | b3_large
predict.py            flip-TTA submission from a saved checkpoint
select_full_data_epoch.py   median best epoch over fold runs (sets the epoch an all-data refit saves)
```

## Setup

PyTorch 2.8.0, torchvision 0.23.0. Run from this folder; data is read from `../data`.

```bash
pip install -r requirements.txt
```

Development runs write `outputs/<run-name>_<timestamp>/`:

| File | Content |
|---|---|
| `history.csv` | per-epoch train loss/acc and validation loss/acc (raw and EMA columns when EMA is on) |
| `best.pt` | the better of raw / EMA weights at the best validation epoch (`weights_source` records which) |
| `best_raw.pt`, `best_ema.pt` | best raw and best EMA weights separately (EMA runs only) |
| `latest.pt` | last epoch, with optimizer state |

With `--skip-submission`, make the CSV afterwards from a checkpoint:

```bash
python predict.py --checkpoint outputs/<run>/best_raw.pt --widths 64 128 256 --blocks-per-stage 4 4 3 --output submissions/b3.csv
```

The submitted B3 and B4 checkpoints are raw weights (`weights_source: raw` in the saved file; see
`../results/b_checkpoints.json`).

## Shared recipe

SGD (Nesterov, momentum 0.9, wd 5e-4), lr 0.1, batch 128, 5 warmup epochs + cosine, 300 epochs,
RandAugment (2 ops), CutMix (α 1.0), EMA 0.999 tracked during development, flip TTA (averaged logits).
Submitted weights are raw (non-EMA).

```bash
COMMON="--model wide --epochs 300 --batch-size 128 --lr 0.1 --scheduler cosine --warmup-epochs 5 \
  --weight-decay 0.0005 --cutmix --cutmix-alpha 1.0 --ema-decay 0.999 --seed 6953 --split-seed 6953 \
  --num-workers 2 --device cuda:0 --tta --amp --cache-images --skip-submission"
```

## Width vs depth (B1–B3)

| ID | Widths × blocks | Params | Extra |
|---|---|---:|---|
| B1 | 56/112/224 × (4,4,4) | 4,500,898 | stochastic depth 0.05 |
| B2 | 80/160/320 × (2,2,2) | 4,336,410 | — |
| B3 | 64/128/256 × (4,4,3) | 4,696,394 | stochastic depth 0.05, RandomErasing 0.5 |

All three: label smoothing 0.05, RandAugment M=9, CutMix p=0.5.

```bash
# B1, B2: refit on all data; the saved epoch (295, 298) is the median best epoch over three folds
python train_full_data.py --recipe b1_deep_narrow  --output-dir runs/b1_deep_narrow
python train_full_data.py --recipe b2_wide_shallow --output-dir runs/b2_wide_shallow

# B3: fold 0; submitted weights = raw, epoch 289, val 96.82
python train.py $COMMON --widths 64 128 256 --blocks-per-stage 4 4 3 --label-smoothing 0.05 \
  --augment strong --randaugment-magnitude 9 --mix-probability 0.5 --stochastic-depth 0.05 \
  --validation-fold 0 --run-name b3_large_fold0
```

## Regularization variant and repeats (B4–B7)

```bash
# B4: deep/narrow, milder RandAugment, stronger CutMix and smoothing; submitted raw epoch 297, val 96.90
python train.py $COMMON --widths 56 112 224 --blocks-per-stage 4 4 4 --label-smoothing 0.1 \
  --augment randaugment --randaugment-magnitude 7 --mix-probability 0.75 --stochastic-depth 0.05 \
  --validation-fold 0 --run-name b4_deep_fold0

```

B5 (B4 recipe), B6 and B7 (B3 recipe) are separate runs of these recipes. Their exact commands
were not preserved; the submission file names indicate validation folds 1 (B5) and 2 (B6).

The B3 recipe can also be refit on all data (`--recipe b3_large`, saved epoch 294).

## Choosing the refit epoch (B1, B2)

The saved epochs (295 for B1, 298 for B2) are fixed in `train_full_data.py`. They were chosen as the
median best epoch over three validation folds; those fold logs were not preserved. To repeat the
selection, train the recipe on folds 0–2 (settings below are the all-data recipe's) and run:

```bash
for k in 0 1 2; do
  python train.py $COMMON --widths 56 112 224 --blocks-per-stage 4 4 4 --label-smoothing 0.05 \
    --augment randaugment --randaugment-magnitude 9 --mix-probability 0.5 --stochastic-depth 0.05 \
    --validation-fold $k --run-name b1_fold$k
done
python select_full_data_epoch.py outputs/b1_fold0_* outputs/b1_fold1_* outputs/b1_fold2_*
```

For B2 use `--widths 80 160 320 --blocks-per-stage 2 2 2 --stochastic-depth 0`.

## Recipe development

Validation on fold 0; each step changes the previous one.

| Step | Change | Val. |
|---|---|---:|
| 1 | WRN 80/160/320 × (2,2,2), RandAugment M=9, CutMix, LS 0.05, 100 ep | 95.70 |
| 2 | 150 ep | 96.08 |
| 3 | 200 ep, no label smoothing (several changes at once) | 96.66 |
| 4 | step 2 re-run as a control | 96.32 |
| 5 | 300 ep, CutMix on 50% of batches, EMA 0.999 | 96.54 |

```bash
# step 1 (step 2: --epochs 150)
python train.py --model wide --augment randaugment --epochs 100 --batch-size 128 --lr 0.1 --scheduler cosine \
  --warmup-epochs 5 --weight-decay 0.0005 --label-smoothing 0.05 --cutmix --amp --cache-images
# step 3
python train.py --model wide --augment randaugment --epochs 200 --batch-size 128 --lr 0.1 --scheduler cosine \
  --warmup-epochs 5 --weight-decay 0.0005 --label-smoothing 0.0 --cutmix --amp
# step 4
python train.py --model wide --widths 80 160 320 --blocks-per-stage 2 2 2 --epochs 150 --batch-size 128 \
  --lr 0.1 --scheduler cosine --warmup-epochs 5 --weight-decay 0.0005 --label-smoothing 0.05 \
  --augment randaugment --randaugment-magnitude 9 --cutmix --mix-probability 1.0 \
  --seed 6953 --split-seed 6953 --validation-fold 0 --tta --amp --cache-images --skip-submission
# step 5
python train.py --model wide --widths 80 160 320 --blocks-per-stage 2 2 2 --epochs 300 --batch-size 128 \
  --lr 0.1 --scheduler cosine --warmup-epochs 5 --weight-decay 0.0005 --label-smoothing 0.05 \
  --augment randaugment --randaugment-magnitude 9 --cutmix --mix-probability 0.5 --ema-decay 0.999 \
  --seed 6953 --split-seed 6953 --validation-fold 0 --tta --amp --cache-images --skip-submission
```

## Notes

- Fold *k* holds out the *k*-th 10% slice of each shuffled class; fold 0 equals the plain 90/10 split.
- `train.py` sets deterministic cuDNN; steps 1–3 were originally run with `cudnn.benchmark=True`.
