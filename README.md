# Residual Networks under a 5M-Parameter Budget (CIFAR-10)

Image classification on 32×32 CIFAR-10-style images with a hard limit of **5,000,000 trainable
parameters**. All models are trained from scratch on the 50,000 provided training images; no
pretrained weights or external data.

Two experiment pipelines:

- **A · `a_resnet_scaling/`** — scale one network step by step: ResNet-18-style baseline → wider
  ResNet with Mixup/CutMix → three-stage preactivation Wide ResNet (8×8 final feature map) → SAM
  continuation.
- **B · `b_width_depth/`** — fix a Wide ResNet recipe, then compare how the same budget is spent:
  wide/shallow vs deep/narrow vs a larger hybrid, with fold-based model selection.

## Results

| ID | Model | Params | Val. | Public | Private |
|---|---|---:|---:|---:|---:|
| A0 | ResNet-18 style 32–256, crop/flip, 100 ep | 2,797,610 | 94.54 | 82.5 | 80.0 |
| A1 | Wider ResNet 40–320, Mixup/CutMix, 200 ep | 4,368,690 | 96.44 | 85.9 | 84.1 |
| A2 | WRN 56/112/224 ×4, RandAugment + Cutout, EMA, 300 ep | 4,480,474 | 97.30 | 88.3 | 85.3 |
| A3 | A2 + 80 epochs of SAM | 4,480,474 | 97.58 | 88.6 | 86.2 |
| B1 | Deep/narrow WRN 56/112/224 ×(4,4,4), 300 ep | 4,500,898 | — | 88.4 | **86.3** |
| B2 | Wide/shallow WRN 80/160/320 ×(2,2,2), 300 ep | 4,336,410 | — | 88.7 | 85.4 |
| B3 | Larger WRN 64/128/256 ×(4,4,3) + RandomErasing | 4,696,394 | 96.82 | 88.9 | 84.1 |
| B4 | Deep/narrow, RandAugment M=7, CutMix 0.75, LS 0.1 | 4,500,898 | 96.90 | 88.5 | 85.4 |
| B5 | B4 recipe, separate run¹ | 4,500,898 | — | 87.9 | 86.0 |
| B6 | B3 recipe, separate run¹ | 4,696,394 | — | 88.8 | 84.2 |
| B7 | B3 recipe, separate run¹ | 4,696,394 | — | 87.9 | 85.1 |

Accuracy in %. *Val.* is the held-out 5,000 images of the 45k/5k development run of the same recipe
(pipelines A and B use different splits). Models refit on all 50,000 images (A1–A3, B1, B2) have no
held-out score. *Public/Private* are the recorded leaderboard scores on the two halves of the
competition test set (test labels are hidden).
¹ Only the recipe and leaderboard scores are recorded for B5–B7; see *Evidence status*.

## Results notebook

[`notebooks/results.ipynb`](notebooks/results.ipynb) (executed; runs on CPU without the dataset)
prints the parameter count of every architecture and the final test accuracy, plots the training
and validation curves, and shows how checkpoints and test-time augmentation were selected. It
reads the logs and submission files in `results/` (needs `torch`, `numpy`, `matplotlib`, `jupyter`):

```text
results/
  leaderboard.csv     validation, public and private accuracy of every submission
  a_logs/             per-epoch history, summaries and evaluation JSON of every pipeline-A run
  a_selection/        candidate comparison behind each pipeline-A model choice
  b_checkpoints.json  epoch, raw/EMA, parameter count and SHA-256 of each submitted B checkpoint
  b_submissions/      submitted prediction files of B1-B4 and the B3 all-data refit
  figures/            plots written by the notebook
```

## Evidence status

| Experiments | Preserved artifacts | Not preserved |
|---|---|---|
| A0–A3, robustness probe | code, configs, per-epoch logs, candidate/TTA selection records (`results/a_logs`, `results/a_selection`) | weights, submission files |
| B1, B2 | all-data training script (saved epochs 295, 298 are fixed in code), submitted weights (metadata in `results/b_checkpoints.json`), submission files | logs of the three fold runs used to choose those epochs |
| B3, B4 | exact training command, submitted weights (raw, epochs 289 and 297), submission files | per-epoch logs |
| B5–B7 | recipe and leaderboard scores | command (fold, seed), weights, logs, submission files |

All code paths were checked on CPU (unit tests, parameter counts, data preparation, short training
and export runs on synthetic data). The full-length GPU runs were not repeated for this release.

## Reproduce

1. Put the dataset in `data/` (layout in [`data/README.md`](data/README.md)).
2. Follow [`a_resnet_scaling/README.md`](a_resnet_scaling/README.md) or
   [`b_width_depth/README.md`](b_width_depth/README.md). Each pipeline has its own requirements file.

Training needs a CUDA GPU; A0 and all evaluation also run on CPU.

## References

He et al., *Deep Residual Learning* (2016); Zagoruyko & Komodakis, *Wide Residual Networks* (2016);
Zhang et al., *mixup* (2018); Yun et al., *CutMix* (2019); DeVries & Taylor, *Cutout* (2017);
Cubuk et al., *RandAugment* (2020); Zhong et al., *Random Erasing* (2020); Szegedy et al., label
smoothing (2016); Huang et al., *Stochastic Depth* (2016); Hu et al., *Squeeze-and-Excitation* (2018);
Foret et al., *Sharpness-Aware Minimization* (2021). The A0 network follows
[kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py) at half width.
