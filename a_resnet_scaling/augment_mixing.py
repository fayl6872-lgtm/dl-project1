"""Training-only Cutout, Mixup, and CutMix; never reads validation/test labels.

References: arXiv:1708.04552, arXiv:1710.09412, arXiv:1905.04899.
"""
import math
import torch
from torch.nn import functional as F


def cutout(x, size, generator):
    if size == 0:
        return x
    n, _, h, w = x.shape
    cy = torch.randint(h, (n, 1, 1, 1), device=x.device, generator=generator)
    cx = torch.randint(w, (n, 1, 1, 1), device=x.device, generator=generator)
    yy = torch.arange(h, device=x.device)[None, None, :, None]
    xx = torch.arange(w, device=x.device)[None, None, None, :]
    mask = (yy >= cy-size//2) & (yy < cy+(size+1)//2) & (xx >= cx-size//2) & (xx < cx+(size+1)//2)
    # Normalized zero is the training-set channel mean.
    return x.masked_fill(mask, 0.)


def mix_batch(x, labels, config, rng, generator, enabled=True):
    """Return images, two target sets and actual contribution of first target."""
    if not enabled or rng.random() >= config.get('mix_probability', 0.):
        return x, labels, labels, 1.
    order = torch.randperm(len(x), device=x.device, generator=generator)
    if rng.random() < config.get('cutmix_fraction', .5):
        lam = float(rng.beta(config['cutmix_alpha'], config['cutmix_alpha']))
        h, w = x.shape[-2:]
        ratio = math.sqrt(1-lam)
        ch, cw = int(h*ratio), int(w*ratio)
        cy, cx = int(rng.integers(h)), int(rng.integers(w))
        y0, y1 = max(0, cy-ch//2), min(h, cy+(ch+1)//2)
        x0, x1 = max(0, cx-cw//2), min(w, cx+(cw+1)//2)
        mixed = x.clone()
        mixed[:, :, y0:y1, x0:x1] = x[order, :, y0:y1, x0:x1]
        # Correct for clipped patches at the image boundary.
        lam = 1-(y1-y0)*(x1-x0)/(h*w)
    else:
        lam = float(rng.beta(config['mixup_alpha'], config['mixup_alpha']))
        mixed = x*lam + x[order]*(1-lam)
    return mixed, labels, labels[order], lam


def mixed_loss(logits, first, second, lam):
    return lam*F.cross_entropy(logits, first)+(1-lam)*F.cross_entropy(logits, second)
