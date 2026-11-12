"""
models/resnet.py
================
Four CIFAR-10 optimised ResNet variants (32x32 input, trained from scratch).

All models:
  - Use a 3x3 conv stem (no 7x7, no max-pool) so the 32x32 spatial size is
    preserved through the first stage.
  - Stay ≤ 5,000,000 trainable parameters (asserted at construction time).
  - Expose a `count_params()` method and a `param_count` property.

Architectures
─────────────
  BaselineResNet  – post-activation (BN after ReLU), ~2.78 M params
  PreActResNet    – pre-activation (BN→ReLU→Conv, ResNet-v2), ~2.78 M params
  SEResNet        – PreActResNet + Squeeze-and-Excitation blocks, ~2.85 M params
  WideResNet      – wider PreAct ResNet (80-160-320, 2 blocks/stage), ~4.3 M params
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BaselineResNet",
    "PreActResNet",
    "SEResNet",
    "WideResNet",
    "DropPath",
]

# ---------------------------------------------------------------------------
# Utility: Stochastic Depth (DropPath)
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Drop entire residual branches stochastically during training.

    Args:
        drop_prob: Probability of dropping a sample's residual path.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # shape: (batch, 1, 1, 1) for 4-D feature maps
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


# ---------------------------------------------------------------------------
# Helper: 3x3 conv with bias=False
# ---------------------------------------------------------------------------

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3,
        stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation module
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Channel Squeeze-and-Excitation (Hu et al., CVPR 2018).

    Args:
        channels: Number of input channels.
        reduction: Reduction ratio for the bottleneck FC layer (default 16,
                   falls back to 8 for channels < 128 to avoid trivial sizes).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # Avoid reduction to < 4 channels
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


# ===========================================================================
# 1.  BaselineResNet – post-activation (classic ResNet style)
# ===========================================================================

class BasicBlockPost(nn.Module):
    """Standard post-activation residual block (Conv → BN → ReLU × 2).

    The skip connection uses a 1x1 conv + BN when spatial size or channel
    count changes (stride > 1 or in_planes ≠ planes).
    """

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        drop_path_prob: float = 0.0,
    ):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.drop_path = DropPath(drop_path_prob)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, planes, stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop_path(out) + identity
        return self.relu(out)


class BaselineResNet(nn.Module):
    """Post-activation ResNet optimised for CIFAR-10 (32x32 images).

    Architecture:
        Stem  : Conv(3, 64, 3, s=1, p=1) → BN → ReLU      [32x32]
        Stage1: 2 × BasicBlockPost(64,  64,  stride=1)     [32x32]
        Stage2: 2 × BasicBlockPost(64,  128, stride=2 1st) [16x16]
        Stage3: 2 × BasicBlockPost(128, 256, stride=2 1st) [8x8]
        Head  : GlobalAvgPool → Linear(256, num_classes)

    ~2.78 M parameters.
    """

    def __init__(
        self,
        num_classes: int = 10,
        drop_path_prob: float = 0.0,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            conv3x3(3, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64,  64,  2, stride=1, drop_path_prob=drop_path_prob)
        self.layer2 = self._make_layer(64,  128, 2, stride=2, drop_path_prob=drop_path_prob)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, drop_path_prob=drop_path_prob)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(256, num_classes)

        self._init_weights()
        assert self.count_params() <= 5_000_000, (
            f"BaselineResNet has {self.count_params():,} params – exceeds 5 M limit!"
        )

    @staticmethod
    def _make_layer(
        in_planes: int,
        planes: int,
        num_blocks: int,
        stride: int,
        drop_path_prob: float,
    ) -> nn.Sequential:
        layers = [BasicBlockPost(in_planes, planes, stride, drop_path_prob)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlockPost(planes, planes, 1, drop_path_prob))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def param_count(self) -> int:
        return self.count_params()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ===========================================================================
# 2.  PreActResNet – pre-activation (ResNet-v2 style)
# ===========================================================================

class PreActBlock(nn.Module):
    """Pre-activation residual block (BN → ReLU → Conv × 2).

    The skip connection is a plain 1×1 conv (no BN) when shapes differ.
    """

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        drop_path_prob: float = 0.0,
    ):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.relu  = nn.ReLU(inplace=True)
        self.drop_path = DropPath(drop_path_prob)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = conv1x1(in_planes, planes, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.drop_path(out) + identity
        return out


class PreActResNet(nn.Module):
    """Pre-activation ResNet (ResNet-v2) optimised for CIFAR-10.

    Architecture:
        Stem  : Conv(3, 64, 3, s=1, p=1)                   [32x32]
        Stage1: 2 × PreActBlock(64,  64,  stride=1)        [32x32]
        Stage2: 2 × PreActBlock(64,  128, stride=2 1st)    [16x16]
        Stage3: 2 × PreActBlock(128, 256, stride=2 1st)    [8x8]
        Head  : BN → ReLU → GlobalAvgPool → Linear(256, C)

    ~2.78 M parameters.
    """

    def __init__(
        self,
        num_classes: int = 10,
        widths: tuple = (64, 128, 256),
        blocks_per_stage: tuple = (2, 2, 2),
        drop_path_prob: float = 0.0,
    ):
        super().__init__()
        w1, w2, w3 = widths
        b1, b2, b3 = blocks_per_stage

        self.stem = conv3x3(3, w1)
        self.layer1 = self._make_layer(w1, w1, b1, stride=1, drop_path_prob=drop_path_prob)
        self.layer2 = self._make_layer(w1, w2, b2, stride=2, drop_path_prob=drop_path_prob)
        self.layer3 = self._make_layer(w2, w3, b3, stride=2, drop_path_prob=drop_path_prob)
        self.bn_out = nn.BatchNorm2d(w3)
        self.relu   = nn.ReLU(inplace=True)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(w3, num_classes)

        self._init_weights()
        assert self.count_params() <= 5_000_000, (
            f"PreActResNet has {self.count_params():,} params – exceeds 5 M limit!"
        )

    @staticmethod
    def _make_layer(
        in_planes: int,
        planes: int,
        num_blocks: int,
        stride: int,
        drop_path_prob: float,
    ) -> nn.Sequential:
        layers = [PreActBlock(in_planes, planes, stride, drop_path_prob)]
        for _ in range(1, num_blocks):
            layers.append(PreActBlock(planes, planes, 1, drop_path_prob))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def param_count(self) -> int:
        return self.count_params()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn_out(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ===========================================================================
# 3.  SEResNet – PreActResNet + Squeeze-and-Excitation
# ===========================================================================

class PreActSEBlock(nn.Module):
    """Pre-activation residual block with Squeeze-and-Excitation."""

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        reduction: int = 16,
        drop_path_prob: float = 0.0,
    ):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.relu  = nn.ReLU(inplace=True)
        self.se    = SEBlock(planes, reduction=reduction)
        self.drop_path = DropPath(drop_path_prob)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = conv1x1(in_planes, planes, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.se(out)
        out = self.drop_path(out) + identity
        return out


class SEResNet(nn.Module):
    """Pre-activation ResNet with Squeeze-and-Excitation (CIFAR-10).

    Architecture identical to PreActResNet but every residual block includes
    an SE channel attention module after the second convolution.

    ~2.85 M parameters.
    """

    def __init__(
        self,
        num_classes: int = 10,
        widths: tuple = (64, 128, 256),
        blocks_per_stage: tuple = (2, 2, 2),
        reduction: int = 16,
        drop_path_prob: float = 0.0,
    ):
        super().__init__()
        w1, w2, w3 = widths
        b1, b2, b3 = blocks_per_stage

        self.stem = conv3x3(3, w1)
        self.layer1 = self._make_layer(w1, w1, b1, stride=1, reduction=reduction, drop_path_prob=drop_path_prob)
        self.layer2 = self._make_layer(w1, w2, b2, stride=2, reduction=reduction, drop_path_prob=drop_path_prob)
        self.layer3 = self._make_layer(w2, w3, b3, stride=2, reduction=reduction, drop_path_prob=drop_path_prob)
        self.bn_out = nn.BatchNorm2d(w3)
        self.relu   = nn.ReLU(inplace=True)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(w3, num_classes)

        self._init_weights()
        assert self.count_params() <= 5_000_000, (
            f"SEResNet has {self.count_params():,} params – exceeds 5 M limit!"
        )

    @staticmethod
    def _make_layer(
        in_planes: int,
        planes: int,
        num_blocks: int,
        stride: int,
        reduction: int,
        drop_path_prob: float,
    ) -> nn.Sequential:
        layers = [PreActSEBlock(in_planes, planes, stride, reduction, drop_path_prob)]
        for _ in range(1, num_blocks):
            layers.append(PreActSEBlock(planes, planes, 1, reduction, drop_path_prob))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def param_count(self) -> int:
        return self.count_params()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn_out(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ===========================================================================
# 4.  WideResNet – wider PreAct ResNet
# ===========================================================================

class WideResNet(nn.Module):
    """Wide pre-activation ResNet for CIFAR-10.

    Default widths (80, 160, 320) with 2 blocks per stage gives ~4.3 M params,
    well within the 5 M constraint.

    Architecture:
        Stem  : Conv(3, 80, 3, s=1, p=1)                   [32x32]
        Stage1: 2 × PreActBlock(80,  80,  stride=1)        [32x32]
        Stage2: 2 × PreActBlock(80,  160, stride=2 1st)    [16x16]
        Stage3: 2 × PreActBlock(160, 320, stride=2 1st)    [8x8]
        Head  : BN → ReLU → GlobalAvgPool → Linear(320, C)

    ~4.3 M parameters.
    """

    def __init__(
        self,
        num_classes: int = 10,
        widths: tuple = (80, 160, 320),
        blocks_per_stage: tuple = (2, 2, 2),
        drop_path_prob: float = 0.0,
    ):
        super().__init__()
        w1, w2, w3 = widths
        b1, b2, b3 = blocks_per_stage

        self.stem = conv3x3(3, w1)
        self.layer1 = self._make_layer(w1, w1, b1, stride=1, drop_path_prob=drop_path_prob)
        self.layer2 = self._make_layer(w1, w2, b2, stride=2, drop_path_prob=drop_path_prob)
        self.layer3 = self._make_layer(w2, w3, b3, stride=2, drop_path_prob=drop_path_prob)
        self.bn_out = nn.BatchNorm2d(w3)
        self.relu   = nn.ReLU(inplace=True)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(w3, num_classes)

        self._init_weights()
        assert self.count_params() <= 5_000_000, (
            f"WideResNet has {self.count_params():,} params – exceeds 5 M limit!"
        )

    @staticmethod
    def _make_layer(
        in_planes: int,
        planes: int,
        num_blocks: int,
        stride: int,
        drop_path_prob: float,
    ) -> nn.Sequential:
        layers = [PreActBlock(in_planes, planes, stride, drop_path_prob)]
        for _ in range(1, num_blocks):
            layers.append(PreActBlock(planes, planes, 1, drop_path_prob))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def param_count(self) -> int:
        return self.count_params()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn_out(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    models = {
        "BaselineResNet": BaselineResNet(),
        "PreActResNet":   PreActResNet(),
        "SEResNet":       SEResNet(),
        "WideResNet":     WideResNet(),
    }
    x = torch.randn(2, 3, 32, 32)
    print(f"{'Model':<20} {'Params':>12}  Output shape")
    print("-" * 50)
    for name, model in models.items():
        out = model(x)
        print(f"{name:<20} {model.count_params():>12,}  {tuple(out.shape)}")
