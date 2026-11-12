"""A CIFAR-sized ResNet with configurable width, initialized from scratch.

Architecture references (no pretrained weights used):
He et al., Deep Residual Learning for Image Recognition, arXiv:1512.03385.
Reference implementation: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
Stage widths are half of the reference (32-256) to stay under the 5M parameter limit.
"""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.branch(x) + self.shortcut(x))


class CifarResNet(nn.Module):
    def __init__(self, base_width=32, blocks_per_stage=(2, 2, 2, 2), num_classes=10):
        super().__init__()
        assert len(blocks_per_stage) == 4 and all(n > 0 for n in blocks_per_stage)
        self.stem = nn.Sequential(nn.Conv2d(3, base_width, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        current = base_width
        stages = []
        for stage, count in enumerate(blocks_per_stage):
            width = base_width * 2**stage
            blocks = []
            for block in range(count):
                stride = 2 if stage > 0 and block == 0 else 1
                blocks.append(ResidualBlock(current, width, stride))
                current = width
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.classifier(self.pool(self.stages(self.stem(x))).flatten(1))


def build_model(config):
    model = CifarResNet(config['base_width'], config['blocks_per_stage'], config['num_classes'])
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count <= config['parameter_limit'], f'{count:,} parameters exceed the project limit'
    return model
