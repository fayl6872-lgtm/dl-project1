"""Preactivation Wide ResNet for 32x32 images, initialized from scratch.

Architecture reference: Zagoruyko and Komodakis, arXiv:1605.07146.
Stem 16 channels, 3 stages, 4 blocks/stage, widths 56/112/224.
This is a fractional-width WRN-28, preserving an 8x8 final feature map.
"""
import torch
from torch import nn
from resnet import build_model as build_original


class WideBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1, dropout=0.):
        super().__init__()
        self.bn1=nn.BatchNorm2d(incoming)
        self.relu1=nn.ReLU(inplace=False)
        self.conv1=nn.Conv2d(incoming,outgoing,3,stride,1,bias=False)
        self.bn2=nn.BatchNorm2d(outgoing)
        self.relu2=nn.ReLU(inplace=False)
        self.dropout=nn.Dropout(dropout) if dropout else nn.Identity()
        self.conv2=nn.Conv2d(outgoing,outgoing,3,1,1,bias=False)
        self.project=incoming!=outgoing or stride!=1
        self.shortcut=nn.Conv2d(incoming,outgoing,1,stride,bias=False) if self.project else nn.Identity()

    def forward(self,x):
        activated=self.relu1(self.bn1(x))
        shortcut=self.shortcut(activated) if self.project else x
        branch=self.conv1(activated)
        branch=self.conv2(self.dropout(self.relu2(self.bn2(branch))))
        return shortcut+branch


class WideResNet(nn.Module):
    def __init__(self,channels=(56,112,224),blocks=4,dropout=0.):
        super().__init__()
        self.stem=nn.Conv2d(3,16,3,1,1,bias=False)
        incoming=16;stages=[]
        for stage,width in enumerate(channels):
            group=[]
            for block in range(blocks):
                group.append(WideBlock(incoming,width,2 if stage>0 and block==0 else 1,dropout))
                incoming=width
            stages.append(nn.Sequential(*group))
        self.stages=nn.Sequential(*stages)
        self.bn=nn.BatchNorm2d(incoming)
        self.relu=nn.ReLU(inplace=False)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.classifier=nn.Linear(incoming,10)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):nn.init.ones_(m.weight);nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear):nn.init.normal_(m.weight,0,.01);nn.init.zeros_(m.bias)

    def forward(self,x):
        x=self.relu(self.bn(self.stages(self.stem(x))))
        return self.classifier(self.pool(x).flatten(1))


def build_model(config):
    if config.get('architecture')!='wide_resnet':return build_original(config)
    model=WideResNet(tuple(config['channels']),config['blocks'],config['dropout'])
    count=sum(p.numel() for p in model.parameters())
    assert count<=min(config['parameter_limit'],5_000_000),f'{count:,} exceeds parameter cap'
    return model
