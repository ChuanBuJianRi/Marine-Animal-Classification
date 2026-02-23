"""
CNN: 4 Conv blocks (3 with MaxPool) + GlobalAvgPool + Dropout + Linear.
Block1: Conv(3->32) BN ReLU MaxPool
Block2: Conv(32->64) BN ReLU MaxPool
Block3: Conv(64->128) BN ReLU MaxPool
Block4: Conv(128->256) BN ReLU
Head: GAP, Dropout, Linear(256 -> num_classes)
"""
import torch
import torch.nn as nn


def _conv_block(in_c: int, out_c: int, pool: bool = True) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class MarineCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.4, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(in_channels, 32, pool=True),   # Block1
            _conv_block(32, 64, pool=True),            # Block2
            _conv_block(64, 128, pool=True),           # Block3
            _conv_block(128, 256, pool=False),         # Block4
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.drop(x)
        x = self.fc(x)
        return x
