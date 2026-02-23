"""
Pretrained ResNet18: replace classifier head for num_classes.
Uses ImageNet pretrained weights; same normalization as train/eval.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class MarineResNet18(nn.Module):
    """ResNet18 with pretrained backbone and replaceable classifier for num_classes."""

    def __init__(self, num_classes: int, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
