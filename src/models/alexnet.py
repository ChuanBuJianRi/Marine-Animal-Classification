"""
AlexNet (Krizhevsky, Sutskever, Hinton, 2012). Pretrained on ImageNet.
Replace classifier head for num_classes. Same normalization as train/eval.
"""
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights


def MarineAlexNet(num_classes: int, dropout: float = 0.5, pretrained: bool = True):
    """AlexNet with pretrained backbone and replaceable classifier for num_classes."""
    weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = alexnet(weights=weights)
    # classifier[6] is Linear(4096, 1000)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model
