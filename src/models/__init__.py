from .cnn import MarineCNN
from .resnet import MarineResNet18
from .alexnet import MarineAlexNet


def get_model(name: str, num_classes: int, **kwargs):
    """Build model by name. Used by train and predict."""
    num_classes = int(num_classes)
    name = (name or "cnn").lower().strip()
    if name == "cnn":
        return MarineCNN(num_classes=num_classes, **{k: v for k, v in kwargs.items() if k in ("dropout", "in_channels")})
    if name == "resnet18":
        return MarineResNet18(num_classes=num_classes, **{k: v for k, v in kwargs.items() if k in ("dropout", "pretrained")})
    if name == "alexnet":
        return MarineAlexNet(num_classes=num_classes, **{k: v for k, v in kwargs.items() if k in ("dropout", "pretrained")})
    raise ValueError(f"Unknown model: {name}. Use 'cnn', 'resnet18', or 'alexnet'.")


__all__ = ["MarineCNN", "MarineResNet18", "MarineAlexNet", "get_model"]
