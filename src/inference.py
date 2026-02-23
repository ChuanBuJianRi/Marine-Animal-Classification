"""
Load best checkpoint and run prediction on an image (PIL or path).
Used by the web dashboard /api/predict.
"""
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from src.models import get_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "best.pt"
IMAGE_SIZE = 224

_model_cache: dict[str, Any] = {}  # "model", "class_names", "device"


def _get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(ckpt_path: Path | None = None) -> tuple[Any, list[str], torch.device]:
    """Load or return cached model. Returns (model, class_names, device)."""
    ckpt_path = ckpt_path or DEFAULT_CKPT
    cache_key = str(ckpt_path.resolve())
    if cache_key in _model_cache:
        c = _model_cache[cache_key]
        return c["model"], c["class_names"], c["device"]

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Use plain Python types to avoid tensor/numpy triggering .item() etc.
    num_classes = int(ckpt["num_classes"])
    raw_names = ckpt.get("class_names", [str(i) for i in range(num_classes)])
    class_names = [str(n) for n in raw_names]
    model_name = str(ckpt.get("model_name", "cnn"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _model_cache[cache_key] = {"model": model, "class_names": class_names, "device": device}
    return model, class_names, device


def predict_image(image: Image.Image, top_k: int = 5, ckpt_path: Path | None = None) -> dict:
    """
    Run prediction on a PIL Image. Returns dict with:
      prediction, confidence, top_k: [{ class_name, confidence }, ...]
    """
    model, class_names, device = load_model(ckpt_path)
    transform = _get_transform()
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    k = min(top_k, len(class_names))
    values, indices = torch.topk(probs, k)

    # Ensure list (tolist() can return a scalar in some cases)
    def to_list(t):
        out = t.tolist()
        return out if isinstance(out, list) else [out]

    values_list = to_list(values)
    indices_list = to_list(indices)
    top_list = []
    for v, i in zip(values_list, indices_list):
        v_f = float(v)
        i_int = int(i)
        name = class_names[i_int] if i_int < len(class_names) else str(i_int)
        top_list.append({
            "class_name": str(name),
            "confidence": float(round(v_f, 4)),
        })
    pred_name = str(top_list[0]["class_name"])
    pred_conf = float(top_list[0]["confidence"])

    # Return only Python built-in types so jsonify does not call .item() on numpy/tensor
    return {
        "prediction": pred_name,
        "confidence": pred_conf,
        "top_k": [{"class_name": str(x["class_name"]), "confidence": float(x["confidence"])} for x in top_list],
    }
