"""
Predict class and confidence for a single image (or top-k). Uses the same
normalization as evaluation.
"""
import argparse
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from src.models import get_model


def get_infer_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint.")
    parser.add_argument("--image", type=str, required=True, help="Path to image.")
    parser.add_argument("--topk", type=int, default=5, help="Number of top classes to print.")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    num_classes = ckpt["num_classes"]
    class_names = ckpt.get("class_names", [str(i) for i in range(num_classes)])
    model_name = ckpt.get("model_name", "cnn")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(img_path)
    img = Image.open(img_path).convert("RGB")
    transform = get_infer_transform(args.image_size)
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    topk = min(args.topk, num_classes)
    values, indices = torch.topk(probs, topk)

    print(f"Image: {img_path}")
    for i in range(topk):
        idx = indices[i].item()
        conf = values[i].item()
        name = class_names[idx] if idx < len(class_names) else str(idx)
        print(f"  {i + 1}. {name}  {conf:.4f}")
    pred_idx = indices[0].item()
    pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
    print(f"Prediction: {pred_name} (confidence {values[0].item():.4f})")


if __name__ == "__main__":
    main()
