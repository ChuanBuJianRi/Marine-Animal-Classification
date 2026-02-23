"""
Evaluate a checkpoint on a split (val or test). Writes metrics to reports/results.json
and confusion matrix to reports/figures/confusion_matrix.png.
"""
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import MarineDataset, get_class_names
from src.models import get_model
from src.utils import top1_accuracy, macro_f1, per_class_metrics, plot_confusion_matrix


def get_eval_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (e.g. checkpoints/best.pt).")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--splits_dir", type=str, default="data/splits")
    parser.add_argument("--reports_dir", type=str, default="reports")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    num_classes = ckpt["num_classes"]
    model_name = ckpt.get("model_name", "cnn")
    class_names = ckpt.get("class_names")
    if class_names is None:
        data_dir = Path(args.data_dir)
        class_names = get_class_names(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    data_dir = Path(args.data_dir)
    splits_dir = Path(args.splits_dir)
    split_file = splits_dir / f"{args.split}.txt"
    ds = MarineDataset(data_dir, split_file, transform=get_eval_transform(args.image_size))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(pred)
            all_labels.append(y.numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    acc = top1_accuracy(y_true, y_pred)
    f1 = macro_f1(y_true, y_pred)
    per_class = per_class_metrics(y_true, y_pred)

    reports_dir = Path(args.reports_dir)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "split": args.split,
        "accuracy": acc,
        "macro_f1": f1,
        "per_class": per_class,
    }
    out_json = reports_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_json}: accuracy={acc:.4f}, macro_f1={f1:.4f}")

    plot_confusion_matrix(
        y_true, y_pred,
        class_names=class_names,
        save_path=figures_dir / "confusion_matrix.png",
    )
    print(f"Confusion matrix saved to {figures_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
