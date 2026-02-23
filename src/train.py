"""
Train the CNN: load config, build dataloaders, train with early stopping and TensorBoard.
Best checkpoint is saved based on validation loss.
Writes metrics to reports/training_log.json for the web dashboard.
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from src.config import load_config, get_data_config, get_train_config, get_model_config, get_paths_config
from src.dataset import MarineDataset, get_class_names
from src.models import get_model
from src.utils import set_seed


def get_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if optimizer is not None:
            optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cnn.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = get_data_config(config)
    train_cfg = get_train_config(config)
    model_cfg = get_model_config(config)
    paths_cfg = get_paths_config(config)

    data_dir = Path(data_cfg["data_dir"])
    splits_dir = Path(data_cfg["splits_dir"])
    image_size = data_cfg.get("image_size", 224)
    train_bs = data_cfg.get("train_batch_size", 32)
    val_bs = data_cfg.get("val_batch_size", 64)
    num_workers = data_cfg.get("num_workers", 4)

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    num_classes = len(get_class_names(data_dir))
    if model_cfg.get("num_classes") is not None:
        assert model_cfg["num_classes"] == num_classes
    model_cfg["num_classes"] = num_classes
    dropout = model_cfg.get("dropout", 0.4)

    train_ds = MarineDataset(
        data_dir,
        splits_dir / "train.txt",
        transform=get_transforms(image_size, train=True),
    )
    val_ds = MarineDataset(
        data_dir,
        splits_dir / "val.txt",
        transform=get_transforms(image_size, train=False),
    )
    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False, num_workers=num_workers, pin_memory=False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    resume_path = args.resume or (str(Path(paths_cfg.get("checkpoint_dir", "checkpoints")) / "best.pt") if (Path(paths_cfg.get("checkpoint_dir", "checkpoints")) / "best.pt").exists() else None)
    if resume_path and Path(resume_path).exists():
        _ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model_name = _ckpt.get("model_name", model_cfg.get("name", "cnn"))
    else:
        model_name = model_cfg.get("name", "cnn")

    model = get_model(
        model_name,
        num_classes,
        dropout=dropout,
        pretrained=model_cfg.get("pretrained", True),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    epochs = train_cfg.get("epochs", 40)
    scheduler_name = train_cfg.get("scheduler", "cosine")
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    ckpt_dir = Path(paths_cfg.get("checkpoint_dir", "checkpoints"))
    runs_dir = Path(paths_cfg.get("runs_dir", "runs"))
    reports_dir = Path(paths_cfg.get("reports_dir", "reports"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(runs_dir)

    training_log_path = reports_dir / "training_log.json"
    training_pid_path = reports_dir / "training_pid.txt"
    try:
        training_pid_path.write_text(str(os.getpid()))
    except Exception:
        pass

    try:
        # Load existing log so history is preserved across resume
        if training_log_path.exists():
            with open(training_log_path) as f:
                training_history: list[dict] = json.load(f)
        else:
            training_history = []

        start_epoch = 0
        best_val_loss = float("inf")
        no_improve = 0

        if resume_path and Path(resume_path).exists():
            print(f"Resuming from {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("val_loss", float("inf"))
            # Restore scheduler state by fast-forwarding (suppress step-order warning)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(start_epoch):
                    if scheduler_name == "cosine":
                        scheduler.step()
            print(f"Resumed at epoch {start_epoch}, best val_loss so far: {best_val_loss:.4f}")

        patience = train_cfg.get("early_stop_patience", 5)

        for epoch in range(start_epoch, epochs):
            train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, device, optimizer=None)

            if scheduler_name == "cosine":
                scheduler.step()
            else:
                scheduler.step(val_loss)
            lr = optimizer.param_groups[0]["lr"]

            writer.add_scalar("loss/train", train_loss, epoch + 1)
            writer.add_scalar("loss/val", val_loss, epoch + 1)
            writer.add_scalar("acc/train", train_acc, epoch + 1)
            writer.add_scalar("acc/val", val_acc, epoch + 1)
            writer.add_scalar("lr", lr, epoch + 1)

            training_history.append({
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_acc": round(val_acc, 4),
                "lr": round(lr, 6),
            })
            with open(training_log_path, "w") as f:
                json.dump(training_history, f, indent=2)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "num_classes": num_classes,
                    "class_names": get_class_names(data_dir),
                    "model_name": model_name,
                }, ckpt_dir / "best.pt")
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1} (val loss did not improve for {patience} checks).")
                break
            print(f"Epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} lr={lr:.2e}")

        writer.close()
        print("Training done. Best checkpoint saved to", ckpt_dir / "best.pt")
    finally:
        try:
            training_pid_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
