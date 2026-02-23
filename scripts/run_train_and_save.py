"""
One-shot: prepare data from archive -> make splits -> train -> save best model.
Run from project root:  python scripts/run_train_and_save.py
Requires: pip install -r requirements.txt
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = "/Users/yiyanggao/Downloads/archive (2)"


def run(cmd: list[str], desc: str) -> None:
    print(f"\n=== {desc} ===\n")
    ret = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if ret.returncode != 0:
        sys.exit(ret.returncode)


def main() -> None:
    run(
        [sys.executable, "-m", "src.prepare_data", "--raw_dir", ARCHIVE, "--out_dir", "data/processed"],
        "Prepare data (clean & copy to data/processed)",
    )
    run(
        [sys.executable, "-m", "src.make_splits", "--data_dir", "data/processed", "--out_dir", "data/splits", "--seed", "42"],
        "Make train/val/test splits",
    )
    run(
        [sys.executable, "-m", "src.train", "--config", "configs/cnn.yaml"],
        "Train model (best checkpoint saved by val loss)",
    )
    ckpt = PROJECT_ROOT / "checkpoints" / "best.pt"
    print(f"\nModel saved to: {ckpt}")
    print("Evaluate:  python -m src.evaluate --ckpt checkpoints/best.pt --split test")
    print("Predict:   python -m src.predict --ckpt checkpoints/best.pt --image <path> --topk 5")


if __name__ == "__main__":
    main()
