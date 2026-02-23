"""
Stratified train/val/test split (70/15/15). Writes split files so each image
appears in exactly one split. Uses a fixed seed for reproducibility.
"""
import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split


def _collect_samples(data_dir: Path) -> list[tuple[str, int]]:
    """Returns list of (relative_path, label_index). Relative path is class_name/filename."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class directories under {data_dir}")

    label_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
    samples: list[tuple[str, int]] = []
    for class_dir in class_dirs:
        rel_base = class_dir.name
        for f in class_dir.iterdir():
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                rel_path = f"{rel_base}/{f.name}"
                samples.append((rel_path, label_to_idx[class_dir.name]))
    return samples


def make_splits(
    data_dir: Path,
    out_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    samples = _collect_samples(data_dir)
    if not samples:
        raise ValueError("No images found in data_dir.")

    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    # First split: train vs (val+test)
    train_paths, rest_paths, train_labels, rest_labels = train_test_split(
        paths, labels, test_size=(1 - train_ratio), stratify=labels, random_state=seed
    )
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        rest_paths, rest_labels, test_size=(1 - val_size), stratify=rest_labels, random_state=seed
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_split(name: str, path_label_pairs: list[tuple[str, int]]) -> None:
        with open(out_dir / f"{name}.txt", "w") as f:
            for path, idx in path_label_pairs:
                f.write(f"{path},{idx}\n")

    write_split("train", list(zip(train_paths, train_labels)))
    write_split("val", list(zip(val_paths, val_labels)))
    write_split("test", list(zip(test_paths, test_labels)))
    print(f"Wrote train({len(train_paths)}), val({len(val_paths)}), test({len(test_paths)}) to {out_dir}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified train/val/test split.")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="ImageFolder root.")
    parser.add_argument("--out_dir", type=str, default="data/splits", help="Where to write train/val/test.txt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    make_splits(
        Path(args.data_dir),
        Path(args.out_dir),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
