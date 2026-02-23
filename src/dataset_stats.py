"""Compute dataset stats (class counts per split) for the dashboard."""
from pathlib import Path


def get_dataset_stats(data_dir: str | Path, splits_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    splits_dir = Path(splits_dir)
    if not data_dir.exists() or not splits_dir.exists():
        return {"error": "data_dir or splits_dir not found", "classes": [], "splits": {}}

    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    n_classes = len(class_names)

    splits = {}
    for split_name in ("train", "val", "test"):
        path = splits_dir / f"{split_name}.txt"
        if not path.exists():
            splits[split_name] = {"total": 0, "per_class": [0] * n_classes}
            continue
        per_class = [0] * n_classes
        total = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",", 1)
                if len(parts) != 2:
                    continue
                total += 1
                idx = int(parts[1])
                if 0 <= idx < n_classes:
                    per_class[idx] += 1
        splits[split_name] = {"total": total, "per_class": per_class}

    return {
        "class_names": class_names,
        "num_classes": n_classes,
        "splits": splits,
    }
