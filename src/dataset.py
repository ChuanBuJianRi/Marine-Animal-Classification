"""Dataset that reads from split files (path,label_index) and loads from data_dir."""
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset
from PIL import Image
import torch


def _parse_split_file(path: str | Path) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            rel_path, idx_str = parts
            out.append((rel_path.strip(), int(idx_str)))
    return out


class MarineDataset(Dataset):
    """Loads images listed in a split file; labels are integers."""

    def __init__(
        self,
        data_dir: str | Path,
        split_file: str | Path,
        transform: Optional[object] = None,
    ):
        self.data_dir = Path(data_dir)
        self.samples = _parse_split_file(split_file)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rel_path, label = self.samples[idx]
        path = self.data_dir / rel_path
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_class_names(data_dir: str | Path) -> list[str]:
    """Ordered list of class names (folder names) under data_dir."""
    data_dir = Path(data_dir)
    return sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
