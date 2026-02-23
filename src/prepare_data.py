"""
Prepare raw data into ImageFolder layout under out_dir.
Cleans: skip corrupted images, drop images with both sides < min_size.
Optional: deduplicate by file hash.
"""
import argparse
import hashlib
import shutil
from pathlib import Path

from PIL import Image


def _min_side(path: Path, min_size: int = 80) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
            return w >= min_size and h >= min_size
    except Exception:
        return False


def _file_hash(path: Path, block_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def _sanitize_class_name(name: str) -> str:
    return name.strip().replace(" ", "_").replace("/", "_") or "unknown"


def prepare_from_folders(
    raw_dir: Path,
    out_dir: Path,
    min_size: int = 80,
    dedup: bool = False,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> tuple[int, int]:
    """
    raw_dir is expected to contain one subdir per class: raw_dir/ClassName/*.jpg
    Copies valid images to out_dir/ClassName/ and returns (kept, removed) counts.
    """
    out_dir = Path(out_dir)
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data dir not found: {raw_dir}")

    seen_hashes: set[str] = set()
    kept, removed = 0, 0

    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = _sanitize_class_name(class_dir.name)
        dest_class = out_dir / class_name
        dest_class.mkdir(parents=True, exist_ok=True)

        for path in class_dir.iterdir():
            if path.suffix.lower() not in extensions:
                continue
            if not _min_side(path, min_size):
                removed += 1
                continue
            if dedup:
                h = _file_hash(path)
                if h in seen_hashes:
                    removed += 1
                    continue
                seen_hashes.add(h)
            dest = dest_class / path.name
            if dest != path and not dest.exists():
                shutil.copy2(path, dest)
            kept += 1

    return kept, removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare raw images into ImageFolder and clean.")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Raw data root (folder per class).")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Output ImageFolder root.")
    parser.add_argument("--min_size", type=int, default=80, help="Drop images with both sides smaller than this.")
    parser.add_argument("--dedup", action="store_true", help="Deduplicate by file hash.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kept, removed = prepare_from_folders(
        raw_dir, out_dir, min_size=args.min_size, dedup=args.dedup
    )
    print(f"Kept {kept} images, skipped/removed {removed}.")


if __name__ == "__main__":
    main()
