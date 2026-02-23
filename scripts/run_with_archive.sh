#!/usr/bin/env bash
# Use data from Downloads/archive (2): prepare -> splits -> train -> evaluate.
# Run from project root. Requires: pip install -r requirements.txt

set -e
ARCHIVE="/Users/yiyanggao/Downloads/archive (2)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Prepare data from $ARCHIVE ==="
python -m src.prepare_data --raw_dir "$ARCHIVE" --out_dir data/processed

echo "=== Make splits (70/15/15) ==="
python -m src.make_splits --data_dir data/processed --out_dir data/splits --seed 42

echo "=== Train ==="
python -m src.train --config configs/cnn.yaml

echo "=== Evaluate on test set ==="
python -m src.evaluate --ckpt checkpoints/best.pt --split test

echo "Done. Check reports/results.json and reports/figures/confusion_matrix.png"
