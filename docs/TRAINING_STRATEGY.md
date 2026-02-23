# Training strategy for higher accuracy

The small CNN reaches about 47% validation accuracy. To improve further, try the following in order.

## 1. Use a pretrained model (recommended)

From the project root, run one or both:

```bash
# ResNet18 (often ~5–15% better than the small CNN)
python3 -m src.train --config configs/resnet18.yaml

# AlexNet (Hinton et al., 2012)
python3 -m src.train --config configs/alexnet.yaml
```

After training, the best weights are saved to `checkpoints/best.pt`. The web Identify page and `predict` / `evaluate` will use this checkpoint automatically (it includes `model_name`).

## 2. Data and augmentation

- Already enabled: random rotation ±30°, random grayscale 20%, horizontal flip, ColorJitter, RandomResizedCrop.
- Suggestions: at least 50–100 images per class; keep classes balanced; use clear, correctly labeled images.

## 3. Hyperparameters (optional)

In the relevant `configs/*.yaml` you can adjust:

- `train.lr`: use 1e-4 for pretrained models, 3e-4 for the small CNN.
- `train.early_stop_patience`: increase from 5 to 8 to train longer.
- `train.epochs`: with more data, try 80–100.

## 4. Quick workflow

1. Train with `configs/resnet18.yaml` or `configs/alexnet.yaml`.
2. Evaluate: `python3 -m src.evaluate --ckpt checkpoints/best.pt --split test`.
3. Open http://127.0.0.1:5001/predict in the browser for drag-and-drop identification.
