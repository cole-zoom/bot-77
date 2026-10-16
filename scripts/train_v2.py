"""Train the v2 unit detector (decisions.md, 2026-10-09) on data/external/train_v2 (real_v2 + synth_v1, 219 classes).

Same script locally and on a rented GPU; the device is picked automatically (cuda > mps > cpu).

  smoke test (Mac, ~10 min):  uv run python scripts/train_v2.py --smoke
  cloud dry run (~$1):        python scripts/train_v2.py --data <path>/data.yaml --epochs 1 --batch 32
  full run:                   python scripts/train_v2.py --data <path>/data.yaml --epochs 100 --batch 32 --model yolo11m.pt

Augmentation follows v1 / KataCR [Wu et al. 2025]: horizontal flips (the arena is mirror-symmetric) and mild HSV,
no mosaic (it cuts units in half at the seams and breaks the arena layout).
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/external/train_v2/data.yaml")
ap.add_argument("--model", default="yolo11s.pt")
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--batch", type=int, default=0, help="0 = auto (4 on Mac, 32 on a GPU)")
ap.add_argument("--imgsz", type=int, default=896)
ap.add_argument("--name", default="v2")
ap.add_argument("--project", default="data/models/bot77_detector")
ap.add_argument("--patience", type=int, default=20, help="0 = no early stopping (run 2 picks checkpoints by the play benchmark)")
ap.add_argument("--save-period", type=int, default=-1, help="also save weights every N epochs")
ap.add_argument("--smoke", action="store_true", help="1 epoch on 3%% of the data, to prove the pipeline end to end")
a = ap.parse_args()

device = "0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
batch = a.batch or (32 if device == "0" else 4)
weights = Path("data/models") / a.model
model = YOLO(str(weights.resolve()) if weights.exists() else a.model)
model.train(
    data=str(Path(a.data).resolve()), epochs=1 if a.smoke else a.epochs, imgsz=a.imgsz, batch=batch, device=device,
    workers=2 if device != "0" else 8, fraction=0.03 if a.smoke else 1.0,
    project=str(Path(a.project).resolve()), name=a.name + ("_smoke" if a.smoke else ""), exist_ok=True,
    fliplr=0.5, hsv_h=0.015, hsv_s=0.5, hsv_v=0.3, degrees=0, translate=0.05, scale=0.2, mosaic=0.0,
    patience=0 if a.smoke else (a.patience or 10_000), save_period=a.save_period, plots=True, val=True, verbose=False, seed=0, deterministic=False,
)
print("device", device, "batch", batch, "->", Path(a.project) / (a.name + ("_smoke" if a.smoke else "")))
