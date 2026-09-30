"""Train bot77's unit detector v1 on the synthetic dataset (scripts/katacr/generate_synth.py).

Standard Ultralytics YOLO, one model for all classes, trained on the Mac's GPU (MPS). The team is
not a model output: it comes from badge colour and the POV play log (decisions.md, 2026-09-28).
Method follows KataCR's generative-dataset training [Wu et al. 2025, §3, §5.2].
"""
import sys
from pathlib import Path

from ultralytics import YOLO

data = Path(sys.argv[1] if len(sys.argv) > 1 else "data/external/synth_v1/data.yaml").resolve()
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
model = YOLO(str(Path("data/models/yolo11s.pt").resolve()) if Path("data/models/yolo11s.pt").exists() else "yolo11s.pt")
model.train(
    data=str(data), epochs=epochs, imgsz=896, rect=False, batch=4, device="mps", workers=2,
    project=str(Path("data/models/bot77_detector").resolve()), name="v1", exist_ok=True,  # absolute, or it lands under runs/
    fliplr=0.5, hsv_h=0.015, hsv_s=0.5, hsv_v=0.3, degrees=0, translate=0.05, scale=0.2, mosaic=0.0,
    patience=0, plots=True, val=True, verbose=False,
)
