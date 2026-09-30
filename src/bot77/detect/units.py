"""Unit detection on the arena with KataCR's YOLOv8 detectors [Wu et al. 2025, §5.2].

KataCR (MIT, third_party/katacr) trains two YOLOv8 detectors that split the ~150 classes by
object size, and appends the unit's faction as an extra output channel: for each box the
model predicts 4 box values, the class scores, and one "belong" score (friendly < 0.5 <
enemy). The NMS and the two-detector merge below port katacr/yolov8/custom_utils.py
(non_max_suppression) and katacr/yolov8/combo_detect.py.

Weights are the converted, tensor-only files in data/models/katacr (see third_party/katacr).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from ultralytics.nn.tasks import DetectionModel

from bot77.layout import Layout

MODEL_DIR = Path("data/models/katacr")
DETECTORS = ("detector1_v0.7.13", "detector2_v0.7.13")
INPUT_WH = (576, 896)  # KataCR's arena crop size (width, height)
CONF = 0.5
IOU = 0.45  # per-detector NMS
MERGE_IOU = 0.6  # across the two detectors, as in combo_detect.py
SKIP = {"padding_1", "padding_2", "padding_belong"}


@dataclass(frozen=True)
class Detection:
    name: str  # KataCR class, e.g. "hog-rider", "bar-level", "queen-tower"
    team: str  # friendly | enemy
    conf: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in source-frame pixels

    @property
    def center(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load(path: Path, device: str) -> tuple[DetectionModel, dict[int, str]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = DetectionModel(cfg=ckpt["yaml"], nc=ckpt["yaml"]["nc"], verbose=False)
    model.load_state_dict(ckpt["state_dict"])
    model.stride = torch.tensor(ckpt["stride"])
    return model.eval().to(device), {int(k): v for k, v in ckpt["names"].items()}


def _nms(pred: torch.Tensor, conf: float, iou: float) -> torch.Tensor:
    """pred: (4 + nc, N) raw head output for one image, where the last class channel is the
    faction. Returns (M, 7): x0, y0, x1, y1, conf, cls, enemy."""
    x = pred.T  # (N, 4 + nc)
    box = torchvision.ops.box_convert(x[:, :4], "cxcywh", "xyxy")
    cls_scores, bel = x[:, 4:-1], x[:, -1:]
    score, j = cls_scores.max(1, keepdim=True)
    keep = score.view(-1) > conf
    out = torch.cat((box, score, j.float(), (bel > 0.5).float()), 1)[keep]
    if not len(out):
        return out
    offset = out[:, 5:6] * 7680  # batched NMS: never suppress across classes
    return out[torchvision.ops.nms(out[:, :4] + offset, out[:, 4], iou)]


class UnitDetector:
    def __init__(self, model_dir: Path = MODEL_DIR, device: str | None = None, conf: float = CONF):
        self.device = device or _device()
        self.conf = conf
        self.models = [_load(model_dir / f"{name}.safe.pt", self.device) for name in DETECTORS]

    def arena(self, frame: np.ndarray, layout: Layout) -> tuple[np.ndarray, tuple[int, int], float, float]:
        """The detector's input crop (padded where the layout's box runs past the frame), plus
        the offset and scale to map boxes back to frame pixels."""
        s = layout.scale(frame.shape[1])
        x0, y0, x1, y1 = (round(v * s) for v in layout.katacr_arena)
        h, w = frame.shape[:2]
        crop = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        crop[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0] = frame[fy0:fy1, fx0:fx1]
        return crop, (x0, y0), crop.shape[1] / INPUT_WH[0], crop.shape[0] / INPUT_WH[1]

    @torch.no_grad()
    def detect(self, frame: np.ndarray, layout: Layout) -> list[Detection]:
        crop, (ox, oy), sx, sy = self.arena(frame, layout)
        img = cv2.resize(crop, INPUT_WH, interpolation=cv2.INTER_AREA)[:, :, ::-1]  # BGR -> RGB
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().div(255).to(self.device)
        rows, names = [], []
        for model, model_names in self.models:
            pred = model(x)
            pred = pred[0] if isinstance(pred, (list, tuple)) else pred
            for r in _nms(pred[0].float().cpu(), self.conf, IOU):
                name = model_names[int(r[5])]
                if name not in SKIP:
                    rows.append(r)
                    names.append(name)
        if not rows:
            return []
        all_rows = torch.stack(rows)
        keep = torchvision.ops.nms(all_rows[:, :4], all_rows[:, 4], MERGE_IOU).tolist()
        out = []
        for i in keep:
            x0, y0, x1, y1, conf, _, enemy = all_rows[i].tolist()
            out.append(Detection(names[i], "enemy" if enemy else "friendly", conf,
                                 (ox + x0 * sx, oy + y0 * sy, ox + x1 * sx, oy + y1 * sy)))
        return out
