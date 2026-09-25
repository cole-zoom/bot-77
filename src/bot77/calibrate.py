"""Draw a layout's regions on random frames of a video, for eyeballing a calibration."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import cv2
import numpy as np

from bot77.layout import Layout


def video_duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout)


def read_frame(video: Path, t: float) -> np.ndarray:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
         "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, check=True,
    )
    return cv2.imdecode(np.frombuffer(out.stdout, np.uint8), cv2.IMREAD_COLOR)


def draw_regions(frame: np.ndarray, layout: Layout) -> np.ndarray:
    out = frame.copy()
    h, w = frame.shape[:2]
    for i, name in enumerate(layout.regions):
        x0, y0, x1, y1 = layout.box(name, w, h)
        color = [(0, 255, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0)][i % 4]
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 3)
        cv2.putText(out, name, (x0 + 3, max(y0 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def overlay_sheet(video: Path, layout: Layout, out_path: Path, n: int = 10, seed: int = 0,
                  thumb_w: int = 432) -> list[float]:
    """Contact sheet of `n` random frames with regions drawn. Returns the sampled times."""
    rng = random.Random(seed)
    duration = video_duration(video)
    times = sorted(rng.uniform(0, duration) for _ in range(n))
    thumbs = []
    for t in times:
        img = draw_regions(read_frame(video, t), layout)
        img = cv2.resize(img, (thumb_w, round(img.shape[0] * thumb_w / img.shape[1])))
        cv2.putText(img, f"{t:.1f}s", (img.shape[1] // 2 - 40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        thumbs.append(img)
    cols = 5
    while len(thumbs) % cols:
        thumbs.append(np.zeros_like(thumbs[0]))
    rows = [np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(rows))
    return times


def region_strips(video: Path, layout: Layout, out_path: Path, times: list[float], pad: int = 12) -> None:
    """One row per region: that region's crop (plus `pad` px of context) from each sampled frame."""
    frames = [read_frame(video, t) for t in times]
    rows = []
    for name in layout.regions:
        crops = []
        for f in frames:
            h, w = f.shape[:2]
            x0, y0, x1, y1 = layout.box(name, w, h)
            c = f[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad].copy()
            cv2.rectangle(c, (min(pad, x0), min(pad, y0)), (c.shape[1] - pad, c.shape[0] - pad), (0, 255, 0), 1)
            crops.append(c)
        ch = max(c.shape[0] for c in crops)
        crops = [cv2.copyMakeBorder(c, 0, ch - c.shape[0], 0, 4, cv2.BORDER_CONSTANT) for c in crops]
        row = np.hstack(crops)
        label = np.zeros((ch, 170, 3), np.uint8)
        cv2.putText(label, name, (4, min(ch - 4, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        rows.append(np.hstack([label, row]))
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 6, 0, width - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    cv2.imwrite(str(out_path), np.vstack(rows))
