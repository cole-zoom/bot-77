"""Stream decoded frames from a video with ffmpeg."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np


def probe(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    s = info["streams"][0]
    return {"width": s["width"], "height": s["height"], "duration": float(info["format"]["duration"])}


def iter_frames(video: Path, fps: float, start: float = 0.0, duration: float | None = None) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (video_time_s, BGR frame) sampled at `fps`, without holding the video in memory."""
    info = probe(video)
    w, h = info["width"], info["height"]
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(video), "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = w * h * 3
    i = 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield start + i / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()
