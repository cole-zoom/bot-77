"""The screen as a live stream instead of screenshots (2026-10-26).

`adb screencap` takes 3.5-5 s on BlueStacks (the emulator's own capture; raw or PNG alike), which held the agent to a
decision every ~4.5 s (it was trained at one a second). Android's screenrecord streams H.264 over the same ADB link;
ffmpeg decodes it (one thread: frame threading holds frames back) and a reader thread keeps only the newest frame.
screenrecord stops itself after --time-limit (at most 180 s), so it is restarted; the screen only sends frames when
something changes, so the last frame is the current screen however old it is.
"""
from __future__ import annotations

import subprocess
import threading
import time

import numpy as np

W, H = 1080, 1920
LIMIT_S = 170


class ScreenStream:
    def __init__(self, adb: str, serial: str):
        self.adb, self.serial = adb, serial
        self.frame: np.ndarray | None = None
        self.t_frame = 0.0
        self.frames = 0
        self.alive = True
        self.procs: list[subprocess.Popen] = []
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while self.alive:
            src = subprocess.Popen([self.adb, "-s", self.serial, "exec-out", "screenrecord", "--output-format=h264",
                                    "--time-limit", str(LIMIT_S), "--size", f"{W}x{H}", "-"],
                                   stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            dec = subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-threads", "1", "-probesize", "65536",
                                    "-analyzeduration", "0", "-f", "h264", "-i", "pipe:", "-f", "rawvideo",
                                    "-pix_fmt", "bgr24", "pipe:"], stdin=src.stdout, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL)
            src.stdout.close()  # ffmpeg owns it now
            self.procs = [src, dec]
            n = W * H * 3
            while self.alive:
                b = dec.stdout.read(n)
                if len(b) < n:
                    break
                self.frame = np.frombuffer(b, np.uint8).reshape(H, W, 3)
                self.t_frame = time.time()
                self.frames += 1
            for p in (src, dec):
                p.kill()
            if self.alive:
                time.sleep(0.5)

    def latest(self) -> np.ndarray | None:
        return None if self.frame is None else self.frame.copy()

    def close(self) -> None:
        self.alive = False
        for p in self.procs:
            p.kill()
