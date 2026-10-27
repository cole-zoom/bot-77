"""The screen as a live stream instead of screenshots (2026-10-26).

`adb screencap` takes 3.5-5 s on BlueStacks (the emulator's own capture; raw or PNG alike), which held the agent to a
decision every ~4.5 s (it was trained at one a second). Android's screenrecord streams H.264 over the same ADB link;
ffmpeg decodes it (one thread: frame threading holds frames back) and a reader thread keeps only the newest frame.
screenrecord stops itself after --time-limit (at most 180 s), so it is restarted; the screen only sends frames when
something changes, so the last frame is the current screen however old it is.

The game recording comes from the same stream (`record_to`: the H.264 is copied into the game's file as it arrives):
a scrcpy recorder alongside took the encoder at each game start and stalled the stream for 30 s or more (2026-10-27).
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
        self.rec: subprocess.Popen | None = None  # the game recording's muxer (record_to)
        self.rec_from = 0  # only streams started from this one on feed it (each starts with the H.264 headers)
        self.starts = 0
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while self.alive:
            src = subprocess.Popen([self.adb, "-s", self.serial, "exec-out", "screenrecord", "--output-format=h264",
                                    "--time-limit", str(LIMIT_S), "--size", f"{W}x{H}", "-"],
                                   stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            dec = subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-threads", "1", "-probesize", "65536",
                                    "-analyzeduration", "0", "-f", "h264", "-i", "pipe:", "-f", "rawvideo",
                                    "-pix_fmt", "bgr24", "pipe:"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL)
            self.procs = [src, dec]
            self.starts += 1
            threading.Thread(target=self._pump, args=(src, dec, self.starts), daemon=True).start()
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

    def _pump(self, src: subprocess.Popen, dec: subprocess.Popen, start: int) -> None:
        """screenrecord's H.264 -> the decoder, and into the game recording."""
        try:
            while True:
                b = src.stdout.read1(1 << 16)
                if not b:
                    break
                dec.stdin.write(b)
                dec.stdin.flush()
                with self.lock:
                    if self.rec is not None and start >= self.rec_from:
                        try:
                            self.rec.stdin.write(b)
                        except (BrokenPipeError, OSError):
                            self.rec = None
        except (BrokenPipeError, ValueError, OSError):
            pass
        finally:
            try:
                dec.stdin.close()
            except OSError:
                pass

    def record_to(self, path) -> None:
        """Record the stream into `path` (.mkv) until record_to(None): a muxer stamping each frame with its arrival
        time (screenrecord's H.264 has no timestamps and its frame rate varies), kept across screenrecord's restarts.
        Starting one restarts the stream (~2 s): screenrecord sends the H.264 headers only when it starts."""
        with self.lock:
            if self.rec is not None:
                try:
                    self.rec.stdin.close()  # the muxer finishes the file on end of input
                except OSError:
                    pass
                self.rec = None
            if path is None:
                return
            self.rec = subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-y", "-use_wallclock_as_timestamps", "1",
                                         "-f", "h264", "-i", "pipe:", "-c", "copy", str(path)],
                                        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self.rec_from = self.starts + 1
        for p in self.procs:
            p.kill()

    def latest(self) -> np.ndarray | None:
        return None if self.frame is None else self.frame.copy()

    def close(self) -> None:
        self.alive = False
        self.record_to(None)
        for p in self.procs:
            p.kill()
