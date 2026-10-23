"""B v0 offline check: the whole loop (bot77.live.loop.Runner) against a recorded BlueStacks video played back at real
speed instead of the live screen. Taps are only logged. Checks battle recognition, decision cadence, tap positions and
the end screen, without touching BlueStacks.

  uv run python scripts/live_dryrun.py data/raw/replays/2026-10-20/replay_24.mp4 --layout replay_bottom --start 5
"""
import argparse
from pathlib import Path

import lancedb

from bot77.detect.yolo import Bot77Detector
from bot77.layout import Layout
from bot77.live.device import FakeDevice
from bot77.live.loop import Runner

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("--layout", default="replay_bottom")
ap.add_argument("--start", type=float, default=0.0); ap.add_argument("--speed", type=float, default=1.0)
from bot77.models import adopted_path  # the registry's adopted live detector (phase 5)
ap.add_argument("--weights", default=adopted_path("detector_live", "data/models/bot77_detector/v6l/best.pt"))
a = ap.parse_args()
cards = lancedb.connect("data/lancedb").open_table("cards").search().where("kind = 'deck_card'").limit(1000).to_arrow().to_pylist()
layout = Layout.load(a.layout)
det = Bot77Detector(Path(a.weights))
dev = FakeDevice(Path(a.video), start=a.start, speed=a.speed)
r = Runner(dev, layout, cards, det, max_games=1, max_hours=0.2, unknown_limit_s=60, dry_run=True)
r.run()
w, h = 1080, 1920
bad = [(t, x, y) for t, x, y in dev.taps if not (0 <= x < w and 0 <= y < h)]
print(f"{len(dev.taps)} taps logged ({len(r.actions)} deploys), {len(bad)} off screen; games: {r.games}")
