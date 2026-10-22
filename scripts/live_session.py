"""A long bot session (2026-10-21): play in chunks of --chunk games, and after each chunk pull those games' replays
from the Battle Log (it keeps only 25) with replay_runner.py, until --hours are up or data/live/STOP exists.
Replays are processed into answer keys afterwards (ingest_replays.py) so the bot isn't slowed during play.

  caffeinate -i -s uv run python scripts/live_session.py --hours 4
"""
import argparse, subprocess, sys, time
from pathlib import Path

import lancedb

from bot77.detect.yolo import Bot77Detector
from bot77.layout import Layout
from bot77.live.device import Device
from bot77.live import ui
from bot77.live.loop import STOP_FILE, Runner

ap = argparse.ArgumentParser()
ap.add_argument("--hours", type=float, default=4.0); ap.add_argument("--chunk", type=int, default=20)
ap.add_argument("--device", default="127.0.0.1:5555"); ap.add_argument("--layout", default="bluestacks_live")
ap.add_argument("--weights", default="data/models/bot77_detector/v6l/best.pt")
a = ap.parse_args()
STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
STOP_FILE.unlink(missing_ok=True)
print(ui.banner(a.device, a.layout, getattr(a, "games", 0) or 999, a.hours), flush=True)
cards = lancedb.connect("data/lancedb").open_table("cards").search().where("kind = 'deck_card'").limit(1000).to_arrow().to_pylist()
det, layout, dev = Bot77Detector(Path(a.weights)), Layout.load(a.layout), Device(a.device)
end, total, chunk_no = time.time() + a.hours * 3600, 0, 0
deck = []
while time.time() < end and not STOP_FILE.exists():
    chunk_no += 1
    left_h = (end - time.time()) / 3600
    print(f"== chunk {chunk_no}: up to {a.chunk} games, {left_h:.2f} h left", flush=True)
    r = Runner(dev, layout, cards, det, max_games=a.chunk, max_hours=left_h)
    r.known_deck = deck
    r.run()
    deck, played = r.known_deck, len(r.games)
    total += played
    print(f"== chunk {chunk_no}: {played} games played ({total} total)", flush=True)
    if played == 0:
        print("no games played in this chunk: stopping (check data/raw/live_ui/unknown/)", flush=True)
        break
    # the replays of this chunk's games are the newest entries of the Battle Log
    rr = subprocess.run([sys.executable, "scripts/replay_runner.py", "-n", str(played), "--no-known", "--device", a.device],
                        capture_output=True, text=True)
    print(f"== replays pulled: {rr.stdout.strip().splitlines()[-1] if rr.stdout.strip() else rr.stderr[-300:]}", flush=True)
print(f"session done: {total} games", flush=True)
