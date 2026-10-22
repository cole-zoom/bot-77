"""B v0: let the bot play in BlueStacks. Starts from the home screen (taps Battle) or joins a battle in progress.
Stop it any time: `touch data/live/STOP` (it finishes the current step and exits). Every action and game is logged to
LanceDB (bot_actions, bot_games). Pull the games' replays afterwards with replay_runner.py to key them.

  uv run python scripts/live_play.py --games 1                      # one game, watch it
  uv run python scripts/live_play.py --games 30 --hours 3           # a grinding session
"""
import argparse
from pathlib import Path

import lancedb

from bot77.detect.yolo import Bot77Detector
from bot77.layout import Layout
from bot77.live.device import Device
from bot77.live import ui
from bot77.live.loop import STOP_FILE, Runner

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="127.0.0.1:5555"); ap.add_argument("--layout", default="bluestacks_live")
ap.add_argument("--games", type=int, default=1); ap.add_argument("--hours", type=float, default=1.0)
ap.add_argument("--weights", default="data/models/bot77_detector/v6l/best.pt")
a = ap.parse_args()
STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
STOP_FILE.unlink(missing_ok=True)
print(ui.banner(a.device, a.layout, getattr(a, "games", 0) or 999, a.hours), flush=True)
cards = lancedb.connect("data/lancedb").open_table("cards").search().where("kind = 'deck_card'").limit(1000).to_arrow().to_pylist()
Runner(Device(a.device), Layout.load(a.layout), cards, Bot77Detector(Path(a.weights)), max_games=a.games,
       max_hours=a.hours).run()
