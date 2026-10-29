"""B v0: let the bot play in BlueStacks. Starts from the home screen (taps Battle) or joins a battle in progress.
Stop it any time: `touch data/live/STOP` (it finishes the current step and exits). Every action and game is logged to
LanceDB (bot_actions, bot_games). Pull the games' replays afterwards with replay_runner.py to key them.

  uv run python scripts/live_play.py --games 1                      # one game, watch it
  uv run python scripts/live_play.py --games 30 --hours 3           # a grinding session
  ... --policy selfplay --ckpt X --decision-s 0.1 --full-bar-guard  # the network, with the full-bar band-aid
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
ap.add_argument("--device", default="127.0.0.1:5555"); ap.add_argument("--stream", action="store_true", help="read the screen from a live stream (bot77.live.stream), not 3.5-5 s screenshots"); ap.add_argument("--layout", default="bluestacks_live")
ap.add_argument("--games", type=int, default=1); ap.add_argument("--hours", type=float, default=1.0)
from bot77.models import adopted_path  # the registry's adopted live detector (phase 5)
ap.add_argument("--weights", default=adopted_path("detector_live", "data/models/bot77_detector/v6l/best.pt"))
ap.add_argument("--policy", default="rules", choices=["rules", "selfplay"],
                help="rules: Policy (playbook rules + A); selfplay: a network checkpoint (bot77.live.selfplay)")
ap.add_argument("--ckpt", default=None, help="--policy selfplay: the checkpoint")
ap.add_argument("--decision-s", type=float, default=1.0, help="--policy selfplay: seconds between decisions (0.1 for the 0.1 s models; turns on the reader's fast mode)")
ap.add_argument("--full-bar-guard", action="store_true",
                help="--policy selfplay: temporary band-aid (docs/plan-elixir-spells.md P1): after 3 s at >= 9.8 elixir with "
                     "nothing of theirs on our side, play the model's most likely affordable card instead of waiting; "
                     "logged as 'full-bar guard'")
a = ap.parse_args()
STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
STOP_FILE.unlink(missing_ok=True)
print(ui.banner(a.device, a.layout, getattr(a, "games", 0) or 999, a.hours), flush=True)
cards = lancedb.connect("data/lancedb").open_table("cards").search().where("kind = 'deck_card'").limit(1000).to_arrow().to_pylist()
factory = None
if a.policy == "selfplay":
    from bot77.live.selfplay import SelfPlayPolicy
    factory = lambda: SelfPlayPolicy(cards, a.ckpt, decision_s=a.decision_s, full_bar_guard=a.full_bar_guard)  # noqa: E731
Runner(Device(a.device, stream=a.stream), Layout.load(a.layout), cards, Bot77Detector(Path(a.weights)), max_games=a.games,
       max_hours=a.hours, policy_factory=factory, fast_reader=a.decision_s < 1.0).run()
