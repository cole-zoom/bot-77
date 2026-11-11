"""Shadow mode on the phone (2026-11-10, Cole: "What if I played the deck with it watching, and it guesses what I do,
and if it doesn't do what I do, it gets penalized?"). Cole plays our deck on the phone; the bot reads the screen as in
a live game and decides every 0.1 s without ever tapping, and Cole's touches say what he actually played and where.

Cole's plays come from the touchscreen itself (`adb shell getevent` on the phone's touch device), not from the screen
read: a touch that starts on a hand slot and lifts on the arena is that card at that spot (a drag), and a tap on a slot
followed by a tap on the arena is the same play (tap-select). The phone renders the game at 1080x1920 (`wm size`) in
the middle of its 1080x2340 panel, so a touch's game y is its panel y minus 210.

Per game, two outputs:
  data/cache/bc/shadow_t01/cole_shadow_<id>.npz     the network's input at every decision with Cole's action as its
                                                    label (his hand slot, 4 = wait; the board cell, -1 for a wait), in
                                                    the imitation demos' format: train_bc.py --games mixes them in.
                                                    Every 4th game goes to shadow_t01_held/ instead (scored, not trained)
  data/live/shadow/summary.jsonl                    how often the bot agreed with him (below)

Agreement, at the decisions where Cole played: the bot's top card (of the 4) is his; the chance it gave his card; its
play chance there against where he waited. Placement: distance from its spot to his when it chose the same card.
Nothing is ever tapped and nothing is saved from the screens between games.

  uv run --no-sync python scripts/shadow_play.py [--device RZCY610FR5J] [--layout a16_live] [--ckpt <v7>] [--games 30]
Stop any time: `touch data/live/STOP` (the game in progress is saved first).
"""
import argparse
import hashlib
import json
import math
import queue
import subprocess
import threading
import time
from pathlib import Path

import sys

import lancedb
import numpy as np

from bot77.detect.yolo import Bot77Detector
from bot77.layout import Layout
from bot77.live import ui
from bot77.live.device import ADB, AIM_DY, Device, screen_kind
from bot77.live.loop import STALE_FRAME_S, STOP_FILE, Runner
from bot77.live.selfplay import SelfPlayPolicy
from bot77.live.state import LiveReader, OpponentTracker
from bot77.models import adopted_path
from bot77.sim.geometry import ours_to_engine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_defence_demos import cell_from_xy  # noqa: E402  (the fork's board binning, as every other demo)
OUT_TRAIN = ROOT / "data/cache/bc/shadow_t01"
OUT_HELD = ROOT / "data/cache/bc/shadow_t01_held"
SUMMARY = ROOT / "data/live/shadow/summary.jsonl"
TOUCH_DEV = "/dev/input/event5"  # the Galaxy A16's "sec_touchscreen"
PANEL_DY = (2340 - 1920) // 2  # the 1080x1920 game sits in the middle of the 1080x2340 panel
COLE_WEIGHT = 3.0  # per-row weight, as Cole's other demos
HOLD_EVERY = 4


class Touches:
    """Cole's touches -> plays (slot, x px, y px, time), from the phone's touch device."""

    def __init__(self, serial: str, slot_boxes: list[tuple[float, float, float, float]]):
        self.boxes, self.plays, self.raw = slot_boxes, queue.Queue(), []
        self.selected: int | None = None
        self.proc = subprocess.Popen([ADB, "-s", serial, "shell", "getevent", "-l", TOUCH_DEV],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=self._run, daemon=True).start()

    def _slot(self, x: float, y: float) -> int | None:
        return next((k for k, (x0, y0, x1, y1) in enumerate(self.boxes) if x0 <= x <= x1 and y0 <= y <= y1), None)

    def _run(self) -> None:
        x = y = 0.0
        down = None  # (x, y, t) where the current touch started
        pending = None  # "down" / "up": applied at the frame's SYN_REPORT, once the positions are in
        for line in self.proc.stdout:
            parts = line.split()
            if len(parts) < 3:
                continue
            code, val = parts[1], parts[2]
            if code == "ABS_MT_POSITION_X":
                x = float(int(val, 16))
            elif code == "ABS_MT_POSITION_Y":
                y = float(int(val, 16)) - PANEL_DY
            elif code == "ABS_MT_TRACKING_ID":
                pending = "up" if val == "ffffffff" else "down"
            elif code == "SYN_REPORT" and pending:
                t = time.time()
                if pending == "down":
                    down = (x, y, t)
                elif down is not None:
                    self._gesture(down, (x, y, t))
                    down = None
                pending = None

    def _gesture(self, d: tuple, u: tuple) -> None:
        s0, s1 = self._slot(d[0], d[1]), self._slot(u[0], u[1])
        top = min(b[1] for b in self.boxes)  # the hand's top edge: above it is the arena
        self.raw.append({"down": [round(d[0]), round(d[1])], "up": [round(u[0]), round(u[1])], "t": round(u[2], 2)})
        if s0 is not None and s1 is None and u[1] < top:  # dragged from a slot onto the arena
            self.plays.put((s0, u[0], u[1], u[2])); self.selected = None
        elif s0 is not None and s1 == s0:  # a tap on a slot: selects it (again: deselects)
            self.selected = None if self.selected == s0 else s0
        elif s0 is None and u[1] < top and self.selected is not None:  # a tap on the arena with a card selected
            self.plays.put((self.selected, u[0], u[1], u[2])); self.selected = None

    def drain(self) -> list[tuple]:
        out = []
        while not self.plays.empty():
            out.append(self.plays.get())
        return out


class ShadowPolicy(SelfPlayPolicy):
    """The self-play agent deciding every 0.1 s, never acting: each decision's input and choice are kept, and
    Cole's plays label them."""

    def __init__(self, *a, **k):
        super().__init__(*a, sample=False, **k)
        self.rows: list[dict] = []
        self._obs = None

    def _mirror(self, st, opp_plays):
        sim = super()._mirror(st, opp_plays)
        self._obs = None if sim is None else np.asarray(sim.env.get_observation_for_team(0), dtype=np.float16)
        return sim

    def decide(self, st, opp_plays):
        n = len(self.log)
        act = super().decide(st, opp_plays)
        if len(self.log) > n and self._obs is not None:
            e = self.log[-1]
            self.rows.append({"t": st.t, "obs": self._obs, "p": e["p"], "hand": list(st.hand), "label": None,
                              "act": None if act is None else (act.slot, act.x, act.y)})
        return None  # shadow: never tap

    def label(self, t: float, slot: int, x: float, y: float) -> bool:
        """Cole's play at `t` labels the last decision at or before it."""
        row = next((r for r in reversed(self.rows) if r["t"] <= t + 0.05), None)
        if row is None or row["label"] is not None:
            return False
        row["label"] = (slot, x, y)
        return True


def save_game(gid: str, pol: ShadowPolicy, result: str, raw_touches: list) -> dict | None:
    rows = pol.rows
    if len(rows) < 50:
        return None
    card = np.array([4 if r["label"] is None else r["label"][0] for r in rows], np.int64)
    cell = np.array([-1 if r["label"] is None else cell_from_xy(*ours_to_engine(r["label"][1], r["label"][2]))
                     for r in rows], np.int64)
    held = int(hashlib.md5(gid.encode()).hexdigest(), 16) % HOLD_EVERY == 0
    out = (OUT_HELD if held else OUT_TRAIN) / f"cole_shadow_{gid}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, obs=np.stack([r["obs"] for r in rows]), card=card, cell=cell,
                        t=np.array([r["t"] for r in rows], np.float32),
                        weight=np.full(len(rows), COLE_WEIGHT, np.float32), split=np.array("train"),
                        side=np.array("team0"), result=np.array(result))
    # agreement at his plays; play chance at his plays vs where he waited
    plays = [r for r in rows if r["label"] is not None]
    top = [int(np.argmax(r["p"][:4]) == r["label"][0]) for r in plays]
    p_his = [r["p"][r["label"][0]] / max(1e-9, sum(r["p"][:4])) for r in plays]
    play_at = [1 - r["p"][4] for r in plays]
    play_else = [1 - r["p"][4] for r in rows if r["label"] is None]
    dist = [math.hypot(r["act"][1] - r["label"][1], r["act"][2] - r["label"][2]) for r in plays
            if r["act"] is not None and r["act"][0] == r["label"][0]]
    s = {"game": gid, "result": result, "held": held, "decisions": len(rows), "cole_plays": len(plays),
         "touch_gestures": len(raw_touches),
         "card_top1": round(float(np.mean(top)), 3) if top else None,
         "p_his_card": round(float(np.mean(p_his)), 3) if p_his else None,
         "play_chance_at_his_plays": round(float(np.mean(play_at)), 3) if play_at else None,
         "play_chance_when_he_waited": round(float(np.mean(play_else)), 4) if play_else else None,
         "placement_tiles_same_card": round(float(np.median(dist)), 2) if dist else None,
         "file": str(out.relative_to(ROOT))}
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY, "a") as f:
        f.write(json.dumps(s) + "\n")
    (SUMMARY.parent / f"{gid}_touches.json").write_text(json.dumps(raw_touches))  # to check the touch mapping
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="RZCY610FR5J")
    ap.add_argument("--layout", default="a16_live")
    ap.add_argument("--ckpt", default="data/models/bc_school_v7/epoch03.pth", help="the model that guesses")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--weights", default=adopted_path("detector_live", "data/models/bot77_detector/v6l/best.pt"))
    ap.add_argument("--deck", default="Giant,Battle Ram,Mini P.E.K.K.A,Mega Minion,Bomber,Goblin Hut,Fireball,Arrows")
    a = ap.parse_args()
    STOP_FILE.unlink(missing_ok=True)
    cards = lancedb.connect("data/lancedb").open_table("cards").search().where("kind = 'deck_card'").limit(1000) \
        .to_arrow().to_pylist()
    by_name = {c["name"]: c["id"] for c in cards}
    deck = [by_name[n.strip()] for n in a.deck.split(",")]
    layout = Layout.load(a.layout)
    dev = Device(a.device, stream=True)
    detector = Bot77Detector(Path(a.weights))
    probe = LiveReader(layout, cards)
    touches = Touches(a.device, [layout.box(f"slot{k + 1}", 1080, 1920) for k in range(4)])
    print(ui.c(f"shadow mode on {a.device}: play our deck, the bot only watches ({a.ckpt}); stop: touch {STOP_FILE}",
               "grey"), flush=True)
    game, done = None, 0
    while done < a.games:
        im = dev.cap()
        if im is None:
            time.sleep(0.2)
            continue
        t = dev.now()
        kind = screen_kind(im)
        if game is not None and (kind == "end" or time.time() - game["seen"] > 20):
            result = Runner._result(im) if kind == "end" else "?"
            s = save_game(game["id"], game["policy"], result, touches.raw[game["n_raw"]:])
            done += s is not None
            if s:
                print(ui.c(f"  game {done}: {result} · his plays {s['cole_plays']} · bot's top card his "
                           f"{s['card_top1']} · its play chance at his plays {s['play_chance_at_his_plays']} vs "
                           f"{s['play_chance_when_he_waited']} elsewhere · spot {s['placement_tiles_same_card']} tiles",
                           "green"), flush=True)
            game = None
            if STOP_FILE.exists():
                break
            continue
        if game is None:
            if STOP_FILE.exists():
                break
            if kind != "end" and probe.read(im, t).in_battle:
                reader = LiveReader(layout, cards, detector, deck=deck)
                reader.fast = True
                game = {"id": time.strftime("%Y-%m-%d_%H%M%S"), "reader": reader, "opp": OpponentTracker(),
                        "policy": ShadowPolicy(cards, a.ckpt, decision_s=0.1), "seen": time.time(),
                        "n_raw": len(touches.raw), "plays": 0}
                touches.drain()  # touches from the menus
                print(ui.c(f"  … game {game['id']}", "grey"), flush=True)
            else:
                time.sleep(0.2)
            continue
        st = game["reader"].read(im, t)
        if not st.in_battle:
            continue
        game["seen"] = time.time()
        game["opp"].update(st)
        for slot, px, py, tp in touches.drain():
            tile, ox, oy = game["reader"].tile_px(im.shape[1])
            x, y = (px - ox) / tile, (py - oy) / tile - AIM_DY  # where his unit lands, as the bot's own aim
            game["reader"].played(slot, st.t)  # the cycle tracker, as after the bot's own play
            if game["policy"].label(tp - dev.t0, slot, x, y):  # his touch on the device clock
                game["plays"] += 1
        if time.time() - dev.stream.t_frame > STALE_FRAME_S:
            continue
        game["policy"].decide(st, game["opp"].plays)


if __name__ == "__main__":
    main()
