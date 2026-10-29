"""Lab battles, the BlueStacks side (docs/plan-sim-1to1.md P0.4): wait for the friendly battle to start, then place
this side's half of a lab test (config/lab/*.yaml, bot77.lab) on schedule over ADB, with the live bot's own screen
reading (bot77.live.state.LiveReader: clock, hand, elixir) and taps (bot77.live.device.deploy). A card that isn't in
hand is cycled in first: the cheapest card not needed next goes to the back of the other lane (spells into its back
corner), only when that leaves enough elixir for the scripted card by its time. Every tap is logged to
data/lab/sessions/<date>.jsonl (gitignored) for scripts/lab_compare.py.

  uv run python scripts/lab_play.py --list                 # the tests, in priority order
  uv run python scripts/lab_play.py --test L01             # live: start it, then Cole starts the friendly battle
  uv run python scripts/lab_play.py --test L01 --dry-run data/raw/live_games/bot_2026-10-29_070758
                                                           # no ADB: saved *_at.jpg frames as the screen, taps logged
                                                           # and drawn on the frames (data/lab/dryrun/<test>/)

Never runs next to scripts/live_play.py or the replay runner (one ADB user at a time). Stop: Ctrl-C.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import lancedb

from bot77 import lab
from bot77.layout import Layout
from bot77.live import device as live_device
from bot77.live.device import deploy, screen_kind
from bot77.live.state import LiveReader

SESSIONS = Path("data/lab/sessions")
DRYRUN = Path("data/lab/dryrun")
EXCLUSIVE = ("live_play.py", "replay_runner.py", "live_with_replays.py", "live_session.py", "lab_play.py")
ACTION_GAP_S = 1.2  # after a tap, let the hand animate and the elixir drop before the next decision


def others_running() -> list[str]:
    """Other ADB users of BlueStacks (pgrep), never this process or the `uv run` wrapper that started it."""
    mine = {os.getpid(), os.getppid()}
    out = []
    for pat in EXCLUSIVE:
        r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
        pids = [int(p) for p in r.stdout.split() if int(p) not in mine]
        if pids:
            out.append(f"{pat} (pid {', '.join(map(str, pids))})")
    return out


class GameClock:
    """Game time from the clock reads (whole seconds) plus the wall clock between them: anchored on the frame where
    the shown second changes, so it is good to about a frame, not a second."""

    def __init__(self):
        self.anchor: tuple[float, float] | None = None  # (game t, wall t)
        self.last: int | None = None

    def update(self, elapsed: int | None, now: float) -> None:
        if elapsed is None:
            return
        if self.last is not None and elapsed == self.last + 1:
            self.anchor = (float(elapsed), now)  # the second just turned
        elif self.anchor is None or abs(self.t(now) - (elapsed + 0.5)) > 1.5:
            self.anchor = (elapsed + 0.5, now)  # first read, or a jump: the middle of the shown second
        self.last = elapsed

    def t(self, now: float) -> float | None:
        return None if self.anchor is None else self.anchor[0] + (now - self.anchor[1])


class FrameDevice:
    """Saved frames as the screen (dry run): each frame stays up for `hold` s of a fake clock that advances `dt` per
    capture, so a run takes no wall time. Taps are only logged."""

    def __init__(self, frames: list[Path], hold: float = 1.0, dt_: float = 0.25):
        self.frames, self.hold, self.dt = frames, hold, dt_
        self.clock = 0.0
        self.taps: list[tuple[float, float, float]] = []

    def now(self) -> float:
        return self.clock

    def cap(self):
        i = int(self.clock / self.hold)
        self.clock += self.dt
        return None if i >= len(self.frames) else cv2.imread(str(self.frames[i]))

    def tap(self, x: float, y: float) -> None:
        self.taps.append((self.clock, x, y))


class LabRunner:
    def __init__(self, dev, layout: Layout, cards: list[dict], test: lab.LabTest, dry_run: bool = False,
                 log=print, out_dir: Path | None = None):
        self.dev, self.layout, self.test, self.dry_run, self.log = dev, layout, test, dry_run, log
        by_name = {c["name"]: c for c in cards}
        self.id_of = {n: by_name[n]["id"] for n in test.decks["bluestacks"]}
        self.name_of = {v: k for k, v in self.id_of.items()}
        self.cost = {by_name[n]["id"]: float(by_name[n]["elixir"]) for n in test.decks["bluestacks"]}
        self.reader = LiveReader(layout, cards)  # no detector: clock, hand and elixir are all this needs
        deck = list(self.id_of.values())
        self.reader.deck, self.reader.reader = deck, self.reader.full.restrict(set(deck))
        self.queue = test.side("bluestacks")
        self.cole = test.side("cole")
        self.dumps = test.dump_spots()
        self.clock = GameClock()
        self.actions: list[dict] = []
        self.last_action = -1e9
        self.started_wall = None
        self.cued: set[int] = set()
        self.out_dir = out_dir
        self.reads = 0

    # --- one decision ---
    def _tap(self, im, card_id: int, slot: int, tile: tuple[int, int], kind: str, t: float, st, sched=None) -> None:
        gx, gy = lab.bluestacks_ground_point(tile)
        taps = deploy(self.dev, self.layout, self.reader, SimpleNamespace(slot=slot, x=gx, y=gy), im.shape)
        k = 1080 / im.shape[1]  # frame px -> device px (saved frames are half size)
        row = {"kind": kind, "card": self.name_of[card_id], "tile": list(tile), "slot": slot,
               "t_sched": None if sched is None else sched.t, "t_est": round(t, 2), "clock_read": st.elapsed_s,
               "elixir": None if st.elixir is None else round(st.elixir, 2), "hand": [self.name_of.get(h) for h in st.hand],
               "taps_px": [round(v * k) for v in taps], "aim_dy": live_device.AIM_DY, "wall": time.time(),
               "placement_i": None if sched is None else sched.i}
        self.actions.append(row)
        self.last_action = self.dev.now()
        late = "" if sched is None else f" ({t - sched.t:+.1f} s vs plan)"
        self.log(f"  {lab.clock_text(t)}  {kind:8s} {row['card']:16s} tile {tuple(tile)}{late}  elixir {row['elixir']}")
        if self.out_dir is not None:
            self._draw(im, taps, tile, f"{len(self.actions):02d}_{kind}_{row['card'].replace(' ', '_')}.jpg")

    def step(self, st, im, now: float) -> None:
        t = self.clock.t(now)
        if t is None:
            return
        for p in self.cole:  # a heads-up on the terminal for Cole's half
            if p.i not in self.cued and p.t - 5 <= t:
                self.cued.add(p.i)
                self.log(f"  >>> COLE at {lab.clock_text(p.t)}: {p.card} on tile {p.tile} {p.cue}")
        if not self.queue or now - self.last_action < ACTION_GAP_S or st.elixir is None:
            return
        nxt = self.queue[0]
        cid = self.id_of[nxt.card]
        if cid in st.hand:
            if t >= nxt.t and st.elixir >= self.cost[cid] - 0.05:
                self._tap(im, cid, st.hand.index(cid), nxt.tile, "scripted", t, st, nxt)
                self.queue.pop(0)
            return
        # not in hand: cycle the cheapest card that isn't needed for the next two placements, if the elixir allows
        keep = {self.id_of[p.card] for p in self.queue[:2]}
        cands = sorted((self.cost[c], k, c) for k, c in enumerate(st.hand) if c is not None and c not in keep)
        if not cands:
            return
        cost, k, c = cands[0]
        regain = max(0.0, nxt.t - t) / lab.ELIXIR_S * max(1, st.multiplier)
        if st.elixir >= cost - 0.05 and (st.elixir - cost + regain >= self.cost[cid] or t >= nxt.t):
            spot = self.dumps["spell" if self.name_of[c] in lab.SPELLS else "troop"]
            self._tap(im, c, k, spot, "cycle", t, st)

    # --- the run ---
    def run(self, wait_s: float = 600.0) -> dict:
        t_start = time.time()
        in_battle, fails = False, 0
        while True:
            im = self.dev.cap()
            now = self.dev.now()
            if im is None:
                if self.dry_run:
                    self.log("frames ran out"); break
                fails += 1
                if fails >= 8:
                    self.log("screenshots keep failing: stopping"); break
                time.sleep(1.0); continue
            fails = 0
            st = self.reader.read(im, now)
            self.reads += 1
            if not in_battle:
                if st.in_battle:
                    in_battle, self.started_wall = True, time.time()
                    self.log(f"battle on screen (clock {st.time_left_s} s left, elixir {st.elixir}); "
                             f"hand {[self.name_of.get(h) for h in st.hand]}")
                    if self.out_dir is not None:
                        self._draw(im, None, None, "00_grid.jpg")
                elif not self.dry_run and time.time() - t_start > wait_s:
                    self.log("no battle came: stopping"); break
                else:
                    if not self.dry_run:
                        time.sleep(0.2)
                    continue
            if screen_kind(im) == "end":
                self.log("end screen"); break
            self.clock.update(st.elapsed_s, now)
            self.step(st, im, now)
            t = self.clock.t(now)
            if t is not None and not self.queue and t > self.test.end_t:
                self.log(f"{self.test.tail_s:.0f} s past the last placement: done (the battle runs on; Cole can let it end)")
                break
            if not self.dry_run:
                time.sleep(0.03)
        return {"test": self.test.id, "dry_run": self.dry_run, "started": self.started_wall, "ended": time.time(),
                "actions": self.actions, "missed": [{"card": p.card, "tile": list(p.tile), "t": p.t} for p in self.queue]}

    def _draw(self, im, taps, tile, name: str) -> None:
        """The frame with the engine grid laid over the arena (bot77.sim.geometry), the target tile filled and the
        taps marked: checks the screen <-> engine mapping by eye (the towers should sit on their engine cells)."""
        from bot77.sim.geometry import engine_to_ours
        out = im.copy()
        size, ox, oy = self.reader.tile_px(im.shape[1])
        px = lambda ex, ey: (round(ox + engine_to_ours(ex, ey)[0] * size), round(oy + engine_to_ours(ex, ey)[1] * size))  # noqa: E731
        for i in range(lab.ENGINE_W + 1):
            cv2.line(out, px(i - 0.5, -0.5), px(i - 0.5, lab.ENGINE_H - 0.5), (255, 255, 255), 1)
        for j in range(lab.ENGINE_H + 1):
            cv2.line(out, px(-0.5, j - 0.5), px(lab.ENGINE_W - 0.5, j - 0.5), (255, 255, 255), 1)
        for (tx, ty) in ((3, 6), (14, 6), (3, 27), (14, 27)):  # the engine's princess tower centres
            cv2.circle(out, px(tx, ty), 4, (0, 0, 255), -1)
        if tile is not None:
            x, y = tile
            cv2.rectangle(out, px(x - 0.5, y + 0.5), px(x + 0.5, y - 0.5), (0, 255, 255), 2)
        if taps is not None:
            for (x, y) in (taps[:2], taps[2:]):
                cv2.drawMarker(out, (round(x), round(y)), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.out_dir / name), out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", help="test id (L00, L01, ...)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", type=int, default=1, help="repeat number of this test tonight (logged)")
    ap.add_argument("--device", default="127.0.0.1:5555")
    ap.add_argument("--layout", default="bluestacks_live")
    ap.add_argument("--no-stream", action="store_true", help="3.5-5 s screenshots instead of the screen stream")
    ap.add_argument("--wait", type=float, default=600, help="seconds to wait for the battle to start")
    ap.add_argument("--aim-dy", type=float, default=None, help=f"override the live tap offset (default {live_device.AIM_DY})")
    ap.add_argument("--dry-run", metavar="GAME_DIR", help="a data/raw/live_games/<game> folder: its *_at.jpg frames are the screen")
    ap.add_argument("--hold", type=float, default=1.0, help="dry run: fake seconds each frame stays up")
    a = ap.parse_args()
    tests = lab.load_tests()
    if a.list:
        for t in tests.values():
            print(f"{t.id}  {t.name:42s} {t.mechanic}")
        return
    test = tests[a.test]
    if a.aim_dy is not None:
        live_device.AIM_DY = a.aim_dy
    cards = lancedb.connect("data/lancedb").open_table("cards").search().where("kind = 'deck_card'").limit(1000).to_arrow().to_pylist()
    layout = Layout.load(a.layout)
    if a.dry_run:
        frames = sorted(Path(a.dry_run).glob("*_at.jpg"))
        if not frames:
            sys.exit(f"no *_at.jpg frames in {a.dry_run}")
        dev = FrameDevice(frames, hold=a.hold)
        out = DRYRUN / test.id
    else:
        busy = others_running()
        if busy:
            sys.exit("not starting: " + "; ".join(busy) + " is running (one ADB user at a time)")
        dev = live_device.Device(a.device, stream=not a.no_stream)
        out = None
    print(f"{test.id} {test.name}: {len(test.side('bluestacks'))} placements here, {len(test.side('cole'))} for Cole"
          f"{' (DRY RUN)' if a.dry_run else ''}")
    for p in test.side("bluestacks"):
        print(f"  bot  {lab.clock_text(p.t)}  {p.card:16s} tile {p.tile}")
    for p in test.side("cole"):
        print(f"  Cole {lab.clock_text(p.t)}  {p.card:16s} tile {p.tile}  {p.cue}")
    print("waiting for the battle..." if not a.dry_run else "", flush=True)
    r = LabRunner(dev, layout, cards, test, dry_run=bool(a.dry_run), out_dir=out)
    try:
        rec = r.run(wait_s=a.wait)
    except KeyboardInterrupt:
        rec = {"test": test.id, "dry_run": bool(a.dry_run), "started": r.started_wall, "ended": time.time(),
               "actions": r.actions, "missed": [{"card": p.card} for p in r.queue], "interrupted": True}
    finally:
        if getattr(dev, "stream", None) is not None:
            dev.stream.close()
    rec["run"] = a.run
    if a.dry_run:
        rec["frames"] = a.dry_run
        rec["taps_logged"] = len(dev.taps)
        out.mkdir(parents=True, exist_ok=True)
        (out / "run.json").write_text(json.dumps(rec, indent=1))
        print(f"dry run: {len(rec['actions'])} placements ({sum(x['kind'] == 'scripted' for x in rec['actions'])} scripted), "
              f"{len(rec['missed'])} not done, {len(dev.taps)} taps logged, {r.reads} frames read -> {out}")
    else:
        SESSIONS.mkdir(parents=True, exist_ok=True)
        with (SESSIONS / f"{dt.date.today().isoformat()}.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"logged to {SESSIONS}/{dt.date.today().isoformat()}.jsonl; missed: {rec['missed'] or 'none'}")


if __name__ == "__main__":
    main()
