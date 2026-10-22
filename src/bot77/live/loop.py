"""Battle -> play -> end -> repeat. Safety: a STOP file ends the run at the next check; max games / hours; an
unrecognised screen for too long or repeated screenshot failures stop the bot rather than tapping blindly. Every
action and game goes to LanceDB (bot_actions, bot_games); each game's replay is pulled later by replay_runner.py and
keyed by the usual pipeline, so the bot's games become training data."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import lancedb
import pyarrow as pa

from bot77.live.device import POS, deploy, screen_kind
from bot77.live.policy import Policy
from bot77.live.state import LiveReader, OpponentTracker

STOP_FILE = Path("data/live/STOP")


class Runner:
    def __init__(self, dev, layout, cards: list[dict], detector=None, db_dir: Path = Path("data/lancedb"),
                 max_games: int = 1, max_hours: float = 1.0, unknown_limit_s: float = 240.0, step_s: float = 0.35,
                 dry_run: bool = False, log=print):
        self.dev, self.layout, self.cards, self.detector = dev, layout, cards, detector
        self.db = lancedb.connect(db_dir) if not dry_run else None
        self.max_games, self.max_hours, self.unknown_limit_s, self.step_s = max_games, max_hours, unknown_limit_s, step_s
        self.dry_run, self.log = dry_run, log
        self.games, self.actions = [], []
        self.probe = LiveReader(layout, cards)  # is a battle on screen? (no detector: clock + elixir are enough)

    def _save(self, table: str, rows: list[dict]) -> None:
        if self.db is None or not rows:
            return
        t = pa.Table.from_pylist(rows)
        if table in self.db.table_names(limit=10_000):
            self.db.open_table(table).add(t)
        else:
            self.db.create_table(table, t)

    def run(self) -> None:
        start, last_known, fails = time.time(), time.time(), 0
        game = None
        while True:
            if STOP_FILE.exists():
                self.log("STOP file found: stopping"); break
            if len(self.games) >= self.max_games and game is None:
                self.log(f"{len(self.games)} games played: done"); break
            if time.time() - start > self.max_hours * 3600 and game is None:
                self.log("time limit: done"); break
            im = self.dev.cap()
            if im is None:
                fails += 1
                if fails >= 5:
                    self.log("screenshots keep failing: stopping"); break
                time.sleep(2); continue
            fails = 0
            t = self.dev.now()
            kind = screen_kind(im)
            if game is not None:
                st = game["reader"].read(im, t)
            else:
                st = None
            if kind == "end":
                if game is not None:
                    self._finish(game, t)
                    game = None
                ok = POS.get("end_ok")
                if ok and not self.dry_run:
                    self.dev.tap(ok[0] + 60, ok[1] + 30)
                last_known = time.time(); time.sleep(2.0); continue
            if kind == "home" and game is None:
                if len(self.games) < self.max_games:
                    tap = POS.get("home_battle_tap")
                    self.log("home: tapping Battle")
                    if tap and not self.dry_run:
                        self.dev.tap(*tap)
                    game = self._new_game(t)
                last_known = time.time(); time.sleep(3.0); continue
            if game is None:  # a battle already running (e.g. the dry run starts mid-video) or loading
                if self.probe.read(im, t).in_battle:
                    game = self._new_game(t)
                    st = game["reader"].read(im, t)
            if st is not None and st.in_battle:
                last_known = time.time()
                self._step(game, st, im)
            elif time.time() - last_known > self.unknown_limit_s:
                self.log(f"unrecognised screen for {self.unknown_limit_s:.0f} s: stopping (no blind taps)"); break
            time.sleep(self.step_s)
        if game is not None:
            self._finish(game, self.dev.now())

    def _new_game(self, t: float) -> dict:
        gid = time.strftime("bot_%Y-%m-%d_%H%M%S")
        self.log(f"game {gid} starting")
        return {"id": gid, "t0": t, "reader": LiveReader(self.layout, self.cards, self.detector),
                "policy": Policy(self.cards), "opp": OpponentTracker(), "n": 0}

    def _step(self, game: dict, st, im) -> None:
        game["opp"].update(st)
        act = game["policy"].decide(st, game["opp"].plays)
        if act is None:
            return
        taps = deploy(self.dev, self.layout, game["reader"], act, im.shape)
        game["policy"].record_own(st.t, act.card)
        game["n"] += 1
        row = {"game_id": game["id"], "t": st.t, "card": act.card, "slot": act.slot, "x_tiles": act.x, "y_tiles": act.y,
               "reason": act.reason, "elixir": st.elixir, "hand": json.dumps(st.hand), "scores": json.dumps(act.scores),
               "taps": json.dumps([round(v) for v in taps]), "time_left_s": st.time_left_s}
        self.actions.append(row)
        self.log(f"  {st.t:7.1f}s  {act.card:16s} -> ({act.x:4.1f}, {act.y:4.1f})  {act.reason}  elixir {st.elixir:.1f}")

    def _finish(self, game: dict, t: float) -> None:
        rows = [a for a in self.actions if a["game_id"] == game["id"]]
        self._save("bot_actions", rows)
        g = {"game_id": game["id"], "t_start": game["t0"], "t_end": t, "n_actions": game["n"],
             "deck": json.dumps(game["reader"].deck), "opp_plays": len(game["opp"].plays)}
        self._save("bot_games", [g])
        self.games.append(g)
        self.log(f"game {game['id']} over: {game['n']} plays by the bot, {len(game['opp'].plays)} opponent plays seen")
