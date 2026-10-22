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

import cv2

from bot77.live.device import POS, between_games_tap, deploy, screen_kind
from bot77.live.policy import Policy
from bot77.live.state import LiveReader, OpponentTracker

STOP_FILE = Path("data/live/STOP")
UNKNOWN_DIR = Path("data/raw/live_ui/unknown")
GAMES_DIR = Path("data/raw/live_games")  # per game: screen recording + a frame at each play and 2 s after (gitignored)
SHOT_W = 540  # saved frames are halved (review pages, not training)  # screens met between games (gitignored), for new rules


class Runner:
    def __init__(self, dev, layout, cards: list[dict], detector=None, db_dir: Path = Path("data/lancedb"),
                 max_games: int = 1, max_hours: float = 1.0, unknown_limit_s: float = 240.0, step_s: float = 0.1,
                 dry_run: bool = False, record: bool = True, log=print):
        self.dev, self.layout, self.cards, self.detector = dev, layout, cards, detector
        self.db = lancedb.connect(db_dir) if not dry_run else None
        self.max_games, self.max_hours, self.unknown_limit_s, self.step_s = max_games, max_hours, unknown_limit_s, step_s
        self.dry_run, self.log, self.record = dry_run, log, record
        self.games, self.actions = [], []
        self.last_nudge = self.last_shot = 0.0
        self.known_deck: list[int] = []  # learned in the first game, reused after (the deck changes rarely)
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
        self.current = None
        try:
            self._run()
        finally:  # whatever happens, close the current game's recording and log
            if self.current is not None:
                self._finish(self.current, self.dev.now())

    def _run(self) -> None:
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
                    game = self.current = None
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
                    game = self.current = self._new_game(t)
                last_known = time.time(); time.sleep(3.0); continue
            if game is None:  # a battle already running (e.g. the dry run starts mid-video) or loading
                if self.probe.read(im, t).in_battle:
                    game = self.current = self._new_game(t)
                    st = game["reader"].read(im, t)
            if st is not None and st.in_battle:
                last_known = time.time()
                try:
                    self._step(game, st, im)
                except Exception as e:  # noqa: BLE001 — one bad step must not stop a session mid-battle
                    import traceback
                    self.log(f"step error (skipped): {e!r}")
                    traceback.print_exc()
            elif game is None and time.time() - last_known > 4.0 and time.time() - self.last_nudge > 2.0:
                # between games: a safe button by its text (Choose / OK / Claim ...; never buy / gems / offers), the
                # left card on "Choose your reward", else a tap on empty space near the top (chest openings). The
                # unknown-screen limit still applies; a screenshot of each new screen is kept for review.
                tap = between_games_tap(im)
                x, y, why = tap if tap else (540, 330, "nudge")
                self.log(f"between games: {why}")
                if not self.dry_run:
                    self.dev.tap(x, y)
                self.last_nudge = time.time()
                if time.time() - self.last_shot > 30:
                    UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(UNKNOWN_DIR / time.strftime("%Y%m%d_%H%M%S.jpg")), im)
                    self.last_shot = time.time()
            if st is None or not st.in_battle:
                if time.time() - last_known > self.unknown_limit_s:
                    self.log(f"unrecognised screen for {self.unknown_limit_s:.0f} s: stopping"); break
            time.sleep(self.step_s)

    def _new_game(self, t: float) -> dict:
        gid = time.strftime("bot_%Y-%m-%d_%H%M%S")
        self.log(f"game {gid} starting")
        (GAMES_DIR / gid).mkdir(parents=True, exist_ok=True)
        rec = None
        if self.record and not self.dry_run:
            import os, subprocess
            from bot77.live.device import ADB
            rec = subprocess.Popen(["scrcpy", "-s", self.dev.s, "--no-window", "--no-audio", "--no-control", "--max-fps", "30",
                                    "--video-bit-rate", "8M", f"--record={GAMES_DIR / gid / 'game.mkv'}"],
                                   env={**os.environ, "ADB": ADB}, stdout=subprocess.DEVNULL,
                                   stderr=open(GAMES_DIR / gid / "scrcpy.log", "w"))
        reader = LiveReader(self.layout, self.cards, self.detector)
        if len(self.known_deck) == 8:  # our deck from earlier games: read only those 8 cards from the first frame
            reader.deck, reader.reader = list(self.known_deck), reader.full.restrict(set(self.known_deck))  # (~1 s -> ~0.1 s)
        return {"id": gid, "t0": t, "reader": reader,
                "policy": Policy(self.cards), "opp": OpponentTracker(), "n": 0, "rec": rec, "after": []}

    def _shot(self, game: dict, im, name: str) -> str:
        f = GAMES_DIR / game["id"] / name
        cv2.imwrite(str(f), cv2.resize(im, (SHOT_W, round(im.shape[0] * SHOT_W / im.shape[1]))), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return name

    def _step(self, game: dict, st, im) -> None:
        for k, due in list(game["after"]):  # the frame ~2 s after each play: what came of it
            if st.t >= due:
                self._shot(game, im, f"{k:03d}_after.jpg"); game["after"].remove((k, due))
        game["opp"].update(st)
        act = game["policy"].decide(st, game["opp"].plays)
        if act is None:
            return
        k = game["n"]
        shot = self._shot(game, im, f"{k:03d}_at.jpg")
        taps = deploy(self.dev, self.layout, game["reader"], act, im.shape)
        game["after"].append((k, st.t + 2.0))
        game["policy"].record_own(st.t, act.card)
        game["n"] += 1
        row = {"game_id": game["id"], "t": float(st.t), "card": act.card, "slot": act.slot, "x_tiles": float(act.x), "y_tiles": float(act.y),
               "reason": act.reason, "elixir": float(st.elixir), "hand": json.dumps(st.hand, default=float),
               "scores": json.dumps([round(float(v), 4) for v in act.scores], default=float),
               "taps": json.dumps([round(float(v)) for v in taps], default=float), "time_left_s": st.time_left_s, "k": k, "shot": shot,
               "frame_w": im.shape[1], "frame_h": im.shape[0],
               "threats": json.dumps([[u.card, round(u.x, 1), round(u.y, 1)] for u in st.threats()], default=float)}
        self.actions.append(row)
        self.log(f"  {st.t:7.1f}s  {act.card:16s} -> ({act.x:4.1f}, {act.y:4.1f})  {act.reason}  elixir {st.elixir:.1f}")

    def _finish(self, game: dict, t: float) -> None:
        if game.get("rec") is not None:
            game["rec"].send_signal(2)  # SIGINT: scrcpy finalises the file
            try:
                game["rec"].wait(timeout=15)
            except Exception:  # noqa: BLE001
                game["rec"].kill()
        rows = [a for a in self.actions if a["game_id"] == game["id"]]
        self._save("bot_actions", rows)
        g = {"game_id": game["id"], "t_start": game["t0"], "t_end": t, "n_actions": game["n"],
             "deck": json.dumps(game["reader"].deck, default=int), "opp_plays": len(game["opp"].plays)}
        self._save("bot_games", [g])
        self.games.append(g)
        if len(game["reader"].deck) == 8:
            self.known_deck = list(game["reader"].deck)
        self.log(f"game {game['id']} over: {game['n']} plays by the bot, {len(game['opp'].plays)} opponent plays seen")
