"""Battle -> play -> end -> repeat. Safety: a STOP file ends the run at the next check; max games / hours; an
unrecognised screen for too long or repeated screenshot failures stop the bot rather than tapping blindly. Every
action and game goes to LanceDB (bot_actions, bot_games); each game's replay is pulled later by replay_runner.py and
keyed by the usual pipeline, so the bot's games become training data."""
from __future__ import annotations

import json
import signal
import time
from dataclasses import asdict
from pathlib import Path

import lancedb
import pyarrow as pa

import cv2

from bot77.live import ui
from bot77.live.device import END_AGAIN, END_OK_WITH_AGAIN, MATCH, POS, between_games_tap, deploy, crowns, screen_kind, screen_texts, seen, winner_y
from bot77.live.policy import Policy
from bot77.live.state import LiveReader, OpponentTracker

STOP_FILE = Path("data/live/STOP")
UNKNOWN_DIR = Path("data/raw/live_ui/unknown")
GAMES_DIR = Path("data/raw/live_games")  # per game: screen recording + a frame at each play and 2 s after (gitignored)
SHOT_W = 540  # saved frames are halved (review pages, not training)  # screens met between games (gitignored), for new rules


class Runner:
    def __init__(self, dev, layout, cards: list[dict], detector=None, db_dir: Path = Path("data/lancedb"),
                 max_games: int = 1, max_hours: float = 1.0, unknown_limit_s: float = 240.0, step_s: float = 0.1,
                 dry_run: bool = False, record: bool = True, log=print, policy_factory=None, fast_reader: bool = False):
        self.dev, self.layout, self.cards, self.detector = dev, layout, cards, detector
        # a new policy per game: Policy (rules + A) by default; the self-play agent with --policy selfplay
        self.policy_factory = policy_factory or (lambda: Policy(self.cards))
        self.db = lancedb.connect(db_dir) if not dry_run else None
        self.max_games, self.max_hours, self.unknown_limit_s, self.step_s = max_games, max_hours, unknown_limit_s, step_s
        self.dry_run, self.log, self.record = dry_run, log, record
        self.fast_reader = fast_reader
        self.games, self.actions = [], []
        self.last_nudge = self.last_shot = 0.0
        self.home_since = None  # when the home screen was first seen in a row (tap Battle once it has settled)
        self.known_deck: list[int] = []
        self.wins = self.losses = 0
        self.names = {c["id"]: c["name"] for c in cards}  # learned in the first game, reused after (the deck changes rarely)
        self.probe = LiveReader(layout, cards)  # is a battle on screen? (no detector: clock + elixir are enough)

    def _save(self, table: str, rows: list[dict]) -> None:
        if self.db is None or not rows:
            return
        t = pa.Table.from_pylist(rows)
        if table in self.db.table_names(limit=10_000):
            tbl = self.db.open_table(table)
            new = [f.name for f in t.schema if f.name not in tbl.schema.names]
            if new:  # a column added since the table was made (e.g. units, 2026-10-22): older rows get null
                tbl.add_columns({c: "CAST(NULL AS STRING)" for c in new})
            tbl.add(t)
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
            if STOP_FILE.exists() and game is None:  # finish the game in progress first, then stop
                self.log("STOP file found: stopping"); break
            if len(self.games) >= self.max_games and game is None:
                self.log(f"{len(self.games)} games played: done"); break
            if time.time() - start > self.max_hours * 3600 and game is None:
                self.log("time limit: done"); break
            im = self.dev.cap()
            if im is None:
                fails += 1
                if fails == 3 and hasattr(self.dev, "recover"):
                    self.log(ui.c("  screenshots failing: restarting ADB", "yellow"))
                    self.dev.recover()
                if fails >= 8:
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
                    game["result"] = self._result(im)
                    t_end = time.time()  # with the stream the end screen is caught before "Winner!" fades in (2026-10-27)
                    while game["result"] == "?" and time.time() - t_end < 4:
                        nim = self.dev.cap()
                        if nim is not None:
                            im = nim
                            game["result"] = self._result(im)
                    cv2.imwrite(str(GAMES_DIR / game["id"] / "end.jpg"), cv2.resize(im, (SHOT_W, round(im.shape[0] * SHOT_W / im.shape[1]))))
                    self._finish(game, t)
                    game = self.current = None
                # Play Again starts the next game straight away and skips the reward screens (Cole, 2026-10-22); the
                # last game of a run taps OK instead, so the session can pull the replays
                more = len(self.games) < self.max_games and not STOP_FILE.exists()
                again = None
                try:
                    texts = screen_texts(im)
                    again = next(((x, y) for t, x, y in texts if "play again" in t or t == "again"), None)
                    okb = next(((x, y) for t, x, y in texts if t == "ok"), None)
                except Exception:  # noqa: BLE001
                    texts, okb = [], None
                ok = POS.get("end_ok")
                if not again and seen(im, "end_again") >= MATCH:  # the Play Again layout, its text unread (2026-10-25)
                    again = END_AGAIN
                if again and not okb:  # the Play Again layout: OK sits next to it (its text doesn't OCR)
                    h_, w_ = im.shape[:2]
                    okb = (END_OK_WITH_AGAIN[0] * w_, END_OK_WITH_AGAIN[1] * h_)
                tap = again if (more and again) else okb or ((ok[0] + 60, ok[1] + 30) if ok else None)
                if tap and not self.dry_run:
                    self.dev.tap(*tap)
                self.log(ui.c(f"  end screen: {'Play Again' if (more and again) else 'OK'}", "grey"))
                last_known = time.time(); time.sleep(2.0); continue
            if kind != "home":
                self.home_since = None
            if kind == "home" and game is None:
                # with the stream (~6 frames/s) home is seen mid-animation, and a tap then can land on the arena
                # (opened the trophy road, 2026-10-26): wait until it has been home for 1 s
                self.home_since = self.home_since or time.time()
                if time.time() - self.home_since < 1.0:
                    time.sleep(0.2); continue
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
            if game is not None and game["n"] == 0 and not game.get("in_battle") and time.time() - game["wall0"] > 45:
                # Battle was tapped but no battle came (a wrong screen took the tap): back to finding our way
                self.log(ui.c("  no battle 45 s after tapping Battle: back to navigating", "yellow"))
                if game.get("rec") is not None:
                    game["rec"].send_signal(2)
                if getattr(self.dev, "stream", None) is not None:
                    self.dev.stream.record_to(None)
                game = self.current = None
                continue
            if st is not None and st.in_battle:
                game["in_battle"] = True
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
            # at 0.1 s decisions the stream's wait for a new frame paces the battle loop; no extra sleep there
            time.sleep(0.0 if (self.fast_reader and st is not None and st.in_battle) else self.step_s)

    def _new_game(self, t: float) -> dict:
        gid = time.strftime("bot_%Y-%m-%d_%H%M%S")
        self.log(ui.c(f"  … game {gid} starting", "grey"))
        (GAMES_DIR / gid).mkdir(parents=True, exist_ok=True)
        rec = None
        if self.record and not self.dry_run and getattr(self.dev, "stream", None) is not None:
            self.dev.stream.record_to(GAMES_DIR / gid / "game.mkv")  # from the stream itself (no second encoder user)
        elif self.record and not self.dry_run:
            import os, subprocess
            from bot77.live.device import ADB
            rec = subprocess.Popen(["scrcpy", "-s", self.dev.s, "--no-window", "--no-audio", "--no-control", "--max-fps", "30",
                                    "--video-bit-rate", "8M", f"--record={GAMES_DIR / gid / 'game.mkv'}"],
                                   env={**os.environ, "ADB": ADB}, stdout=subprocess.DEVNULL,
                                   stderr=open(GAMES_DIR / gid / "scrcpy.log", "w"),
                                   # under nohup / `&` SIGINT is ignored and children inherit that: restore it, or the
                                   # recorder never stops (2026-10-24)
                                   preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
        reader = LiveReader(self.layout, self.cards, self.detector)
        reader.fast = self.fast_reader  # decisions every 0.1 s: clock / hand / units cached (bot77.live.state)
        if len(self.known_deck) == 8:  # our deck from earlier games: read only those 8 cards from the first frame
            reader.deck, reader.reader = list(self.known_deck), reader.full.restrict(set(self.known_deck))  # (~1 s -> ~0.1 s)
        return {"id": gid, "t0": t, "reader": reader,
                "policy": self.policy_factory(), "opp": OpponentTracker(), "n": 0, "rec": rec, "after": [],
                "num": len(self.games) + 1, "wall0": time.time(), "opp_name": None, "last_status": 0.0, "result": "?"}

    @staticmethod
    def _result(im) -> str:
        """win / loss from the end screen: "Winner!" sits over the winner's side (ours is the bottom half). The side
        mapping is checked against known results (end.jpg is kept per game)."""
        top, ours = crowns(im)  # more crowns won (a tie, e.g. a tiebreak: the "Winner!" lettering decides)
        if top != ours:
            return "win" if ours > top else "loss"
        # "Winner!" sits just above the winner's own banner: ~36% of the height when we (bottom) won, near the top
        # when the opponent did (Cole's screenshot, 2026-10-22). Found by its lettering (winner_y), else by OCR.
        y = winner_y(im)
        if y is not None:
            return "win" if y > 0.25 else "loss"
        try:
            ys = [y for t, x, y in screen_texts(im) if "winner" in t]
        except Exception:  # noqa: BLE001
            return "?"
        if not ys:
            return "?"
        return "win" if ys[0] > 0.25 * im.shape[0] else "loss"

    def _shot(self, game: dict, im, name: str) -> str:
        f = GAMES_DIR / game["id"] / name
        cv2.imwrite(str(f), cv2.resize(im, (SHOT_W, round(im.shape[0] * SHOT_W / im.shape[1]))), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return name

    def _step(self, game: dict, st, im) -> None:
        for k, due in list(game["after"]):  # the frame ~2 s after each play: what came of it
            if st.t >= due:
                self._shot(game, im, f"{k:03d}_after.jpg"); game["after"].remove((k, due))
        game["opp"].update(st)
        if game["opp_name"] is None:  # the opponent's name, once, for the terminal
            try:
                from bot77.readers.clock import ocr_text
                game["opp_name"] = ocr_text(self.layout.crop(im, "opp_name"), 2).strip()[:24]
            except Exception:  # noqa: BLE001
                game["opp_name"] = ""
            self.log(ui.game_start(game["num"], game["opp_name"]))
        if st.t - game["last_status"] >= 10:
            self.log(ui.status_line(st, self.names)); game["last_status"] = st.t
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
               "threats": json.dumps([[u.card, round(u.x, 1), round(u.y, 1)] for u in st.threats()], default=float),
               "model_spot": json.dumps(None if act.model_spot is None else [round(float(v), 2) for v in act.model_spot]),
               # the whole board (2026-10-22: for board features and the placement model): card, class, team, x, y, conf
               "units": json.dumps([[u.card, u.cls, u.team, round(u.x, 2), round(u.y, 2), round(u.conf, 2)] for u in st.units],
                                   default=float)}
        self.actions.append(row)
        self.log(ui.play_line(st, act))

    def _finish(self, game: dict, t: float) -> None:
        if getattr(self.dev, "stream", None) is not None:
            self.dev.stream.record_to(None)
        if game.get("rec") is not None:
            game["rec"].send_signal(2)  # SIGINT: scrcpy finalises the file
            try:
                game["rec"].wait(timeout=15)
            except Exception:  # noqa: BLE001
                game["rec"].kill()
        rows = [a for a in self.actions if a["game_id"] == game["id"]]
        self._save("bot_actions", rows)
        # A v4 in shadow: what it would have done once a second, next to the rules (2026-10-22)
        self._save("bot_shadow", [{"game_id": game["id"], **r} for r in getattr(game["policy"], "shadow", [])])
        # the self-play / imitation agent's every decision (p(wait), the slot it picked, elixir): why it did or didn't play
        log = getattr(game["policy"], "log", None)
        if log:
            (GAMES_DIR / game["id"] / "decisions.json").write_text(json.dumps(log))
        g = {"game_id": game["id"], "t_start": game["t0"], "t_end": t, "n_actions": game["n"], "result": game.get("result", "?"),
             "deck": json.dumps(game["reader"].deck, default=int), "opp_plays": len(game["opp"].plays)}
        self._save("bot_games", [g])
        self.games.append(g)
        self.wins += game.get("result") == "win"; self.losses += game.get("result") == "loss"
        if len(game["reader"].deck) == 8:
            self.known_deck = list(game["reader"].deck)
        self.log(ui.game_end(game["num"], game.get("result", "?"), game["n"], t - game["t0"], self.wins, self.losses))
