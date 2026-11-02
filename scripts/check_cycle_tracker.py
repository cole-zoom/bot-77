"""Recorded games through the live reader with the cycle tracker, paced like the live bot (the newest frame every
0.1 s), to check the tracker before (and after) a live run (2026-11-02, bot77.live.cycle).

  uv run --no-sync python scripts/check_cycle_tracker.py bot_2026-10-31_015151 [more game ids] [--from-s 0]

Per game: when the tracker started, seconds any slot was unknown (the reader alone vs the tracker), plays the tracker
counted against the bot's own plays (bot_actions, after duplicate taps), and its re-syncs (mismatches). The live
recordings are data/raw/live_games/<id>/game.mkv (or the remuxed mp4 the review pages made).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import lancedb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bot77.layout import Layout  # noqa: E402
from bot77.live.state import LiveReader  # noqa: E402

DECK = ["Giant", "Battle Ram", "Mini P.E.K.K.A", "Mega Minion", "Bomber", "Goblin Hut", "Fireball", "Arrows"]
VIDEOS = [ROOT / "data/reviews/mined_2026-10-31/videos", ROOT / "data/raw/live_games"]


def video_of(gid: str) -> Path | None:
    for p in (VIDEOS[0] / f"{gid}.mp4", VIDEOS[1] / gid / "game.mkv"):
        if p.exists():
            return p
    return None


def check(gid: str, cards: list[dict], deck: list[int], db) -> dict:
    path = video_of(gid)
    if path is None:
        return {"game": gid, "error": "no video"}
    r = LiveReader(Layout.load("bluestacks_live"), cards, deck=deck)
    r.fast = True
    cap = cv2.VideoCapture(str(path))

    def frames():
        while True:
            ok, fr = cap.read()
            if not ok:
                return
            yield cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, fr

    g = frames()
    cur, nxt = next(g, None), None
    nxt = next(g, None)
    t = cur[0]
    unknown_read = unknown_tracked = battle = 0.0
    slow = 0
    while nxt is not None:
        while nxt is not None and nxt[0] <= t:
            cur, nxt = nxt, next(g, None)
        w0 = time.time()
        st = r.read(cur[1], t)
        took = time.time() - w0
        slow += took > 0.2
        step = max(0.1, took)
        if st.elapsed_s is not None:
            battle += step
            if any(c is None for c in (st.hand_read or [])):
                unknown_read += step
            if any(c is None for c in st.hand):
                unknown_tracked += step
        t += step
    tr = r.tracker
    acts = db.open_table("bot_actions").search().where(f"game_id = '{gid}'").select(["t", "card"]).limit(1000).to_list()
    acts.sort(key=lambda a: a["t"])
    taps = [a for i, a in enumerate(acts) if not (i and a["card"] == acts[i - 1]["card"] and a["t"] - acts[i - 1]["t"] < 1.5)]
    start = next((e.t for e in tr.events if e.kind == "start"), None)
    return {"game": gid, "battle_s": round(battle), "started_at_s": round(start, 1) if start else None,
            "unknown_s_reader": round(unknown_read, 1), "unknown_s_tracked": round(unknown_tracked, 1),
            "plays_tracked": sum(e.kind == "play" for e in tr.events), "bot_taps": len(taps),
            "plays_without_elixir_drop": sum(e.kind == "play" and bool(e.note) for e in tr.events),
            "mismatches": tr.mismatches(), "slow_reads": slow,
            "resyncs": [(round(e.t, 1), e.kind, e.old, e.new) for e in tr.events if e.kind in ("override", "next_override")][:6]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    db = lancedb.connect(ROOT / "data/lancedb")
    cards = db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).to_arrow().to_pylist()
    by = {c["name"]: c["id"] for c in cards}
    deck = [by[n] for n in DECK]
    out = []
    for gid in a.games:
        res = check(gid, cards, deck, db)
        out.append(res)
        print(json.dumps(res), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
