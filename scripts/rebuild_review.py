"""Step 5 of the input rebuild (2026-11-16): Cole's review answers (data/cache/bc/review_v1, built from the Battle Log
replays of the bot's games) rebuilt from the bot's own screen recordings through the live path, with tower / troop
HP and the new input block.

  uv run --no-sync python scripts/rebuild_review.py [--games bot_2026-10-31_013343 ...] [--layout bluestacks_live]
  in:  data/cache/bc/review_v1/review_<game>_<t>_<label>.npz (+ summary.json: the replay each answer was read from),
       data/raw/live_games/<game>/game.mkv, bot_actions (the bot's taps)
  out: data/cache/bc/review_v2/  same files, rows, labels and weights; obs 13,977 (live) + 437 (block)

Per game: LiveReader (fast, our deck, BOT77_LIVE_HP) over game.mkv at 10 fps from its start, as
eval_live_tracker.run_game; OppInputs takes every read; each bot tap (bot_actions, t - the game's t_start = video
time) goes to OppInputs.own_play(card, x, y) and LiveReader.played(slot), as live play does. The observation is built
as SelfPlayPolicy._mirror builds it (PlayTracker's plays for the cycle blocks). Each old row takes the frame at replay t
+ the game's shift: the one (-20..+5 s) at which the old rows' hands (read off the replay's HUD)
agree most with the recording's (load_game's offset and the game clock both sat seconds off).
Checked per file: the time gap to the nearest frame, and whether a labelled row's slot holds the same card in the new
hand as in the old.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "clash-royale-env"))

from bot77.detect.yolo import Bot77Detector  # noqa: E402
from bot77.layout import Layout  # noqa: E402
from bot77.live.extra_obs import OppInputs  # noqa: E402
from bot77.live.rebuild import DECK, HAND0, NCARD, SCAL0, LiveObs, cards as load_cards, diff_sections  # noqa: E402
from bot77.live.state import LiveReader  # noqa: E402
from bot77.models import adopted_path  # noqa: E402
from bot77.video import iter_frames  # noqa: E402

CLOCK_S = 360.0  # the time scalar is a fraction of 3600 engine ticks (10 a second)
SRC, OUT = ROOT / "data/cache/bc/review_v1", ROOT / "data/cache/bc/review_v2"


def hand_of(row: np.ndarray, slot: int) -> int | None:
    nz = np.nonzero(row[HAND0 + 5 + slot * NCARD: HAND0 + 5 + (slot + 1) * NCARD])[0]
    return int(nz[0]) if len(nz) else None


def best_shift(zs: dict, ft: np.ndarray, fh: np.ndarray) -> float:
    """video t - replay t for one game: the shift (0.1 s steps over -20..+5 s) at which the old rows' hands (read off
    the replay's HUD) agree most with the recording's (slot by slot). The replay offset and the game clocks both sat
    seconds off (first game: the clock-aligned rows put the bot's play ~1 s early)."""
    old_t = np.concatenate([z["t"] for z in zs.values()]).astype(np.float64)
    old_h = np.concatenate([[[hand_of(r, k) or -2 for k in range(4)] for r in z["obs"]] for z in zs.values()])
    best = (-1.0, 0.0)
    for s in np.arange(-20.0, 5.0001, 0.1):
        i = np.clip(np.searchsorted(ft, old_t + s), 0, len(ft) - 1)
        score = float((fh[i] == old_h).mean())
        best = max(best, (score, round(float(s), 1)))
    return best[1]


def taps(db, game: str) -> list[dict]:
    t0 = db.open_table("bot_games").search().where(f"game_id = '{game}'").limit(1).to_arrow().to_pylist()[0]["t_start"]
    rows = db.open_table("bot_actions").search().where(f"game_id = '{game}'").limit(10**5).to_arrow().to_pylist()
    return sorted(({"t": r["t"] - t0, "card": r["card"], "slot": r["slot"], "x": r["x_tiles"], "y": r["y_tiles"]}
                   for r in rows), key=lambda r: r["t"])


def run_game(game: str, t_max: float, cards: list[dict], det, layout: str, own: list[dict]) -> tuple[list, dict]:
    by_name = {c["name"]: c["id"] for c in cards}
    reader = LiveReader(Layout.load(layout), cards, det, deck=[by_name[n] for n in DECK.split(",")])
    reader.fast = True
    reader.read_hp = True
    opp = OppInputs(cards, DECK.split(","))
    lo = LiveObs(cards)
    out, k = [], 0
    for t, im in iter_frames(ROOT / "data/raw/live_games" / game / "game.mkv", 10.0, 0.0, t_max + 1.0):
        while k < len(own) and own[k]["t"] <= t:  # the bot's tap, before the next read (live: tap, then read on)
            p = own[k]
            opp.own_play(p["card"], p["t"], p["x"], p["y"])
            reader.played(p["slot"], p["t"])
            k += 1
        st = reader.read(im, t)
        if not st.in_battle:
            continue
        opp.update(st)
        o = lo.obs(st, opp.plays)
        if o is not None:
            out.append((t, o, opp.block(t).astype(np.float16)))
    # the cycle tracker's own plays vs the taps: how far the video clock sits from t - t_start
    ev = [e.t for e in reader.tracker.events if e.kind == "play"] if reader.tracker else []
    lag = [min((e - p["t"] for e in ev if 0 <= e - p["t"] < 4), default=None) for p in own if p["t"] < t_max]
    lag = [x for x in lag if x is not None]
    return out, {"frames": len(out), "opp_plays": len(opp.plays), "taps": len(own),
                 "tap_to_cycle_play_s_median": round(float(np.median(lag)), 2) if lag else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*")
    ap.add_argument("--layout", default="bluestacks_live", help="the 10-31 / 11-02 games ran on BlueStacks")
    a = ap.parse_args()
    import lancedb
    from bot77.sim.validate import load_game
    db = lancedb.connect(str(ROOT / "data/lancedb"))
    cards = load_cards()
    det = Bot77Detector(Path(adopted_path("detector_live", "data/models/bot77_detector/v6l/best.pt")))
    summ = json.loads((SRC / "summary.json").read_text())
    replay = {(s["game"], f"{s['t']:.1f}"): s["replay"] for s in summ if s.get("kept")}
    files = defaultdict(list)
    for f in sorted(SRC.glob("review_*.npz")):
        game = "_".join(f.stem.split("_")[1:4])
        files[game].append(f)
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for game, fs in sorted(files.items()):
        if a.games and game not in a.games:
            continue
        t0 = time.time()
        zs = {f: dict(np.load(f, allow_pickle=True)) for f in fs}
        rid = replay.get((game, fs[0].stem.split("_")[4]))
        off = load_game(rid, "bot", "replay_bot").offset if rid else 0.0
        t_max = max(float(z["t"].max()) for z in zs.values()) + 5.0  # the shift search reaches +5 s
        frames, info = run_game(game, t_max, cards, det, a.layout, taps(db, game))
        ft = np.array([x[0] for x in frames])
        fh = np.array([[hand_of(x[1], k) or -1 for k in range(4)] for x in frames])
        shift = best_shift(zs, ft, fh)
        for f, z in zs.items():
            rid_f = replay.get((game, f.stem.split("_")[4]), rid)
            old = z["obs"]
            idx = [int(np.argmin(np.abs(ft - (float(tr) + shift)))) for tr in z["t"]]
            gap = np.abs(ft[idx] - (z["t"] + shift))
            oc = old[:, SCAL0].astype(np.float64)
            fc = np.array([float(frames[i][1][SCAL0]) for i in idx])
            base = np.stack([frames[i][1] for i in idx])
            blk = np.stack([frames[i][2] for i in idx])
            lab = [r for r in range(len(z["card"])) if 0 <= z["card"][r] < 4]
            same = sum(hand_of(old[r], int(z["card"][r])) == hand_of(base[r], int(z["card"][r])) for r in lab)
            out = {k: v for k, v in z.items() if k != "obs"}
            out["obs"] = np.concatenate([base, blk], 1)
            out["video_t"] = ft[idx].astype(np.float32)
            out["rebuilt"] = np.array(f"live path 2026-11-16: game.mkv 10 fps, LiveReader {a.layout} fast hp, "
                                      f"OppInputs, bot taps; replay {rid_f}, video t = replay t {shift:+.1f} s")
            np.savez_compressed(OUT / f.name, **out)
            r = {"file": f.name, "rows": len(z["t"]), "shift_s": shift, "frame_gap_s_max": round(float(gap.max()), 2), "clock_gap_s_median": round(float(np.median(np.abs(fc - oc))) * CLOCK_S, 2), "hand_slots_same": round(float(np.mean([[hand_of(old[r], k) == hand_of(base[r], k) for k in range(4)] for r in range(len(idx))])), 3),
                 "labelled": len(lab), "label_card_same": same, "base_diff_cells": diff_sections(old, base),
                 "block_nonzero_per_row": round(float((blk != 0).sum(1).mean()), 1),
                 "rows_hand_named": int((blk[:, 3] > 0).sum()), "rows_spell_slot": int((blk[:, 377] > 0).sum()),
                 **info, "s": round(time.time() - t0, 1)}
            report.append(r)
            print(json.dumps(r), flush=True)
    (OUT / "rebuild_summary.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
