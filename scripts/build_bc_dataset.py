"""Imitation data for the self-play network (option 3, 2026-10-24): every human replay, each player, once a second.

  uv run --no-sync python scripts/build_bc_dataset.py [--games N] [--workers 6]
  -> data/cache/bc/shards/<match>_<side>.npz   (gitignored)

Per decision (1 s, the network's cadence): the acting player's view mirrored into an engine (bot77.sim.mirror: their hand
and elixir from the HUD, the opponent's elixir from theirs, units from the detector (v5l_e15), tower HP (tower_hp), the
opponent's plays so far) -> the network's observation (float16); the label is what they did in that second: the hand
slot of the card played (4 = wait) and its board cell (the engine's own binning), -1 where unknown (the card wasn't in
the read hand, or the play has no spot). Weight 2 for the winner's rows, 1 for the loser's (Cole: weight winners up).
Splits from matches.split: train + dev to train on, held-back to check; test / held-out never.
"""
import argparse
import bisect
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from bot77.features.frames import FEET_DY, FLIP_Y, RIVER_Y
from bot77.sim import cards as sc
from bot77.sim import store
from bot77.sim.geometry import ours_to_engine
from bot77.sim.mirror import mirror, observation
from bot77.sim.validate import TOWER_KEYS, _frame, load_game, tower_frac

OUT = Path("data/cache/bc/shards")
STEP_S = 1.0
WIN_WEIGHT = 2.0
SWAP = {"own_king": "opp_king", "own_l": "opp_r", "own_r": "opp_l", "opp_king": "own_king", "opp_l": "own_r",
        "opp_r": "own_l"}  # bottom-frame tower names -> the top player's own frame
BOARD_W, BOARD_H = 18, 34
FORK = Path(__file__).resolve().parents[1] / "third_party" / "clash-royale-env"


def cell_from_xy(x: float, y: float) -> int:
    """The fork's binning (python_ai/trainers/bc_pretrain.py _cell_from_xy): truncation, as the engine bins."""
    col = int(np.clip(int(x), 0, BOARD_W - 1))
    row = int(np.clip(int(y), 0, BOARD_H - 1))
    return row * BOARD_W + col


def to_side(x: float, y: float, side: str, spell: bool) -> tuple[float, float]:
    """A ground point in the bottom frame -> the acting player's frame (feet / spell centres)."""
    if side == "bottom":
        return x, y
    return 18.0 - x, (2 * RIVER_Y if spell else FLIP_Y + 2 * FEET_DY) - y


def hud_series(match_id: str):
    rows = sorted(store.read("hud_states", f"match_id = '{match_id}'"), key=lambda r: r["t_video"])
    return [r["t_video"] for r in rows], rows


def at(ts, rows, t):
    i = bisect.bisect_right(ts, t) - 1
    return rows[i] if i >= 0 else None


def build(args) -> tuple[str, int]:
    m, top_id, split, results = args
    try:
        g = load_game(m["match_id"], split, m["source"])
    except Exception as e:  # noqa: BLE001
        return f"{m['match_id']}: load failed {e!r}", 0
    huds = {"bottom": hud_series(m["match_id"]), "top": hud_series(top_id)}
    n_rows = 0
    for side in ("bottom", "top"):
        opp = "top" if side == "bottom" else "bottom"
        out = OUT / f"{m['match_id']}_{side}.npz"
        if out.exists():
            continue
        ts, hud = huds[side]
        ots, ohud = huds[opp]
        if not ts:
            continue
        own_ids, opp_ids = g.card_map[side], g.card_map[opp]
        known = [r for r in hud if min(r["hand"]) >= 0]
        if not known:
            continue
        t, t_end = known[0]["t_video"], (m["t_end"] or hud[-1]["t_video"])
        my_plays = [p for p in g.plays if p["side"] == side and p["kind"] == "play"]
        op_plays = [p for p in g.plays if p["side"] == opp and p["kind"] == "play"]
        obs, card, cell, tt = [], [], [], []
        hand_prev = None
        while t < t_end:
            h = at(ts, hud, t)
            hand = [own_ids.get(c) if c >= 0 else None for c in h["hand"]]
            if None in hand and hand_prev:
                hand = [a if a is not None else b for a, b in zip(hand, hand_prev)]
            if None in hand:
                t += STEP_S
                continue
            hand_prev = hand
            oh = at(ots, ohud, t)
            units = []
            for u in _frame(g, t) or []:
                x, y = to_side(u["x"], u["y"], side, False)
                team = u["team"] if side == "bottom" else ("opp" if u["team"] == "own" else "own")
                units.append({**u, "x": x, "y": y, "team": team})
            towers = {}
            for key in TOWER_KEYS:
                f = tower_frac(g, key if side == "bottom" else SWAP[key], t)
                if f is not None:
                    towers[key] = f
            opp_played = [p["engine_card"] for p in op_plays if p["t"] < t and p["engine_card"] is not None]
            sim = mirror(g.decks[side], g.decks[opp], h["t_match"], hand, h["elixir"],
                         oh["elixir"] if oh else 5.0, units, towers, opp_played)
            if sim is None:
                t += STEP_S
                continue
            # the label: the first play of this player in [t, t + STEP_S)
            c, ce = 4, -1
            p = next((p for p in my_plays if t <= p["t"] < t + STEP_S), None)
            if p is not None:
                e = p["engine_card"]
                c = hand.index(e) if e in hand else -1
                xy = p.get("spot") or ((p["x"], p["y"]) if p["x"] is not None else None)
                if xy is not None and e is not None:
                    spell = sc.engine_cards()[e]["is_spell"]
                    gx, gy = (xy[0], xy[1]) if spell else (xy[0], xy[1] + FEET_DY)
                    ex, ey = ours_to_engine(*to_side(gx, gy, side, spell))
                    ce = cell_from_xy(ex, ey)
            obs.append(observation(sim).astype(np.float16))
            card.append(c)
            cell.append(ce)
            tt.append(t)
            t += STEP_S
        if not obs:
            continue
        res = results.get(side)
        w = WIN_WEIGHT if res == "win" else 1.0
        np.savez(out, obs=np.stack(obs), card=np.array(card, np.int64), cell=np.array(cell, np.int64),
                 t=np.array(tt, np.float32), weight=np.full(len(obs), w, np.float32),
                 split=np.array(split), side=np.array(side), result=np.array(res or ""))
        n_rows += len(obs)
    return f"{m['match_id']}: {n_rows} rows", n_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ms = store.read("matches", "source IN ('replay_tv', 'replay_bl')")
    by = {m["match_id"]: m for m in ms}
    jobs = []
    for m in sorted(ms, key=lambda m: m["match_id"]):
        if m["pov_side"] != "bottom" or m["split"] not in ("train", "dev", "heldback"):
            continue
        top = by.get(m["pair_match_id"])
        if not top:
            continue
        split = "heldback" if m["split"] == "heldback" else "train"
        jobs.append((m, top["match_id"], split, {"bottom": m["result"], "top": top["result"]}))
    if a.games:
        jobs = jobs[: a.games]
    t0, total = time.time(), 0
    with Pool(a.workers) as pool:
        for i, (msg, n) in enumerate(pool.imap_unordered(build, jobs)):
            total += n
            print(f"{i + 1}/{len(jobs)} {msg} ({total} rows, {time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
