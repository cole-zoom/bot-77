"""Step 5 of the input rebuild (2026-11-16): the pro imitation shards rebuilt from live-path board reads, with troop HP
and the new input block (bot77.live.extra_obs).

  uv run --no-sync python scripts/build_bc_live.py --reads /Volumes/LaCie/bot-77-data/pod_job/reads \
      --out /Volumes/LaCie/bot-77-data/bc/shards_t02 [--workers 6] [--games N] [--opp live|truth]

As scripts/build_bc_dataset.py (shards_t01: same matches, both players, a decision every 0.1 s, the same labels and
weights; hands, our elixir and the clock from hud_states), except what live play reads differently:
  units      the live detector's reads (scripts/read_replay_boards.py: v6l through LiveReader.read_board, fast mode,
             so the last detector run at or before t, up to ~0.3 s old), with their HP bars (unit_hp); the top player's
             board mirrored (teams swapped, x -> 18 - x, the feet mirrored as build_bc_dataset mirrors them)
  opponent   OppInputs per player (PlayTracker + SpellTracker) over those reads, the player's own plays (from
             `plays`, side top for the top player) fed as live play feeds the bot's taps; its plays drive the cycle
             blocks (real ages) and, with --opp live (default), the opponent's deck so far and elixir as live play
             has them (their counted cards; 5 + regeneration - their counted plays); --opp truth keeps their full
             deck and HUD elixir bar as shards_t01 had
  block      OppInputs.block(t) after the 13,977 floats (437)
  towers     the replays' OCR'd tower HP (tower_frac) as before: the live tower reader needs bright frames and the
             replays' are dimmed (bot77.readers.tower_hp)
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party" / "clash-royale-env"))

import build_bc_dataset as bc  # noqa: E402
from bot77.features.frames import FEET_DY, FLIP_Y  # noqa: E402
from bot77.live.extra_obs import OppInputs  # noqa: E402
from bot77.live.state import Unit  # noqa: E402
from bot77.sim import cards as sc  # noqa: E402
from bot77.sim.mirror import mirror, observation  # noqa: E402
from bot77.sim.scenario import REAL_ELIXIR_S, real_multiplier  # noqa: E402
from bot77.sim.validate import TOWER_KEYS, load_game, tower_frac  # noqa: E402

STEP_S = 0.1
_C: dict = {}


def load_reads(path: Path, names: list[str]) -> tuple[np.ndarray, list[list[Unit]]]:
    """(run times, units of each run) from a read_replay_boards cache."""
    z = np.load(path)
    classes = json.loads(str(z["meta"]))["classes"]
    runs = z["run_t"].astype(np.float64)
    out = [[] for _ in runs]
    k = np.searchsorted(runs, z["det_t"].astype(np.float64) - 1e-4)
    team = {0: "friendly", 1: "enemy"}
    for i in range(len(z["det_t"])):
        c = int(z["det_card"][i])
        hp = float(z["det_hp"][i])
        out[k[i]].append(Unit(names[c] if c >= 0 else None, classes[int(z["det_cls"][i])],
                              team.get(int(z["det_team"][i]), "other"), float(z["det_xy"][i, 0]),
                              float(z["det_xy"][i, 1]), float(z["det_conf"][i]), tuple(z["det_box"][i]),
                              None if np.isnan(hp) else hp))
    return runs, out


def flip(units: list[Unit]) -> list[Unit]:
    """The bottom player's read -> the top player's view: teams swapped, the board turned (feet mirrored as
    build_bc_dataset.to_side mirrors a ground point)."""
    sw = {"friendly": "enemy", "enemy": "friendly"}
    out = []
    for u in units:
        feet = FLIP_Y + 2 * FEET_DY - (u.y + FEET_DY)
        out.append(Unit(u.card, u.cls, sw.get(u.team, u.team), 18.0 - u.x, feet - FEET_DY, u.conf, None, u.hp_frac))
    return out


def mirror_units(units: list[Unit]) -> list[dict]:
    """As SelfPlayPolicy._mirror hands units to mirror()."""
    return [{"card": u.card, "cls": u.cls, "team": "own" if u.team == "friendly" else "opp", "x": u.x,
             "y": u.y + FEET_DY, **({"hp_frac": u.hp_frac} if u.hp_frac is not None else {})}
            for u in units if u.card and u.team in ("friendly", "enemy") and "tower" not in u.cls and "bar" not in u.cls]


def build(job) -> tuple[str, int]:
    try:
        return _build(job)
    except Exception as e:  # noqa: BLE001 -- one bad game must not stop the run
        import traceback
        return f"{job[0]['match_id']}: failed {e!r} {traceback.format_exc()[-400:]}", 0


def _build(job) -> tuple[str, int]:
    m, top_id, split, results, reads, out_dir, opp_el, cards = job
    names = [c["name"] for c in cards]
    g = load_game(m["match_id"], split, m["source"])
    runs, run_units = load_reads(Path(reads), names)
    huds = {"bottom": bc.hud_series(m["match_id"]), "top": bc.hud_series(top_id)}
    id_name = {c["id"]: c["name"] for c in cards}
    n_rows, notes = 0, []
    for side in ("bottom", "top"):
        opp = "top" if side == "bottom" else "bottom"
        out = Path(out_dir) / f"{m['match_id']}_{side}.npz"
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
        deck_names = [sc.engine_cards()[e]["name"] for e in g.decks[side] if e in sc.engine_cards()]
        oi = OppInputs(cards, deck_names)
        t, t_end = known[0]["t_video"], (m["t_end"] or hud[-1]["t_video"])
        my_plays = [p for p in g.plays if p["side"] == side and p["kind"] == "play"]
        own_q = sorted(my_plays, key=lambda p: p["t"])
        k_own, k_run = 0, -1
        # the reads before the first decision feed the tracker too (live, it runs from the first battle frame)
        t_feed = max(float(m["t_start"]), runs[0] if len(runs) else t)
        obs, blocks, card, cell, tt = [], [], [], [], []
        hand_prev = None
        t_feed = round(t_feed, 1)
        while t < t_end:
            # feed OppInputs every 0.1 s up to t, as live play reads every frame
            while t_feed <= t + 1e-6:
                while k_own < len(own_q) and own_q[k_own]["t"] <= t_feed:
                    p = own_q[k_own]
                    k_own += 1
                    e = p["engine_card"]
                    if e is None or e not in sc.engine_cards():
                        continue
                    xy = p.get("spot") or ((p["x"], p["y"]) if p["x"] is not None else None)
                    x, y = bc.to_side(xy[0], xy[1], side, sc.engine_cards()[e]["is_spell"]) if xy else (None, None)
                    oi.own_play(sc.engine_cards()[e]["name"], p["t"], x, y)
                k_run = int(np.searchsorted(runs, t_feed + 1e-6)) - 1
                if k_run >= 0:
                    us = run_units[k_run] if side == "bottom" else flip(run_units[k_run])
                    h0 = bc.at(ts, hud, t_feed)
                    oi.update_units(t_feed, us, h0["t_match"] if h0 else None)
                t_feed = round(t_feed + STEP_S, 1)
            h = bc.at(ts, hud, t)
            hand = [own_ids.get(c) if c >= 0 else None for c in h["hand"]]
            if None in hand and hand_prev:
                hand = [a if a is not None else b for a, b in zip(hand, hand_prev)]
            if None in hand:
                t += STEP_S
                continue
            hand_prev = hand
            oh = bc.at(ots, ohud, t)
            us = [] if k_run < 0 else (run_units[k_run] if side == "bottom" else flip(run_units[k_run]))
            towers = {}
            for key in TOWER_KEYS:
                f = tower_frac(g, key if side == "bottom" else bc.SWAP[key], t)
                if f is not None:
                    towers[key] = f
            plays = [(sc.engine_id(p["card"]), p) for p in oi.plays]
            opp_played = [(e, max(0.0, t - p["t"])) for e, p in plays if e is not None]
            opp_deck = list(dict.fromkeys(e for e, _ in plays if e is not None))[:8]
            if opp_el == "live":
                regen = sum(real_multiplier(s) for s in range(int(h["t_match"]))) / REAL_ELIXIR_S
                el_opp = 5.0 + regen - sum(sc.cost(e) for e, _ in plays if e is not None)
            else:
                el_opp = oh["elixir"] if oh else 5.0
            sim = mirror(g.decks[side], opp_deck if opp_el == "live" else g.decks[opp], h["t_match"], hand,
                         h["elixir"], el_opp, mirror_units(us), towers, opp_played, unit_hp=True)
            if sim is None:
                t += STEP_S
                continue
            c, ce = 4, -1
            p = next((p for p in my_plays if t <= p["t"] < t + STEP_S), None)
            if p is not None:
                e = p["engine_card"]
                c = hand.index(e) if e in hand else -1
                xy = p.get("spot") or ((p["x"], p["y"]) if p["x"] is not None else None)
                if xy is not None and e is not None:
                    spell = sc.engine_cards()[e]["is_spell"]
                    gx, gy = (xy[0], xy[1]) if spell else (xy[0], xy[1] + FEET_DY)
                    ex, ey = bc.ours_to_engine(*bc.to_side(gx, gy, side, spell))
                    ce = bc.cell_from_xy(ex, ey)
            obs.append(observation(sim).astype(np.float16))
            blocks.append(oi.block(t).astype(np.float16))
            card.append(c)
            cell.append(ce)
            tt.append(t)
            t += STEP_S
        if not obs:
            continue
        res = results.get(side)
        w = bc.WIN_WEIGHT if res == "win" else 1.0
        np.savez_compressed(out, obs=np.concatenate([np.stack(obs), np.stack(blocks)], 1),
                            card=np.array(card, np.int64), cell=np.array(cell, np.int64), t=np.array(tt, np.float32),
                            weight=np.full(len(obs), w, np.float32), split=np.array(split), side=np.array(side),
                            result=np.array(res or ""),
                            opp_plays=np.array(json.dumps([{"t": round(q["t"], 2), "card": q["card"]} for q in oi.plays])),
                            rebuilt=np.array(f"live path 2026-11-16: reads {Path(reads).name}, opponent {opp_el}"))
        n_rows += len(obs)
        notes.append(f"{side} {len(obs)} rows, their plays counted {len(oi.plays)}")
    return f"{m['match_id']}: {n_rows} rows ({'; '.join(notes)})", n_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cards", default=None, help="the cards.json the reads were made with (default: next to reads)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--matches", nargs="*")
    ap.add_argument("--opp", choices=["live", "truth"], default="live")
    a = ap.parse_args()
    from bot77.sim import store
    cards = json.loads(Path(a.cards or Path(a.reads).parent / "cards.json").read_text())
    Path(a.out).mkdir(parents=True, exist_ok=True)
    ms = store.read("matches", "source IN ('replay_tv', 'replay_bl')")
    by = {m["match_id"]: m for m in ms}
    jobs = []
    for m in sorted(ms, key=lambda m: m["match_id"]):
        if m["pov_side"] != "bottom" or m["split"] not in ("train", "dev", "heldback"):
            continue
        top = by.get(m["pair_match_id"])
        reads = Path(a.reads) / f"{m['match_id']}.npz"
        if not top or not reads.exists() or (a.matches and m["match_id"] not in a.matches):
            continue
        split = "heldback" if m["split"] == "heldback" else "train"
        jobs.append((m, top["match_id"], split, {"bottom": m["result"], "top": top["result"]}, str(reads), a.out,
                     a.opp, cards))
    if a.games:
        jobs = jobs[: a.games]
    t0, total = time.time(), 0
    with Pool(a.workers) as pool:
        for i, (msg, n) in enumerate(pool.imap_unordered(build, jobs)):
            total += n
            print(f"{i + 1}/{len(jobs)} {msg} ({total} rows, {time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
