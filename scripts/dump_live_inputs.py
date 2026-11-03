"""What the screen reader hands the model, decision by decision (2026-11-03; Cole: "make a file with the exact output
the detector gives to the decision maker").

  uv run --no-sync python scripts/dump_live_inputs.py bot_2026-11-03_113837 [--ckpt X] [--start 0 --seconds 999]
      [--every 5]

Runs the live stack exactly as live_play.py --policy selfplay --decision-s 0.1 builds it (LiveReader with the adopted
detector and our 8-card deck in fast mode, OpponentTracker, SelfPlayPolicy) over the game's own recording at 10 fps,
and at every decision records the four hand-offs:

  1. screen_read   the LiveState: clock, elixir, hand (after the cycle tracker) and what the slots read, Next, the
                   queue, every detected unit (card, class, team, x / y in arena tiles, confidence)
  2. opp_plays     the opponent's plays the tracker has counted so far
  3. mirror_args   the exact arguments to bot77.sim.mirror (the screen read -> an engine)
  4. observation   the vector the net reads (13,977 floats), decoded: the 21 board channels' non-zero cells, the
                   hand, the 10 scalars, the opponent card-cycle blocks; the raw vector is in obs.npz
  5. decision      the net's chances for the 4 slots and wait, and the choice

Out: data/reviews/live_inputs/<game>/decisions.jsonl (every decision), obs.npz, and sample.md (one decision every
--every seconds, readable). Offline differences from the live run: frames come from the recording, not the stream (the
same pixels, a frame's timing can differ by ~0.1 s), and no taps are made, so the "tapped slot stays blank" hold
(LiveReader.played) never fires; the hand is what the screen and tracker show. The opponent's plays read "just now"
as in every live run (BOT77_FLAT_OPP_CYCLE=1); --real-cycle for their real ages.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import lancedb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "third_party" / "clash-royale-env"
sys.path.insert(0, str(FORK))

import bot77.live.selfplay as SP  # noqa: E402
from bot77.detect.yolo import Bot77Detector  # noqa: E402
from bot77.layout import Layout  # noqa: E402
from bot77.live.state import LiveReader, OpponentTracker  # noqa: E402
from bot77.models import adopted_path  # noqa: E402
from bot77.sim import cards as sc  # noqa: E402
from bot77.video import iter_frames  # noqa: E402

DECK = "Giant,Battle Ram,Mini P.E.K.K.A,Mega Minion,Bomber,Goblin Hut,Fireball,Arrows"
OUT = ROOT / "data/reviews/live_inputs"
W, H, NCH, NCARD = 18, 34, 21, 185
CHANNELS = ["own melee HP", "own ranged HP", "own building-targeter HP", "own building HP",
            "enemy melee HP", "enemy ranged HP", "enemy building-targeter HP", "enemy building HP", "river / bridges",
            "own units in cell", "enemy units in cell", "own flyer", "enemy flyer", "own hits air", "enemy hits air",
            "own damage", "enemy damage", "own range", "enemy range", "own speed", "enemy speed"]
SCALARS = ["time fraction (of 3600 ticks)", "own elixir spent / 280", "opp elixir spent / 280", "own king HP",
           "own left tower HP", "own right tower HP", "enemy king HP", "enemy left tower HP", "enemy right tower HP",
           "elixir multiplier / 3"]


def ename(e):
    if e is None or e < 0:
        return None
    c = sc.engine_cards().get(e)
    return c["name"] if c else f"#{e}"


def decode(obs: np.ndarray) -> dict:
    sp = obs[: NCH * W * H].reshape(NCH, H, W)
    board = []
    for ch in range(NCH):
        if ch == 8:  # the river never changes: left out
            continue
        ys, xs = np.nonzero(sp[ch])
        for y, x in zip(ys, xs):
            board.append({"channel": ch, "what": CHANNELS[ch], "x": int(x), "y": int(y), "value": round(float(sp[ch, y, x]), 3)})
    i = NCH * W * H
    elixir = float(obs[i]) * 10
    costs = [round(float(v) * 10, 1) for v in obs[i + 1: i + 5]]
    hand = []
    for k in range(4):
        blk = obs[i + 5 + k * NCARD: i + 5 + (k + 1) * NCARD]
        nz = np.nonzero(blk)[0]
        hand.append(ename(int(nz[0])) if len(nz) else None)
    j = i + 5 + 4 * NCARD
    scal = {SCALARS[k]: round(float(obs[j + k]), 4) for k in range(10)}
    cyc = obs[j + 10:]
    seen, rec = cyc[:NCARD], cyc[NCARD:]
    opp_cycle = [{"card": ename(int(c)), "seen": 1, "recency": round(float(rec[c]), 3)} for c in np.nonzero(seen)[0]]
    return {"size": int(obs.size), "elixir": round(elixir, 2), "hand": hand, "hand_costs": costs, "scalars": scal,
            "opp_cycle": opp_cycle, "board_nonzero_cells": board}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--ckpt", default="data/models/bc_school_v5/epoch03.pth")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--every", type=float, default=5.0, help="sample.md: one decision every this many seconds")
    ap.add_argument("--real-cycle", action="store_true",
                    help="the opponent's plays with their real ages; by default every play reads 'just now' "
                         "(BOT77_FLAT_OPP_CYCLE=1), as every live run so far")
    a = ap.parse_args()
    if not a.real_cycle:
        os.environ["BOT77_FLAT_OPP_CYCLE"] = "1"
    video = ROOT / "data/raw/live_games" / a.game / "game.mkv"
    out = OUT / a.game
    out.mkdir(parents=True, exist_ok=True)

    cards = lancedb.connect(str(ROOT / "data/lancedb")).open_table("cards").search().where("kind = 'deck_card'") \
        .limit(1000).to_arrow().to_pylist()
    by_id, by_name = {c["id"]: c["name"] for c in cards}, {c["name"]: c["id"] for c in cards}
    deck = [by_name[n] for n in DECK.split(",")]
    det = Bot77Detector(Path(adopted_path("detector_live", "data/models/bot77_detector/v6l/best.pt")))
    reader = LiveReader(Layout.load("bluestacks_live"), cards, det, deck=deck)
    reader.fast = True
    opp = OpponentTracker()
    pol = SP.SelfPlayPolicy(cards, a.ckpt, decision_s=0.1)

    cap = {}
    orig = SP.mirror

    def spy(*args, **kw):
        cap["args"], cap["sim"] = args, orig(*args, **kw)
        return cap["sim"]

    SP.mirror = spy
    names = lambda ids: [by_id.get(c) if c is not None else None for c in ids]  # noqa: E731
    rows, obs_all, n_log = [], [], 0
    with open(out / "decisions.jsonl", "w") as f:
        for t, im in iter_frames(video, 10.0, a.start, a.seconds):
            st = reader.read(im, t)
            if not st.in_battle:
                continue
            opp.update(st)
            cap.clear()
            pol.decide(st, opp.plays)
            if len(pol.log) == n_log or "sim" not in cap or cap["sim"] is None:
                continue
            n_log = len(pol.log)
            d = pol.log[-1]
            obs = np.asarray(cap["sim"].env.get_observation_for_team(0), dtype=np.float32)
            obs_all.append(obs)
            ar = cap["args"]
            row = {
                "video_t": round(t, 1),
                "screen_read": {"clock_elapsed_s": st.elapsed_s, "time_left_s": st.time_left_s, "phase": st.phase,
                                "multiplier": st.multiplier, "elixir": st.elixir, "hand": names(st.hand),
                                "hand_read_by_screen": names(st.hand_read or []), "next": by_id.get(st.next),
                                "queue": names(st.queue),
                                "units": [{"card": u.card, "cls": u.cls, "team": u.team, "x": round(u.x, 2),
                                           "y": round(u.y, 2), "conf": round(u.conf, 2)} for u in st.units]},
                "opp_plays": [{"t": round(p["t"], 1), "card": p["card"]} for p in opp.plays],
                "mirror_args": {"own_deck": [ename(e) for e in ar[0]], "opp_deck": [ename(e) for e in ar[1]],
                                "t_match": ar[2], "hand": [ename(e) for e in ar[3]], "elixir_own": ar[4],
                                "elixir_opp_estimate": round(float(ar[5]), 2),
                                "units": [{**u, "x": round(float(u["x"]), 2), "y": round(float(u["y"]), 2)}
                                          for u in ar[6]], "towers": ar[7],
                                "opp_played": [(ename(e), round(s, 1)) for e, s in ar[8]]},
                "observation": decode(obs),
                "decision": {"p_slots_then_wait": d["p"], "chose": "wait" if d["slot"] == 4 else
                             (names(st.hand)[d["slot"]] or f"slot {d['slot']}"), "value": round(d["value"], 3)},
            }
            f.write(json.dumps(row, default=float) + "\n")
            rows.append(row)
    np.savez_compressed(out / "obs.npz", obs=np.stack(obs_all) if obs_all else np.zeros((0,)),
                        t=np.array([r["video_t"] for r in rows], np.float32))

    md = [f"# What the model sees: {a.game}", "",
          f"{len(rows)} decisions (one every 0.1 s in battle), checkpoint `{a.ckpt}`. Every decision is in "
          "`decisions.jsonl`; the raw 13,977-float vectors are in `obs.npz`. Below: one every "
          f"{a.every:g} s.", ""]
    last = -1e9
    for r in rows:
        if r["video_t"] - last < a.every:
            continue
        last = r["video_t"]
        s, o, m, dec = r["screen_read"], r["observation"], r["mirror_args"], r["decision"]
        md += [f"## {r['video_t']:.1f} s into the video, game clock {s['clock_elapsed_s']} s", "",
               "**1. Screen read (detector + readers -> LiveState)**", "",
               f"- elixir {s['elixir']}, hand {s['hand']} (the slots read {s['hand_read_by_screen']}), next "
               f"{s['next']}, queue {s['queue']}",
               f"- {len(s['units'])} units:", ""]
        md += [f"  - {u['team']:8s} {str(u['card']):18s} {u['cls']:22s} x {u['x']:5.1f} y {u['y']:5.1f} conf {u['conf']}"
               for u in s["units"]]
        md += ["", f"**2. Opponent plays counted so far**: {[(p['t'], p['card']) for p in r['opp_plays']]}", "",
               "**3. Into the engine (mirror)**", "",
               f"- our deck {m['own_deck']}; their deck so far {m['opp_deck']}",
               f"- hand {m['hand']}, our elixir {m['elixir_own']}, their elixir (estimate) {m['elixir_opp_estimate']}",
               f"- towers {m['towers'] or 'all full (not read live)'}; opp played (card, s ago) {m['opp_played']}",
               f"- units: {m['units']}", "",
               "**4. The vector the net reads** (decoded)", "",
               f"- elixir {o['elixir']}, hand {o['hand']} costing {o['hand_costs']}",
               f"- scalars: {o['scalars']}",
               f"- opponent cycle: {[(c['card'], c['recency']) for c in o['opp_cycle']]}",
               f"- board: {len(o['board_nonzero_cells'])} non-zero cells:", ""]
        md += [f"  - ({c['x']:2d},{c['y']:2d}) {c['what']}: {c['value']}" for c in o["board_nonzero_cells"]
               if c["channel"] < 8]
        md += ["  - (attribute channels 9-20 for the same cells are in decisions.jsonl)", "",
               f"**5. Decision**: chances (4 slots, wait) {dec['p_slots_then_wait']} -> **{dec['chose']}**", ""]
    (out / "sample.md").write_text("\n".join(md))
    print(f"{len(rows)} decisions -> {out}")


if __name__ == "__main__":
    main()
