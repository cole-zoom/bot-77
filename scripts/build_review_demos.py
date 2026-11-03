"""Cole's review answers as imitation demos (school v5, 2026-11-03; Cole: "i feel like ive reviewed enough games to
build the school v5").

  uv run --no-sync python scripts/build_review_demos.py [--out data/cache/bc/review_v1]

One episode per answer, on the bot's own game read from its Battle Log replay (source replay_bot; the bottom player is
the bot), mirrored into the engine exactly as the pro data is (scripts/build_bc_dataset.py), a decision every 0.1 s.
The bot's own plays are never labels: every row is -1 (no loss) except the answer's, so the episode only gives the
LSTM the game so far. Episodes run from CONTEXT_S before the answer to just after it (under one 25.6 s training
window, so the window always starts at the episode's first step).

Answers, from data/reviews/cole/:
  loss_review_*.json   "mark this moment" notes with a "should": wait or a card. Anchor: the bot's play nearest the
                       note (within PLAY_NEAR_S; the decision Cole is pointing at), else the note's own time.
  round1_moments_review_*.json  yes (a mistake) with a "should", as above; no (Cole: the bot's play was right): the
                       bot's play itself, card and cell, is the label. Unanswered or yes without a "should": nothing.
Labels: wait on the ticks in [anchor - WAIT_S, anchor] (the bot played on the last of them, so each is a moment it
could have); a card on [anchor - CARD_S, anchor] where it's in the hand (an unaffordable one is dropped by the
trainer's mask, as Cole's "we should have defended, but we have no elixir"). Its cell: for Arrows / Fireball the
spot that wins the most elixir on the board then (scripts/build_frame_states.py best_spell), for a troop unknown (-1,
the card loss only), for a confirmed play the bot's own spot.
Game time -> replay time: + the replay's offset (load_game). Pairing: the miner's (moments.json) for 10-31, else the
review page's match by opponent name (build_loss_review.match_replays).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_bc_dataset as bc  # noqa: E402
import build_frame_states as bfs  # noqa: E402
import build_loss_review as blr  # noqa: E402

from bot77.sim import cards as sc  # noqa: E402
from bot77.sim.mirror import mirror, observation  # noqa: E402
from bot77.sim.validate import TOWER_KEYS, _frame, load_game, tower_frac  # noqa: E402

OUT = ROOT / "data/cache/bc/review_v1"
STEP_S, CONTEXT_S = 0.1, 24.0
PLAY_NEAR_S = 4.0
WAIT_S, CARD_S = 1.0, 0.3
WEIGHT = 3.0  # as Cole's situation demos
RUNS = {"2026-10-31": ["run40v4", "run40v4b", "run40v4c"], "2026-11-02": ["run20cyc2"]}
MINED = ROOT / "data/reviews/mined_v2_2026-10-31/moments.json"


def pairing() -> dict[str, str]:
    out = {m["game"]: m["replay"] for m in json.loads(MINED.read_text())["moments"] if m.get("replay")}
    for date, tags in RUNS.items():
        games = blr.parse_runs(tags)
        blr.match_replays(games, blr.replay_texts(date))
        for g in games:
            mp4 = g.get("_replay")
            if mp4 is not None and g["gid"] not in out:
                out[g["gid"]] = f"bl_{date}_r{mp4.stem.split('_')[1]}_m01"
    return out


def answers() -> list[dict]:
    out = []
    for f in sorted((ROOT / "data/reviews/cole").glob("loss_review_*.json")):
        for g in json.loads(f.read_text())["games"]:
            for m in g.get("moments", []):
                out.append({"game": g["gid"], "t": m["t"], "should": (m.get("should") or "").strip(), "ok": False,
                            "src": f.name, "note": m.get("note") or ""})
    for f in sorted((ROOT / "data/reviews/cole").glob("round1_moments_review_*.json")):
        for m in json.loads(f.read_text())["moments"]:
            if m.get("v") == "no":
                out.append({"game": m["game"], "t": m["t"], "should": "", "ok": True, "src": f.name,
                            "note": m.get("note") or ""})
            elif m.get("v") == "yes":
                out.append({"game": m["game"], "t": m["t"], "should": (m.get("should") or "").strip(), "ok": False,
                            "src": f.name, "note": m.get("note") or ""})
    return out


_GAMES: dict = {}


def game(rid: str):
    if rid not in _GAMES:
        g = load_game(rid, "bot", "replay_bot")
        _GAMES[rid] = (g, bc.hud_series(rid), bc.hud_series(rid.replace("_m01", "_top_m01")))
    return _GAMES[rid]


def episode(a: dict, rid: str) -> tuple[dict | None, str]:
    g, (ts, hud), (ots, ohud) = game(rid)
    if not ts:
        return None, "no hand reads"
    own_ids = g.card_map["bottom"]
    my = [p for p in g.plays if p["side"] == "bottom" and p["kind"] == "play"]
    op = [p for p in g.plays if p["side"] == "top" and p["kind"] == "play"]
    tv_note = a["t"] + g.offset
    near = min((p for p in my if abs(p["t"] - tv_note) <= PLAY_NEAR_S), key=lambda p: abs(p["t"] - tv_note), default=None)
    if a["ok"] and near is None:
        return None, "no bot play near the confirmed moment"
    anchor = near["t"] if near is not None else tv_note
    should = None if a["ok"] else (4 if a["should"].lower() == "wait" else sc.engine_id(a["should"]))
    if not a["ok"] and should is None:
        return None, f"should {a['should']!r}: not a card or wait"
    span = WAIT_S if should == 4 else CARD_S
    t = max(ts[0], anchor - CONTEXT_S)
    t = round(t, 1)
    obs, card, cell, tt = [], [], [], []
    hand_prev = None
    while t <= anchor + 0.3 + 1e-9:
        h = bc.at(ts, hud, t)
        hand = [own_ids.get(c) if c >= 0 else None for c in h["hand"]] if h else [None] * 4
        if None in hand and hand_prev:
            hand = [x if x is not None else y for x, y in zip(hand, hand_prev)]
        if None in hand:
            t = round(t + STEP_S, 1)
            continue
        hand_prev = hand
        oh = bc.at(ots, ohud, t)
        units = [dict(u) for u in _frame(g, t) or []]
        towers = {k: f for k in TOWER_KEYS if (f := tower_frac(g, k, t)) is not None}
        opp_played = [(p["engine_card"], t - p["t"]) for p in op if p["t"] < t and p["engine_card"] is not None]
        sim = mirror(g.decks["bottom"], g.decks["top"], h["t_match"], hand, h["elixir"],
                     oh["elixir"] if oh else 5.0, units, towers, opp_played)
        if sim is None:
            t = round(t + STEP_S, 1)
            continue
        c, ce = -1, -1
        if anchor - span - 1e-9 <= t <= anchor + 1e-9:
            if a["ok"]:
                e = near["engine_card"]
                if e in hand and t >= anchor - STEP_S - 1e-9:  # the play's own tick
                    c = hand.index(e)
                    xy = near.get("spot") or ((near["x"], near["y"]) if near["x"] is not None else None)
                    if xy is not None:
                        spell = sc.engine_cards()[e]["is_spell"]
                        ce = bc.cell_from_xy(*bc.ours_to_engine(xy[0], xy[1] if spell else xy[1] + bc.FEET_DY))
            elif should == 4:
                c = 4
            elif should in hand:
                c = hand.index(should)
                if a["should"] in ("Arrows", "Fireball"):
                    _, b = bfs.board(g, t)
                    v, xy = bfs.best_spell(b, a["should"])
                    if xy is not None and v > 0:
                        ce = bc.cell_from_xy(*xy)
        obs.append(observation(sim).astype(np.float16))
        card.append(c)
        cell.append(ce)
        tt.append(t)
        t = round(t + STEP_S, 1)
    n_lab = sum(c >= 0 for c in card)
    if not n_lab:
        return None, "the answer's card was never in the hand there" if should not in (None, 4) else "no labelled tick"
    return {"obs": np.stack(obs), "card": np.array(card, np.int64), "cell": np.array(cell, np.int64),
            "t": np.array(tt, np.float32)}, f"{len(obs)} steps, {n_lab} labelled"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pair = pairing()
    summary = []
    for ans in answers():
        if not ans["ok"] and not ans["should"]:
            summary.append({**ans, "kept": False, "why": "no answer given"})
            continue
        rid = pair.get(ans["game"])
        if rid is None:
            summary.append({**ans, "kept": False, "why": "no replay paired"})
            continue
        try:
            ep, why = episode(ans, rid)
        except Exception as e:  # noqa: BLE001 -- one bad game must not stop the run
            ep, why = None, f"failed {e!r}"
        if ep is not None:
            label = "ok" if ans["ok"] else ans["should"].replace(" ", "").replace(".", "")
            f = out / f"review_{ans['game']}_{ans['t']:.1f}_{label}.npz"
            np.savez_compressed(f, **ep, weight=np.full(len(ep["card"]), WEIGHT, np.float32), result=np.array(0.0),
                                split=np.array("train"), side=np.array("bottom"))
        summary.append({**ans, "replay": rid, "kept": ep is not None, "why": why})
        print(f"{'+' if ep is not None else '-'} {ans['game']} {ans['t']:6.1f} {('ok' if ans['ok'] else ans['should']):15s} "
              f"{rid} {why}", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    kept = [s for s in summary if s["kept"]]
    print(f"{len(kept)} of {len(summary)} answers -> {out}")


if __name__ == "__main__":
    main()
