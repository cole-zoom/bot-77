"""S3: the search on Cole's reviewed plays, against the gate set beforehand (docs/simulator.md).

  uv run python scripts/sim_search.py --run s3_v1 [--calibrate 300]

1. Verdict margin, from the pros (human replays, train split): at --calibrate pro play moments the search also scores
   the pro's actual play; the margin is the 80th percentile of (best - pro's play). Set here, before any reviewed play
   is scored, so the gate isn't tuned on Cole's labels.
2. Cole's reviewed plays (labels review_cole; the blind re-review's verdict and fix where a play has both), on the
   bot's paired replay moment (bot_play_moments): the search also scores the bot's actual play (its card at its tap,
   now). Search verdict: bad when the bot's play scores more than the margin below the best.
   Gate: verdict agreement >= 65% on all reviewed plays; >= 12 of the 15 on which both of Cole's reviews agree;
   Cole's fix (card or wait) in the search's top 2 on >= 50% of his mistakes.
-> LanceDB sim_runs (each moment's per-card best and its top 20 candidates) and a summary on stdout.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from bot77.features.frames import FEET_DY
from bot77.sim import cards as sc
from bot77.sim import store
from bot77.sim.search import Candidate, search
from bot77.sim.spots import recover
from bot77.sim.value import WinModel

BLIND = Path("data/ground_truth/reviews/blind_1_cole.json")
MARGIN_PCT = 80
TIMING = re.compile(r"\b(earlier|later|ago|too late|too early|wait until|when (the|their|it|its)\b.*\b(cross|lock|reach))",
                    re.I)


def tower_fracs(match_id: str, t: float) -> dict:
    """tower_hp fractions at video time t (bottom frame names = the bottom player's own frame)."""
    last = {}
    for r in sorted(store.read("tower_hp", f"match_id = '{match_id}' AND accepted = true AND t <= {t}"),
                    key=lambda r: r["t"]):
        if r["frac"] is not None:
            last[r["tower"]] = r["frac"]
    return last


def name(card) -> str:
    return "wait" if card is None else sc.our_name(card)


def run_one(moment_id: str, model: WinModel, actual: tuple[str, float, float] | None) -> tuple:
    m = store.read("moments", f"moment_id = '{moment_id}'")[0]
    tw = tower_fracs(m["match_id"], m["t"]) if m["side"] == "bottom" else None
    extra = []
    if actual:
        card, x, y = actual
        e = None if card == "wait" else sc.engine_id(card)
        extra = [Candidate(e, x, y, 0.0, note="actual")]
    r = search(moment_id, model, tower_hp=tw, extra=extra)
    act = next((c for c in r.candidates if c.note == "actual"), None)
    return r, act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--calibrate", type=int, default=300)
    args = ap.parse_args()
    model = WinModel.load()
    rows, t0 = [], time.time()

    # 1. margin from the pros
    rng = np.random.default_rng(77)
    pro_games = {m["match_id"] for m in store.read("matches", "source IN ('replay_tv', 'replay_bl') AND split = 'train'")}
    pros = [m for m in store.read("moments", "source IN ('replay_tv', 'replay_bl') AND kind = 'play' AND side = 'bottom'")
            if m["game_id"] in pro_games]
    gaps = []
    for m in rng.permutation(pros)[: args.calibrate * 2]:
        if len(gaps) >= args.calibrate:
            break
        p = store.read("plays", f"game_id = '{m['game_id']}' AND source = 'hand' AND side = 'bottom' "
                       f"AND kind = 'play' AND t >= {m['t'] - 0.05} AND t <= {m['t'] + 0.05}")
        if not p or not p[0]["card"]:
            continue
        xy = recover(m["game_id"], p[0])
        if xy is None:
            continue
        spell = sc.engine_id(p[0]["card"]) is not None and sc.engine_cards()[sc.engine_id(p[0]["card"])]["is_spell"]
        x, y = xy[0], xy[1] if spell else xy[1] + FEET_DY
        try:
            r, act = run_one(m["moment_id"], model, (p[0]["card"], x, y))
        except Exception as e:  # noqa: BLE001  (a moment the scenario can't build: count it, go on)
            print("skip", m["moment_id"], e)
            continue
        if act is None or act.score <= -1:
            continue
        gaps.append(r.best().score - act.score)
    margin = float(np.percentile(gaps, MARGIN_PCT))
    print(f"margin from {len(gaps)} pro plays: {MARGIN_PCT}th percentile of (best - pro) = {margin:.4f} "
          f"(median gap {np.median(gaps):.4f}); {time.time() - t0:.0f} s", flush=True)

    # 2. Cole's reviewed plays
    labels = store.read("labels", "source = 'review_cole'")
    bots = {r["moment_id"]: r for r in store.read("labels", "source = 'bot'")}
    blind = {(p["game"], p["k"]): p for p in json.loads(BLIND.read_text())["plays"]}
    bpm = {(r["game_id"], r["k"]): r for r in store.read("bot_play_moments")}
    results = []
    for lab in labels:
        bm = store.read("moments", f"moment_id = '{lab['moment_id']}'")[0]
        key = (bm["game_id"], bm["decision"])
        link = bpm.get(key)
        b = blind.get(key)
        verdict = b["verdict"] if b else lab["verdict"]
        fix = (("wait" if b["card"].startswith("wait") else b["card"]) if b else lab["action"]) if verdict == "bad" else None
        first_verdict = lab["verdict"]
        if link is None or not store.read("moments", f"moment_id = '{link['replay_moment_id']}'"):
            results.append({"id": f"{key[0]}:{key[1]}", "status": "no replay moment"})
            continue
        act = bots.get(lab["moment_id"])
        actual = (act["action"], act["x"], act["y"]) if act and act["x"] is not None else None
        try:
            r, a = run_one(link["replay_moment_id"], model, actual)
        except Exception as e:  # noqa: BLE001
            results.append({"id": f"{key[0]}:{key[1]}", "status": f"scenario failed: {e}"})
            continue
        best = r.best()
        per = sorted(r.best_per_card().items(), key=lambda kv: -kv[1].score)
        top2 = [name(c) for c, _ in per[:2]]
        gap = best.score - a.score if a is not None and a.score > -1 else None
        results.append({"id": f"{key[0]}:{key[1]}", "status": "ok", "cole": verdict, "first": first_verdict,
                        "blind": b is not None, "fix": fix, "note": (b or lab)["note"] or "",
                        "gap": gap, "search": None if gap is None else ("bad" if gap > margin else "ok"),
                        "top2": top2, "best": (name(best.card), best.x, best.y, best.delay), "p0": r.p0,
                        "bot_card": actual[0] if actual else None})
        for c in sorted(r.candidates, key=lambda c: -c.score)[:20] + [v for _, v in per] + ([a] if a else []):
            rows.append({"run": args.run, "moment_id": link["replay_moment_id"], "review_id": f"{key[0]}:{key[1]}",
                         "card": name(c.card), "x": c.x, "y": c.y, "delay": c.delay, "score": c.score,
                         "p0": r.p0, "p_end": c.p_end, "note": c.note, "margin": margin})
        print(f"{results[-1]['id']}: Cole {verdict} (fix {fix}), search {results[-1]['search']} "
              f"(gap {gap if gap is None else round(gap, 4)}), top2 {top2}", flush=True)
    commit = store.engine_commit()
    for r in rows:
        r["engine"] = commit
    store.write("sim_runs", rows, where=f"run = '{args.run}'")
    Path(f"data/cache/sim_search_{args.run}.json").write_text(json.dumps({"margin": margin, "gaps": gaps,
                                                                          "results": results}, default=str))

    ok = [r for r in results if r["status"] == "ok" and r["search"] is not None]
    agree = [r["search"] == r["cole"] for r in ok]
    clean = [r for r in ok if r["blind"] and r["first"] == r["cole"]]
    mistakes = [r for r in ok if r["cole"] == "bad" and r["fix"]]
    top2 = [r["fix"] in r["top2"] for r in mistakes]
    print(f"\n{len(results)} reviewed plays, {len(ok)} searched ({len(results) - len(ok)} not: "
          f"{sorted({r['status'] for r in results if r['status'] != 'ok'})})")
    print(f"GATE verdict agreement: {np.mean(agree):.1%} ({sum(agree)}/{len(agree)}); gate >= 65%")
    print(f"GATE clean set (both reviews agree): {sum(r['search'] == r['cole'] for r in clean)}/{len(clean)}; gate >= 12/15")
    print(f"GATE Cole's fix in top 2: {np.mean(top2):.1%} ({sum(top2)}/{len(top2)}); gate >= 50%")
    timing = [r for r in mistakes if TIMING.search(r["note"])]
    later = [r for r in timing if re.search(r"later|wait until|when", r["note"], re.I)]
    print(f"timing notes: {len(timing)} (of which 'later' / 'when it crosses': {len(later)}; search's best waits or "
          f"delays on {sum(r['best'][0] == 'wait' or r['best'][3] > 0 for r in later)})")


if __name__ == "__main__":
    main()
