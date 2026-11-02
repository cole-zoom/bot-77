"""A review round from frame_states (2026-11-02, docs/frame-states.md): ~100 moments the failure-mode searches flag
(rule and hybrid, scripts/query_modes.py) plus ~50 random plays, for Cole's yes / no on the moment page.

  uv run --no-sync python scripts/build_round.py --out data/reviews/round1_2026-11-02 [--flagged 100 --random 50]
  uv run --no-sync python scripts/build_moment_review.py data/reviews/round1_2026-11-02
  uv run --no-sync python scripts/serve_review.py data/reviews/round1_2026-11-02 --port 8794

Yes = the bot made a mistake in the clip (the number keys say which kind if the guess is wrong), No = it played fine.
The random plays are the "fine" examples the searches lack: 27 noted mistakes and no fine plays can't separate a bad
Giant push from a good one. Moments within MATCH_S of a note Cole already wrote, and blind rows (the reader bug), are
left out. The order is shuffled (fixed seed) so the random plays sit among the flagged ones.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import lancedb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import failure_modes as fm  # noqa: E402
import query_modes as qm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHARE = {"missed_spell": 20, "pointless_push": 20, "spent_counter": 12, "wasted_spell": 12, "too_early": 10,
         "bad_placement": 10, "no_defence": 10, "missed_finish": 6}  # of 100, by how often Cole named the mode
VIDEOS_FROM = ROOT / "data/reviews/mined_2026-10-31"  # remuxed videos and their offsets (scripts/build_moment_review.py)


def describe(r: dict) -> str:
    """One line of what was going on, from the row."""
    side = lambda x: "left" if x < 9 else "right"
    parts = []
    p = r.get("our_play_now")
    if p:
        where = f" ({side(p['x'])}, {'their side' if p['y'] >= 16.5 else 'bridge' if p['y'] >= 12.5 else 'our side'})" \
            if p["x"] is not None else ""
        parts.append(f"we play {p['card']}{where}")
    parts.append(f"elixir {r['our_elixir']:.1f} vs their {r['their_elixir']:.1f}")
    parts.append(f"our hand {', '.join(r['our_hand'])}")
    parts.append(f"their hand {', '.join(r['their_hand'])}")
    if r["threat_left"] + r["threat_right"] >= 2:
        parts.append(f"their troops on our side: left {r['threat_left']:.0f}, right {r['threat_right']:.0f} elixir")
    return " · ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--flagged", type=int, default=100)
    ap.add_argument("--random", type=int, default=50)
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tbl = lancedb.connect(ROOT / "data/lancedb").open_table("frame_states")
    noted = list(fm.COLE)
    near = lambda m, keys: any(m["game"] == g and abs(m["t"] - t) <= qm.MATCH_S for g, t in keys)
    taken: list[tuple[str, float]] = []
    picked = []
    for mode, n in SHARE.items():
        n = round(n * a.flagged / 100)
        rule = qm.rule(tbl, mode)
        examples = [r for r in (qm.note_row(tbl, *key, qm.SCOPE[mode][1]) for key, v in fm.COLE.items() if mode in v)
                    if r is not None and not r["blind"]]
        hyb = qm.hybrid(tbl, mode, examples, a.k) if examples else []
        both = [m for m in rule if near(m, [(h["game"], h["t"]) for h in hyb])]
        rest = [m for m in rule if m not in both] + [m for m in hyb if not near(m, [(r["game"], r["t"]) for r in rule])]
        rng.shuffle(rest)  # past the moments both agree on, no ranking yet
        order = both + rest
        got = []
        for m in order:
            if len(got) >= n:
                break
            if near(m, noted) or near(m, taken):
                continue
            m["how"] = "rule + similar" if m in both else "rule" if m in rule else "similar to your notes"
            got.append(m)
            taken.append((m["game"], m["t"]))
        picked += got
        print(f"{mode:15s} {len(got)} (rule {len(rule)}, similar {len(hyb)}, both {len(both)})")
    plays = tbl.search().where(f"{qm.BASE} AND our_play_now IS NOT NULL").select(qm.COLS).limit(10**6).to_list()
    rng.shuffle(plays)
    rand = []
    for r in plays:
        m = {"game": r["game"], "t": round(float(r["t"]), 1), "clock": r["clock"], "kind": "random", "how": "random play"}
        if len(rand) >= a.random:
            break
        if near(m, noted) or near(m, taken):
            continue
        rand.append(m)
        taken.append((m["game"], m["t"]))
    print(f"random plays    {len(rand)}")
    allm = picked + rand
    rng.shuffle(allm)
    vids = json.loads((VIDEOS_FROM / "moments.json").read_text())["games"]
    for m in allm:
        r = tbl.search().where(f"game = '{m['game']}' AND t >= {m['t'] - 0.06} AND t <= {m['t'] + 0.06}").limit(1).to_list()[0]
        m.update(text=describe(r), opp=vids.get(m["game"], {}).get("opp"), result=r["result"], replay=r["replay_id"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "videos").exists():
        (out / "videos").symlink_to((VIDEOS_FROM / "videos").resolve())
    games = {g: vids[g] for g in {m["game"] for m in allm} if g in vids}
    (out / "moments.json").write_text(json.dumps({"games": games, "moments": allm}, indent=1))
    print(f"{len(allm)} moments ({len(picked)} flagged, {len(rand)} random) from {len(games)} games -> {out / 'moments.json'}")


if __name__ == "__main__":
    main()
