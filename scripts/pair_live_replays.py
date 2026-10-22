"""Answer keys for Cole's live recordings from the Battle Log replays recorded after them (2026-10-17).

Every match of the live videos is tried against every replay (processed by ingest_replays.py as bl_<date>_rNN with
the bottom hand and bl_<date>_rNN_top with the top hand); bot77.pairing accepts a pair only when Cole's own live plays
reappear in the replay's bottom hand at the match-clock offset, so the order of the games doesn't matter and a live
game stopped early still pairs. Writes data/ground_truth/opponent_truth_live_<live match>.json.

  uv run python scripts/pair_live_replays.py --live cole_2026-10-17_a,cole_2026-10-17_b,cole_2026-10-17_c --replays bl_2026-10-17
"""
import argparse
from collections import Counter

import lancedb

from pathlib import Path

from bot77.pairing import pair_live_replay, write_live_pair

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True, help="comma list of live video ids")
ap.add_argument("--replays", required=True, help="replay video id prefix, e.g. bl_2026-10-17")
a = ap.parse_args()

db = lancedb.connect("data/lancedb")
vids = {r["video_id"] for r in db.open_table("videos").search().limit(10**6).select(["video_id"]).to_arrow().to_pylist()}
reps = sorted(v for v in vids if v.startswith(a.replays + "_r") and not v.endswith("_top") and f"{v}_top" in vids)
print(len(reps), "replays:", reps[:3], "...")
paired = Counter()
for live in a.live.split(","):
    n_matches = db.open_table("matches").count_rows(f"video_id = '{live}'")
    cands = [dict(p, top=f"{rep}_top") for rep in reps for p in pair_live_replay(live, rep)]
    # one-to-one: best agreement first; each live match and each replay used once (a same-deck game can also clear the
    # bar against another replay, 2026-10-20)
    used_a, used_b, chosen = set(), set(), []
    for p in sorted(cands, key=lambda p: -p["agree"]):
        if p["a"] not in used_a and p["b"] not in used_b:
            chosen.append(p); used_a.add(p["a"]); used_b.add(p["b"])
    for old in Path("data/ground_truth").glob(f"opponent_truth_live_{live}_m*.json"):
        old.unlink()  # rewritten below from the one-to-one choice
    for p in sorted(chosen, key=lambda p: p["a"]):
        write_live_pair(p, p["top"])
    paired[live] = len(chosen)
    print(f"{live}: {len(chosen)} of {n_matches} matches paired ({len(cands)} candidate pairs)")
print("total keyed live matches:", sum(paired.values()))
