"""Score a detector on the held-out opponent-play benchmark (config/benchmark_holdout.json):
Alina match 1 (Cole's 52-play truth) and Cole/friend 2026-09-29 match 5 (pairing truth, 34 + 25 plays).

Detections go to separate tables (unit_detections_<tag>, opponent_events_<tag>), so nothing used in training changes.
  uv run python scripts/eval_v2.py <weights.pt> <tag> [--only uz4VVzGlOjE_m01]
"""
import argparse, json
from pathlib import Path

from bot77.detect.run import detect_video
from bot77.detect.yolo import Bot77Detector
from bot77.opponent import opponent_events
from bot77.review import score_opponent

ap = argparse.ArgumentParser()
ap.add_argument("weights"); ap.add_argument("tag"); ap.add_argument("--only", default=None)
ap.add_argument("--events-only", action="store_true", help="reuse unit_detections_<tag>; re-run plays + scoring only")
ap.add_argument("--rules", default=None, help="JSON overrides of bot77.playrules (fit on the dev set only, never here)")
ap.add_argument("--out", default=None, help="events table / log suffix (default: tag)")
a = ap.parse_args()
DB = Path("data/lancedb")
TRUTH = {"uz4VVzGlOjE_m01": "data/ground_truth/opponent_truth_uz4VVzGlOjE_m01.json",
         "cole_2026-09-29_c_m05": "data/ground_truth/opponent_truth_auto_cole_2026-09-29_c_m05.json",
         "friend_2026-09-29_c_m07": "data/ground_truth/opponent_truth_auto_friend_2026-09-29_c_m07.json"}
det = None if a.events_only else Bot77Detector(Path(a.weights))
out = a.out or a.tag
results = {}
for match, tf in TRUTH.items():
    if a.only and match != a.only:
        continue
    vid, idx = match.rsplit("_m", 1)
    n_det = detect_video(vid, DB, matches=[int(idx)], detector=det, table=f"unit_detections_{a.tag}") if det else 1
    n_ev = opponent_events(vid, DB, matches=[int(idx)], det_table=f"unit_detections_{a.tag}", out_table=f"opponent_events_{out}",
                           thumbs=not a.events_only, rules=json.loads(a.rules) if a.rules is not None else None) if n_det else 0
    print(match, n_det, "detections,", n_ev, "opponent events", flush=True)
    if not n_ev:  # nothing detected: every truth play is missed (no table is written, so don't try to read one)
        results[match] = {"detections": n_det, "opponent_events": 0, "note": "nothing detected; all plays missed"}
    else:
        results[match] = score_opponent(DB, json.loads(Path(tf).read_text()), table=f"opponent_events_{out}")
    print(match, json.dumps(results[match], default=str)[:600], flush=True)
Path(f"data/logs/eval_{out}.json").write_text(json.dumps(results, indent=1, default=str))
