"""Development set for the opponent-play pipeline (decisions.md, 2026-10-10): every dual-POV paired match with exact
opponent truth (data/ground_truth/opponent_truth_auto_*.json) EXCEPT the held-out benchmark matches
(config/benchmark_holdout.json). Tune here; score the benchmark only with eval_v2.py, never tune on it.

Caveat: the detector trained on real frames from these videos, so detection here is a bit optimistic; the play-logic
errors (fake plays, grouping, timing, wrong card) are what this set is for.

  uv run python scripts/eval_dev.py <weights.pt> <tag> [--events-only] [--only m1,m2] [--set replay]
--set replay scores the replay test set (opponent_truth_replay_*: games the detector never trained on).
--events-only reuses unit_detections_<tag> and re-runs only detections -> plays -> score (fast iteration).
Writes data/logs/evaldev_<tag>.json with per-match and pooled scores.
"""
import argparse, glob, json
from pathlib import Path

from bot77.opponent import opponent_events
from bot77.review import score_opponent

ap = argparse.ArgumentParser()
ap.add_argument("weights"); ap.add_argument("tag")
ap.add_argument("--events-only", action="store_true"); ap.add_argument("--only", default=None)
ap.add_argument("--quiet", action="store_true")
ap.add_argument("--decode", default=None, help="JSON of DecodeParams fields to use the game-rules decoder ('{}' = defaults)")
ap.add_argument("--rules", default=None, help="JSON overrides of bot77.playrules.DEFAULT_RULES ('{}' = defaults)")
ap.add_argument("--out", default=None, help="events table suffix (default: tag)")
ap.add_argument("--set", default="auto", help="truth set: auto (dev, paired POVs) or replay (both-hands replays)")
a = ap.parse_args()
DB = Path("data/lancedb")
import lancedb
DB_ = lancedb.connect(DB)
hold = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])
truth = {}
for f in sorted(glob.glob(f"data/ground_truth/opponent_truth_{a.set}_*.json")):
    t = json.loads(Path(f).read_text())
    if t["matches"][0] not in hold:
        truth[t["matches"][0]] = t
if a.only:
    truth = {k: v for k, v in truth.items() if k in a.only.split(",")}
det = None
if not a.events_only:
    from bot77.detect.run import detect_video
    from bot77.detect.yolo import Bot77Detector
    det = Bot77Detector(Path(a.weights))
log = (lambda *x, **k: None) if a.quiet else print
dp = None
if a.decode is not None:
    from bot77.decode import DecodeParams
    dp = DecodeParams(**json.loads(a.decode))
results = {}
for match, t in truth.items():
    vid, idx = match.rsplit("_m", 1)
    if det is not None:
        tn = f"unit_detections_{a.tag}"
        done = tn in DB_.table_names(limit=10_000) and DB_.open_table(tn).count_rows(f"match_id = '{match}'") > 0
        if not done:
            detect_video(vid, DB, matches=[int(idx)], detector=det, table=tn)
    n_ev = opponent_events(vid, DB, matches=[int(idx)], det_table=f"unit_detections_{a.tag}",
                           out_table=f"opponent_events_dev_{a.out or a.tag}", log=log, decode_params=dp, thumbs=False,
                           rules=json.loads(a.rules) if a.rules is not None else None)
    results[match] = score_opponent(DB, t, table=f"opponent_events_dev_{a.out or a.tag}") if n_ev else \
        {"true_plays": len(t["items"]), "detected": 0, "fully_right": 0, "wrong_card": 0, "not_a_play": 0, "missed": len(t["items"])}
    r = results[match]
    print(f"{match:28s} plays {r['true_plays']:3d} right {r['fully_right']:3d} wrong {r['wrong_card']:3d} "
          f"fake {r['not_a_play']:3d} missed {r['missed']:3d}", flush=True)
tot = {k: sum(r[k] for r in results.values()) for k in ("true_plays", "fully_right", "wrong_card", "not_a_play", "missed", "detected")}
tot["fully_right_rate"] = round(tot["fully_right"] / tot["true_plays"], 3)
tot["found_recall"] = round((tot["fully_right"] + tot["wrong_card"]) / tot["true_plays"], 3)
tot["fake_per_match"] = round(tot["not_a_play"] / len(results), 1)
print("POOLED", json.dumps(tot))
Path(f"data/logs/evaldev_{a.out or a.tag}.json").write_text(json.dumps({"pooled": tot, "per_match": results}, indent=1, default=str))
