"""Find videos whose plays almost never confirm (layout reads the clock but not the elixir or hand), score every
candidate layout from the same creator plus a per-video auto layout with bot77.layoutcheck, and write a re-run
job list for those where a layout reads everything (decisions.md, 2026-10-02)."""
import collections, json, sys
from pathlib import Path
import lancedb, numpy as np
from bot77.autolayout import derive_layout
from bot77.layoutcheck import score_layout
from bot77.video import probe

MIN_HIGH, MIN_SCORE = 0.3, 0.4
db = lancedb.connect("data/lancedb")
vids = {v["video_id"]: v for v in db.open_table("videos").search().limit(10_000).to_arrow().to_pylist()}
ev = db.open_table("events").search().where("kind = 'play'").limit(10**7).select(["video_id", "confidence"]).to_arrow().to_pylist()
hi = collections.defaultdict(list)
for e in ev:
    hi[e["video_id"]].append(e["confidence"] == "high")
bad = [v for v, h in hi.items() if np.mean(h) < MIN_HIGH and not v.startswith(("cole_", "friend_"))]
by_creator = collections.defaultdict(set)
for v in vids.values():
    if np.mean(hi.get(v["video_id"], [0])) >= MIN_HIGH:
        by_creator[v["creator"]].add(v["layout_id"])  # layouts proven on other videos of the creator
jobs, report = [], []
for v in sorted(bad):
    info, p = vids[v], Path(vids[v]["path"])
    cands = {info["layout_id"]} | by_creator[info["creator"]]
    try:
        derive_layout(p, f"auto_{v}", [probe(p)["duration"] * q for q in np.linspace(0.1, 0.9, 9)], f"{info['creator']} per-video")
        cands.add(f"auto_{v}")
    except Exception as e:  # noqa: BLE001
        report.append(f"{v}: auto layout failed ({str(e)[:60]})")
    scores = sorted((score_layout(p, c) for c in cands), key=lambda s: -s.total)
    cur = next(s for s in scores if s.layout_id == info["layout_id"])
    best = scores[0]
    report.append(f"{v} [{info['creator']}] high {np.mean(hi[v]):.0%}\n    now  {cur}\n    best {best}")
    if best.total >= MIN_SCORE and best.layout_id != info["layout_id"]:
        jobs.append({"id": v, "video": str(p), "layout": best.layout_id, "creator": info["creator"]})
Path("data/logs/run_all_jobs_fix.json").write_text(json.dumps(jobs, indent=1))
Path("data/logs/fix_layouts.log").write_text("\n".join(report))
print("\n".join(report)); print(len(bad), "bad videos;", len(jobs), "fixable")
