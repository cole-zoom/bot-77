"""V4 data engine, step 3: auto-label every trusted play from the newest detector's detections (worker k of n).
Same labelling as autolabel_all.py (exact card + form from the hand reader / paired truth), but tracks come from
unit_detections_<table> (mine_v4_detect.py import) and frames go to data/external/real_v4_raw. Benchmark excluded.
  uv run python scripts/autolabel_v4.py <k> <n> <dets table>
"""
import json, random, sys
from pathlib import Path

from bot77.autolabel import AutoLabeler, collect_items

k, n, table = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
OUT = Path("data/external/real_v4_raw")
hold = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])
items = [i for i in collect_items() if i["match_id"] not in hold][k::n]
al = AutoLabeler(out=OUT, dets_table=table)
rng = random.Random(k)
out = []
for i, it in enumerate(items):
    try:
        out += [x for x in al.label(it, rng) if x]
    except Exception as e:  # noqa: BLE001
        print("ERR", it["video_id"], it["t_video"], e, flush=True)
    if i % 100 == 0:
        print(f"worker {k}: {i}/{len(items)} plays, {len(out)} frames", flush=True)
(OUT / f"manifest_w{k}.json").write_text(json.dumps(out, indent=1))
(OUT / f"stats_w{k}.json").write_text(json.dumps(dict(al.stats), indent=1))
print(f"worker {k} done: {len(out)} frames", flush=True)
