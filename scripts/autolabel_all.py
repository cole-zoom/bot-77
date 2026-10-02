"""Auto-label every trusted play (worker k of n), writing into data/external/real_v1."""
import json, random, sys
from pathlib import Path
from bot77.autolabel import AutoLabeler, collect_items

k, n = int(sys.argv[1]), int(sys.argv[2])
tag = sys.argv[3] if len(sys.argv) > 3 else str(k)  # manifest name; with a tag, only the listed videos
only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
items = [i for i in collect_items() if only is None or i["video_id"] in only][k::n]
al = AutoLabeler()
rng = random.Random(k)
out = []
for i, it in enumerate(items):
    try:
        out += [x for x in al.label(it, rng) if x]
    except Exception as e:  # noqa: BLE001
        print("ERR", it["video_id"], it["t_video"], e, flush=True)
    if i % 50 == 0:
        print(f"worker {k}: {i}/{len(items)} plays, {len(out)} frames", flush=True)
Path(f"data/external/real_v1/manifest_{tag}.json").write_text(json.dumps(out, indent=1))
Path(f"data/external/real_v1/stats_{tag}.json").write_text(json.dumps(dict(al.stats), indent=1))
print(f"worker {k} done: {len(out)} frames", flush=True)
