"""Auto-label the ground Spirit Empress plays (cost ~3), which collect_items skipped before 2026-10-01."""
import json, random
from pathlib import Path
from bot77.autolabel import AutoLabeler, collect_items
items = [i for i in collect_items() if i["entity"] == "spirit-empress-ground"]
print(len(items), "ground plays", flush=True)
al, rng, out = AutoLabeler(), random.Random(7), []
for it in items:
    try:
        out += [x for x in al.label(it, rng) if x]
    except Exception as e:  # noqa: BLE001
        print("ERR", it["video_id"], it["t_video"], e, flush=True)
Path("data/external/real_v1/manifest_se_ground.json").write_text(json.dumps(out, indent=1))
print("done:", len(out), "frames", dict(al.stats), flush=True)
