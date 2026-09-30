"""After run_all: any video that produced 0 matches gets a per-video auto layout and is rerun
(creators' HUD offsets can differ between their own videos — OYASSUU did)."""
import json
from pathlib import Path

import lancedb

from bot77.autolayout import derive_layout
from bot77.layout import LAYOUT_DIR
from bot77.video import probe

jobs = json.loads(Path("data/logs/run_all_jobs.json").read_text())
db = lancedb.connect("data/lancedb")
have = {m["video_id"] for m in db.open_table("matches").search().limit(100_000).select(["video_id"]).to_arrow().to_pylist()}
redo = []
for j in jobs:
    if j["id"] in have:
        continue
    lid = f"auto_{j['id']}"
    if not (LAYOUT_DIR / f"{lid}.json").exists() and not (LAYOUT_DIR / f"oyassuu_{j['id']}.json").exists():
        du = probe(Path(j["video"]))["duration"]
        derive_layout(Path(j["video"]), lid, [du * q for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)],
                      f"per-video auto layout for {j['id']}")
    j = {**j, "layout": f"oyassuu_{j['id']}" if (LAYOUT_DIR / f"oyassuu_{j['id']}.json").exists() else lid}
    j.pop("convert", None)
    redo.append(j)
Path("data/logs/rerun_jobs.json").write_text(json.dumps(redo, indent=1))
print(len(redo), "videos to rerun:", [j["id"] for j in redo])
