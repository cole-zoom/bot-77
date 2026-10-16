#!/bin/zsh
# After mining run 3: rebuild crops for every video whose unit_crops rows are duplicated (build_crops used to
# append on re-run). build_crops now deletes a video's crops first.
cd "$(dirname "$0")/.."
until grep -q MINE3_DONE data/logs/run_all.log; do sleep 30; done
uv run python - <<'PY' > data/logs/dedupe_crops.log 2>&1
import collections, lancedb
from bot77.mine import build_crops
t = lancedb.connect("data/lancedb").open_table("unit_crops")
total = t.count_rows(); c = collections.Counter(); dup = set(); seen = set()
for off in range(0, total, 200_000):
    for cid, vid in zip(*t.search().select(["crop_id", "video_id"]).offset(off).limit(200_000).to_arrow().to_pydict().values()):
        if cid in seen:
            dup.add(vid)
        seen.add(cid)
print("videos with duplicated crops:", sorted(dup), flush=True)
for v in sorted(dup):
    print(v, build_crops(v), "crops", flush=True)
PY
echo "DEDUPE_DONE $(date +%H:%M)" >> data/logs/run_all.log
