#!/bin/bash
# V4 mining chain B (decisions.md, 2026-10-10): detections -> LanceDB -> auto-label (exact card/form from the hand
# reader / paired truth) -> clean (crop-classifier check) -> empty-arena composites (real_v4). Prints CHAIN_B_DONE.
#   mine_v4_chainB.sh <pseudo-label weights> [workers=4]
set -euo pipefail
W=$1; N=${2:-2}
cd "$(dirname "$0")/.."
uv run python scripts/mine_v4_detect.py import data/upload/mine_v4/detections.jsonl unit_detections_mine4
rm -rf data/external/real_v4_raw
for k in $(seq 0 $((N - 1))); do
  uv run python scripts/autolabel_v4.py $k $N unit_detections_mine4 > data/logs/autolabel_v4_w$k.log 2>&1 &
done
wait
grep -h "done:" data/logs/autolabel_v4_w*.log
uv run python scripts/clean_dataset.py data/external/real_v4_raw data/external/real_v4_clean > data/logs/clean_v4.log 2>&1
tail -3 data/logs/clean_v4.log
uv run python scripts/build_real_v3.py --src data/external/real_v4_clean --dst data/external/real_v4 --pseudo "$W" \
  --bg-dir data/external/real_v3/bg --jobs 4 > data/logs/build_real_v4.log 2>&1
tail -3 data/logs/build_real_v4.log
echo CHAIN_B_DONE
