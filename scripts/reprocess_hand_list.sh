#!/bin/zsh
# Re-run hand reading / play detection (`bot77 process`) for a list of videos, 2 at a time, after the evo/hero
# reader fixes (2026-10-07); then auto-label them, re-label clicks, re-clean, recount once.
# usage: scripts/reprocess_hand_list.sh <tag> <file with one video_id per line>
cd "$(dirname "$0")/.."
TAG=$1; LIST=$2; TODO=${3:-$2}  # optional 3rd arg: the subset still to re-read (the rest were already re-read)
while pgrep -f "bot77 process" > /dev/null; do sleep 30; done  # never more than 2 re-reads at once
# exactly 2 at a time: a `jobs -r` count inside $(...) runs in a subshell that sees no jobs, so the first version
# of this script started all 30 videos at once (killed within 30 s, no writes; 2026-10-07)
xargs -n 1 -P 2 scripts/reprocess_one.sh < "$TODO"
V=$(paste -sd, "$LIST")
python3 - "$V" <<'PY'
import json, sys
from pathlib import Path
ids = set(sys.argv[1].split(",")); n = 0
for f in Path("data/external/real_v1").glob("manifest_*.json"):
    if f.stem in ("manifest_clicks", "manifest_reviewed", "manifest_mined"):
        continue  # Cole-verified labels stay
    m = json.loads(f.read_text()); keep = [x for x in m if x["video_id"] not in ids]; n += len(m) - len(keep)
    f.write_text(json.dumps(keep, indent=1))
print("dropped", n, "auto-labels made with the old hand reader")
PY
uv run python scripts/autolabel_all.py 0 2 "${TAG}_0" "$V" > "data/logs/autolabel_${TAG}_0.log" 2>&1 &
uv run python scripts/autolabel_all.py 1 2 "${TAG}_1" "$V" > "data/logs/autolabel_${TAG}_1.log" 2>&1 &
wait
uv run python scripts/label_clicks.py > "data/logs/label_clicks_$TAG.log" 2>&1
rm -rf data/external/real_v2
uv run python scripts/clean_dataset.py > "data/logs/clean_$TAG.log" 2>&1
uv run python scripts/coverage.py data/external/real_v2 2>&1 | grep -vi warn > "data/logs/coverage_$TAG.log"
echo "REPROCESS_${TAG}_DONE $(date +%H:%M)" >> data/logs/run_all.log
