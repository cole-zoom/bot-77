#!/bin/zsh
# Re-run videos whose layout was fixed (scripts/fix_layouts.py + the ian_split_v2a/b refit), drop the
# auto-labels made under the broken layout, auto-label again, recount, click page 7.
cd "$(dirname "$0")/.."
echo "FIX_STARTED $(date +%H:%M)" >> data/logs/run_all.log
V=$(python3 -c "import json; print(','.join(j['id'] for j in json.load(open('data/logs/run_all_jobs_fix.json'))))")
python3 - "$V" <<'PY'
import json, sys
from pathlib import Path
ids = set(sys.argv[1].split(",")); n = 0
for f in Path("data/external/real_v1").glob("manifest_*.json"):
    if f.stem in ("manifest_clicks", "manifest_reviewed"):
        continue  # Cole's clicks and approvals stay: they don't depend on the layout's elixir/hand reading
    m = json.loads(f.read_text()); keep = [x for x in m if x["video_id"] not in ids]; n += len(m) - len(keep)
    f.write_text(json.dumps(keep, indent=1))
print("dropped", n, "auto-labels from re-run videos")
PY
uv run python scripts/run_all.py data/logs/run_all_jobs_fix.json > data/logs/run_all_fix.out 2>&1
uv run python scripts/autolabel_all.py 0 2 fix_0 $V > data/logs/autolabel_fix_0.log 2>&1 &
uv run python scripts/autolabel_all.py 1 2 fix_1 $V > data/logs/autolabel_fix_1.log 2>&1 &
wait
uv run python scripts/label_clicks.py > data/logs/label_clicks_fix.log 2>&1
uv run python scripts/coverage.py 2>&1 | grep -vi warn > data/logs/coverage_fix.log
uv run python scripts/build_click_short.py data/logs/coverage_fix.log data/reviews/click_short7.html > data/logs/click_short7.log 2>&1
echo "FIX_DONE $(date +%H:%M)" >> data/logs/run_all.log
