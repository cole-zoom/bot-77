#!/bin/zsh
# After CLEAN5: one more click pass (box-off 2 arrived mid-run), re-clean, recount, and the 50-frame click-box re-check.
cd "$(dirname "$0")/.."
until grep -q CLEAN5_DONE data/logs/run_all.log; do sleep 60; done
uv run python scripts/label_clicks.py > data/logs/label_clicks9.log 2>&1
rm -rf data/external/real_v2
uv run python scripts/clean_dataset.py > data/logs/clean_dataset6.log 2>&1
uv run python scripts/coverage.py data/external/real_v2 2>&1 | grep -vi warn > data/logs/coverage_clean6.log
uv run python -c "
from pathlib import Path
from bot77.spotcheck import write_spotcheck
R = Path('data/external/real_v1')
print(write_spotcheck(R, Path('data/reviews/spotcheck4_clicks.html'), 50, 4, [R / 'manifest_clicks.json'], 'Spot check 4 · click boxes after the box fix'))
" > data/logs/spotcheck4.log 2>&1
echo "CLEAN6_DONE $(date +%H:%M)" >> data/logs/run_all.log
