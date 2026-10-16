#!/bin/zsh
# After the layout-fix chain: rebuild the cleaned set (real_v2) with the re-run videos and Cole's newest clicks,
# recount on cleaned labels only, and build the click page from the cleaned shortfall.
cd "$(dirname "$0")/.."
until grep -q FIX_DONE data/logs/run_all.log; do sleep 60; done
rm -rf data/external/real_v2
uv run python scripts/clean_dataset.py > data/logs/clean_dataset3.log 2>&1
uv run python scripts/coverage.py data/external/real_v2 2>&1 | grep -vi warn > data/logs/coverage_clean3.log
uv run python scripts/build_click_short.py data/logs/coverage_clean3.log data/reviews/click_short7.html > data/logs/click_short7.log 2>&1
echo "CLEAN3_DONE $(date +%H:%M)" >> data/logs/run_all.log
