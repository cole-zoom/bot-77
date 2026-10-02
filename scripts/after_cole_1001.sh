#!/bin/zsh
# Cole's evo Valkyrie / evo Ice Spirit recording: process, then (after batch 4's chain) auto-label it, turn
# click_short3 into labels, recount coverage and rebuild the click page.
cd "$(dirname "$0")/.."
uv run python scripts/run_all.py data/logs/run_all_jobs_c1001.json > data/logs/run_all_c1001.out 2>&1
echo "C1001_DONE $(date +%H:%M)" >> data/logs/run_all.log
until grep -q AFTER_BATCH4_DONE data/logs/run_all.log; do sleep 60; done
uv run python scripts/autolabel_all.py 0 2 c1001_0 cole_2026-10-01 > data/logs/autolabel_c1001_0.log 2>&1 &
uv run python scripts/autolabel_all.py 1 2 c1001_1 cole_2026-10-01 > data/logs/autolabel_c1001_1.log 2>&1 &
wait
uv run python scripts/label_clicks.py > data/logs/label_clicks3.log 2>&1
uv run python scripts/coverage.py 2>&1 | grep -vi warn > data/logs/coverage_c1001.log
uv run python scripts/build_click_short.py data/logs/coverage_c1001.log data/reviews/click_short4.html > data/logs/click_short4.log 2>&1
echo "AFTER_C1001_DONE $(date +%H:%M)" >> data/logs/run_all.log
