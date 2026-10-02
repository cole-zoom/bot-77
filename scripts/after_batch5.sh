#!/bin/zsh
# Batch 5 (queue4 YouTube videos): once downloaded and the Cole 2026-10-01 chain is done, pick layouts,
# process, auto-label the new videos, recount coverage, rebuild the click page.
cd "$(dirname "$0")/.."
until grep -q DL4_DONE data/logs/run_all.log && grep -q AFTER_C1001_DONE data/logs/run_all.log; do sleep 60; done
uv run python scripts/prepare_batch.py data/logs/run_all_jobs5.json > data/logs/prepare_batch5.out 2>&1
uv run python scripts/run_all.py data/logs/run_all_jobs5.json > data/logs/run_all_batch5.out 2>&1
echo "BATCH5_DONE $(date +%H:%M)" >> data/logs/run_all.log
V=$(python3 -c "import json; print(','.join(j['id'] for j in json.load(open('data/logs/run_all_jobs5.json'))))")
uv run python scripts/autolabel_all.py 0 2 b5_0 $V > data/logs/autolabel_b5_0.log 2>&1 &
uv run python scripts/autolabel_all.py 1 2 b5_1 $V > data/logs/autolabel_b5_1.log 2>&1 &
wait
uv run python scripts/label_clicks.py > data/logs/label_clicks5.log 2>&1
uv run python scripts/coverage.py 2>&1 | grep -vi warn > data/logs/coverage_batch5.log
uv run python scripts/build_click_short.py data/logs/coverage_batch5.log data/reviews/click_short5.html > data/logs/click_short5.log 2>&1
echo "AFTER_BATCH5_DONE $(date +%H:%M)" >> data/logs/run_all.log
