#!/bin/zsh
cd "$(dirname "$0")/.."
until grep -q SESSION_F_PAGE_DONE data/logs/run_all.log; do sleep 30; done
uv run python scripts/label_clicks.py > data/logs/label_clicks_DE.log 2>&1
rm -rf data/external/real_v2
uv run python scripts/clean_dataset.py > data/logs/clean_DE.log 2>&1
uv run python scripts/coverage.py data/external/real_v2 2>&1 | grep -vi warn > data/logs/coverage_DE.log
echo "RECOUNT_DE_DONE $(date +%H:%M)" >> data/logs/run_all.log
