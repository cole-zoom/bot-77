#!/bin/zsh
cd "$(dirname "$0")/.."
uv run python scripts/mine_import.py data/ground_truth/reviews/mining_2026-10-05.json > data/logs/mine_import.log 2>&1
rm -rf data/external/real_v2
uv run python scripts/clean_dataset.py > data/logs/clean_dataset7.log 2>&1
uv run python scripts/coverage.py data/external/real_v2 2>&1 | grep -vi warn > data/logs/coverage_clean7.log
echo "MINED_IMPORT_DONE $(date +%H:%M)" >> data/logs/run_all.log
