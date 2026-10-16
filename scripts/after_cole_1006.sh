#!/bin/zsh
# Session 1 recording (evo Skeleton Army, evo Lumberjack, hero Magic Archer): process, auto-label, re-clean, recount.
cd "$(dirname "$0")/.."
uv run python scripts/run_all.py data/logs/run_all_jobs_c1006.json > data/logs/run_all_c1006.out 2>&1
uv run python scripts/autolabel_all.py 0 1 c1006 cole_2026-10-06 > data/logs/autolabel_c1006.log 2>&1
rm -rf data/external/real_v2
uv run python scripts/clean_dataset.py > data/logs/clean_dataset8.log 2>&1
uv run python scripts/coverage.py data/external/real_v2 2>&1 | grep -vi warn > data/logs/coverage_clean8.log
echo "C1006_DONE $(date +%H:%M)" >> data/logs/run_all.log
