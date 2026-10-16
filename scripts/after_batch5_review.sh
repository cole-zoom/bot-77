#!/bin/zsh
# Morning review set: spot-check 3 (A auto, B clicks, C grey boxes), built once batch 5's chain has finished.
cd "$(dirname "$0")/.."
until grep -q AFTER_BATCH5_DONE data/logs/run_all.log; do sleep 60; done
uv run python scripts/spotcheck3.py > data/logs/spotcheck3.log 2>&1
echo "SPOTCHECK3_READY $(date +%H:%M)" >> data/logs/run_all.log
