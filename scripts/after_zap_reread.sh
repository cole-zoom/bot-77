#!/bin/zsh
# Re-read the 6 Zap videos that weren't in round 1, after the session A click chain finishes (one writer at a time).
cd "$(dirname "$0")/.."
until grep -q SA_CLEAN_DONE data/logs/run_all.log; do sleep 60; done
scripts/reprocess_hand_list.sh zap data/logs/reprocess_zap.txt
