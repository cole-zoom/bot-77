#!/bin/bash
# Keep recording new TV Royale games for a few hours (decisions.md, 2026-10-12): wait for any run in progress, then
# re-run the runner every INTERVAL seconds (it skips games already recorded). Keeps the Mac awake while it runs.
#   caffeinate -i -s bash scripts/tv_loop.sh [hours=5] [interval_s=1800] [max_tabs=6]
# max_tabs: leagues to go through from the top (Cole 2026-10-13: all of them, Master 1 and every arena -> 60)
cd "$(dirname "$0")/.."
HOURS=${1:-5}; INTERVAL=${2:-1800}; TABS=${3:-6}
END=$(( $(date +%s) + HOURS * 3600 ))
LOG=data/logs/tv_runner_$(date +%Y-%m-%d).log
while pgrep -f "replay_runner.py --source tv" >/dev/null; do sleep 20; done
while [ "$(date +%s)" -lt "$END" ]; do
  echo "== $(date +%H:%M) tv_loop: run" >> "$LOG"
  uv run python scripts/replay_runner.py --source tv -n 60 --max-tabs "$TABS" >> "$LOG" 2>&1
  echo "== $(date +%H:%M) tv_loop: run ended ($?)" >> "$LOG"
  sleep "$INTERVAL"
done
echo "== $(date +%H:%M) tv_loop: done" >> "$LOG"
