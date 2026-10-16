#!/bin/bash
# Keep processing replays while recording runs (decisions.md, 2026-10-13): wait for any ingest in progress, then
# re-run ingest_replays.py every INTERVAL seconds (it only processes new recordings); exit once the TV loop has
# stopped and a pass found nothing new.
#   bash scripts/ingest_loop.sh [interval_s=600]
cd "$(dirname "$0")/.."
INTERVAL=${1:-600}
LOG=data/logs/ingest_loop.log
while pgrep -f "scripts/ingest_replays.py" >/dev/null; do sleep 30; done
while true; do
  echo "== $(date '+%m-%d %H:%M') ingest pass" >> "$LOG"
  uv run python scripts/ingest_replays.py > data/logs/ingest_pass.log 2>&1
  echo "== $(date '+%m-%d %H:%M') pass ended ($?): $(grep -E 'videos to process|truth written' data/logs/ingest_pass.log | tr '\n' ' ')" >> "$LOG"
  if grep -q " 0 videos to process" data/logs/ingest_pass.log && ! pgrep -f "tv_loop.sh" >/dev/null; then
    echo "== $(date '+%m-%d %H:%M') all recordings processed; recording stopped; done" >> "$LOG"; break
  fi
  sleep "$INTERVAL"
done
