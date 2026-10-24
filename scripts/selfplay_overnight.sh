#!/usr/bin/env bash
# Keep the self-play training running until a deadline: restart it (it resumes from its checkpoint) if it exits.
#   nohup scripts/selfplay_overnight.sh 07:30 >/dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
UNTIL="${1:-07:30}"
END=$(date -j -f "%H:%M" "$UNTIL" +%s)
[ "$END" -lt "$(date +%s)" ] && END=$((END + 86400))
LOG=data/models/selfplay_v1/logs/supervisor.log
mkdir -p "$(dirname "$LOG")"
while [ "$(date +%s)" -lt "$END" ]; do
  echo "$(date +%H:%M:%S) start" >> "$LOG"
  SMOKE_SECONDS=$((END - $(date +%s))) scripts/selfplay_train.sh >> "$LOG" 2>&1
  echo "$(date +%H:%M:%S) exited ($?)" >> "$LOG"
  sleep 30
done
echo "$(date +%H:%M:%S) deadline reached" >> "$LOG"
