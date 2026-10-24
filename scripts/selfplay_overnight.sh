#!/usr/bin/env bash
# Keep the self-play training running until a deadline: restart it (it resumes from its checkpoint) if it exits; at the
# deadline, wait for its next periodic save, then stop it.
#   nohup scripts/selfplay_overnight.sh 13:15 >/dev/null 2>&1 &
# (SIGINT is ignored under nohup, so the trainer's own Ctrl-C save can't be used: 2026-10-24.)
set -uo pipefail
cd "$(dirname "$0")/.."
UNTIL="${1:-07:30}"
END=$(date -j -f "%H:%M" "$UNTIL" +%s)
[ "$END" -lt "$(date +%s)" ] && END=$((END + 86400))
CKPT=data/models/selfplay_v1/model_weights.pth
LOG=data/models/selfplay_v1/logs/supervisor.log
mkdir -p "$(dirname "$LOG")"
pid=""
while [ "$(date +%s)" -lt "$END" ]; do
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    echo "$(date +%H:%M:%S) start" >> "$LOG"
    scripts/selfplay_train.sh >> "$LOG" 2>&1 &
    pid=$!
  fi
  sleep 30
done
saved=$(stat -f %m "$CKPT" 2>/dev/null || echo 0)
for _ in $(seq 1 40); do  # up to 20 min for the next periodic save (every 250 episodes, ~5 min)
  [ "$(stat -f %m "$CKPT" 2>/dev/null || echo 0)" -gt "$saved" ] && break
  sleep 30
done
sleep 20
pkill -TERM -f "python_ai/trainers/train.py"
echo "$(date +%H:%M:%S) deadline: stopped after a fresh save" >> "$LOG"
