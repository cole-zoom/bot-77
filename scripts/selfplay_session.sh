#!/usr/bin/env bash
# After the self-play training stops: freeze its checkpoint and play a real session with it.
#   scripts/selfplay_session.sh [session_hours=2]
set -uo pipefail
cd "$(dirname "$0")/.."
HOURS="${1:-2}"
LOG=data/logs/selfplay_session_$(date +%Y-%m-%d).log
while pgrep -f "selfplay_overnight.sh" >/dev/null || pgrep -f "python_ai/trainers/train.py" >/dev/null; do sleep 30; done
FROZEN=data/models/selfplay_v1/frozen_$(date +%Y%m%d-%H%M).pth
cp data/models/selfplay_v1/model_weights.pth "$FROZEN"
echo "$(date +%H:%M) training stopped; playing ${HOURS} h with $FROZEN" >> "$LOG"
caffeinate -i -s uv run --no-sync python scripts/live_session.py --hours "$HOURS" --policy selfplay --ckpt "$FROZEN" >> "$LOG" 2>&1
echo "$(date +%H:%M) session ended ($?)" >> "$LOG"
