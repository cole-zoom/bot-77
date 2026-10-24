#!/usr/bin/env bash
# The end of the 2026-10-24 overnight run: free BlueStacks, freeze the self-play agent, and let it play a real session.
#   scripts/overnight_end.sh [stop_recorder_at=06:45] [session_hours=1]
set -uo pipefail
cd "$(dirname "$0")/.."
STOP_AT="${1:-06:45}"; HOURS="${2:-1}"
LOG=data/logs/overnight_end.log
until_time() { local t; t=$(date -j -f "%H:%M" "$1" +%s); [ "$t" -lt "$(date +%s)" ] && return; sleep $((t - $(date +%s))); }
echo "$(date +%H:%M) waiting for $STOP_AT" >> "$LOG"
until_time "$STOP_AT"
pkill -f "tv_loop.sh"; pkill -f "replay_runner.py"; sleep 5; pkill -INT -f "scrcpy -s 127.0.0.1:5555 .*tv_royale"; sleep 10
echo "$(date +%H:%M) recorder stopped; replays tonight: $(ls data/raw/tv_royale/2026-10-24/*.mp4 2>/dev/null | wc -l)" >> "$LOG"
while pgrep -f "selfplay_overnight.sh" >/dev/null; do sleep 30; done  # training ends at its deadline (07:15)
sleep 20
cp data/models/selfplay_v1/model_weights.pth data/models/selfplay_v1/frozen.pth
echo "$(date +%H:%M) training done; frozen checkpoint; starting a ${HOURS} h session with the self-play agent" >> "$LOG"
pkill -f "ingest_loop.sh" || true
caffeinate -i -s uv run --no-sync python scripts/live_session.py --hours "$HOURS" --policy selfplay \
  --ckpt data/models/selfplay_v1/frozen.pth >> data/logs/session_selfplay_2026-10-24.log 2>&1
echo "$(date +%H:%M) session ended ($?)" >> "$LOG"
