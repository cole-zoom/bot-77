#!/bin/bash
# After the 2026-10-24 overnight (55 TV Royale replays + the self-play agent's 20 games): process the new replays, then
# rebuild the tables in dependency order (as after_session_2026-10-23.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --no-sync python scripts/ingest_replays.py
for s in build_sources build_detections build_plays build_moments pair_bot_replays import_reviews build_indexes; do
  echo "=== $s"; uv run --no-sync python scripts/$s.py
done
echo "=== done"
