#!/bin/bash
# After recording replays (2026-11-16): ingest them, then rebuild the tables in dependency order. Duplicates are found
# from the replays' hand plays, so build_sources runs again once the plays exist (split 'duplicate' for the later copy).
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo "=== $1 $(date +%H:%M)"; nice -n 10 uv run --no-sync python "scripts/$1.py"; }
run ingest_replays
for s in build_sources build_detections build_plays find_duplicate_replays build_sources build_moments build_tower_hp \
         read_card_levels pair_bot_replays import_reviews build_indexes; do
  run "$s"
done
echo "=== done $(date +%H:%M)"
