#!/bin/bash
# After session 2 (2026-10-23, 61 games, new deck): rebuild the tables from the new replays, in dependency order.
# Run after scripts/ingest_replays.py has processed data/raw/replays/2026-10-23.
set -euo pipefail
cd "$(dirname "$0")/.."
for s in build_sources build_detections build_plays build_moments pair_bot_replays import_reviews build_indexes; do
  echo "=== $s"; uv run python scripts/$s.py
done
echo "=== done"
