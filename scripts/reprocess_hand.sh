#!/bin/zsh
# Re-run only the hand-reading/play-detection stage (`bot77 process`) for the given videos after the evo/hero
# template-scale fix, then auto-label them, re-label clicks, re-clean, recount.
# usage: scripts/reprocess_hand.sh <tag> <video_id> [video_id ...]
cd "$(dirname "$0")/.."
TAG=$1; shift
for v in "$@"; do
  row=$(uv run python -c "
import lancedb; r=lancedb.connect('data/lancedb').open_table('videos').search().where(\"video_id = '$v'\").to_arrow().to_pylist()[0]
print(r['path'], r['layout_id'], r['creator'] or 'unknown', sep='|')" 2>/dev/null | tail -1)
  p=${row%%|*}; rest=${row#*|}; lay=${rest%%|*}; cr=${rest#*|}
  uv run bot77 process --video "$p" --layout "$lay" --creator "$cr" > "data/logs/reprocess_$v.log" 2>&1 && echo "$(date +%H:%M:%S) $v: reprocessed hand" >> data/logs/run_all.log
done
V=$(IFS=,; echo "$*")
uv run python scripts/autolabel_all.py 0 1 "$TAG" "$V" > "data/logs/autolabel_$TAG.log" 2>&1
uv run python scripts/label_clicks.py > "data/logs/label_clicks_$TAG.log" 2>&1
rm -rf data/external/real_v2
uv run python scripts/clean_dataset.py > "data/logs/clean_$TAG.log" 2>&1
uv run python scripts/coverage.py data/external/real_v2 2>&1 | grep -vi warn > "data/logs/coverage_$TAG.log"
echo "REPROCESS_${TAG}_DONE $(date +%H:%M)" >> data/logs/run_all.log
