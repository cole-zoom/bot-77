#!/bin/zsh
# Re-run `bot77 process` for one video (used by reprocess_hand_list.sh through xargs -P 2).
cd "$(dirname "$0")/.."
v=$1
row=$(uv run python -c "
import lancedb; r=lancedb.connect('data/lancedb').open_table('videos').search().where(\"video_id = '$v'\").to_arrow().to_pylist()[0]
print(r['path'], r['layout_id'], r['creator'] or 'unknown', sep='|')" 2>/dev/null | tail -1)
p=${row%%|*}; rest=${row#*|}; lay=${rest%%|*}; cr=${rest#*|}
uv run bot77 process --video "$p" --layout "$lay" --creator "$cr" > "data/logs/reprocess_$v.log" 2>&1 \
  && echo "$(date +%H:%M:%S) $v: reprocessed hand" >> data/logs/run_all.log \
  || echo "$(date +%H:%M:%S) ERROR reprocess $v" >> data/logs/run_all.log
