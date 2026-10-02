#!/bin/zsh
# Batch 6 (queue5: the 27 evo/hero forms with no class yet): download after queue4, process after batch 5,
# auto-label, recount coverage, build the seed click page. Mining waits for Cole's seed clicks.
cd "$(dirname "$0")/.."
until grep -q DL4_DONE data/logs/run_all.log; do sleep 60; done
for i in 1 2 3; do uv run yt-dlp --js-runtimes node --no-warnings -q --continue --retries 15 --fragment-retries 15 --sleep-requests 2 --sleep-interval 20 -S "res:1440" -o "data/raw/%(id)s.%(ext)s" --print-to-file "%(id)s | %(channel)s | %(title)s | %(duration_string)s | %(width)sx%(height)s" data/raw/youtube/downloaded.txt -a data/raw/youtube/queue5.txt >> data/logs/yt_download5.log 2>&1; ls data/raw | grep -q "\.part$" || break; sleep 300; done
echo "DL5_DONE $(date +%H:%M)" >> data/logs/run_all.log
until grep -q AFTER_BATCH5_DONE data/logs/run_all.log; do sleep 60; done
uv run python scripts/prepare_batch.py data/logs/run_all_jobs6.json > data/logs/prepare_batch6.out 2>&1
uv run python scripts/run_all.py data/logs/run_all_jobs6.json > data/logs/run_all_batch6.out 2>&1
echo "BATCH6_DONE $(date +%H:%M)" >> data/logs/run_all.log
V=$(python3 -c "import json; print(','.join(j['id'] for j in json.load(open('data/logs/run_all_jobs6.json'))))")
uv run python scripts/autolabel_all.py 0 2 b6_0 $V > data/logs/autolabel_b6_0.log 2>&1 &
uv run python scripts/autolabel_all.py 1 2 b6_1 $V > data/logs/autolabel_b6_1.log 2>&1 &
wait
uv run python scripts/label_clicks.py > data/logs/label_clicks6.log 2>&1
uv run python scripts/coverage.py 2>&1 | grep -vi warn > data/logs/coverage_batch6.log
uv run python scripts/build_click_short.py data/logs/coverage_batch6.log data/reviews/click_short6.html > data/logs/click_short6.log 2>&1
echo "AFTER_BATCH6_DONE $(date +%H:%M)" >> data/logs/run_all.log
