#!/bin/zsh
# Download a queue of YouTube videos into data/raw, logging to downloaded.txt. 1080p cap (decisions.md,
# 2026-10-01): the detector sees ~900 px of arena, layouts are built at 1080 wide, and 1440p only made decoding
# ~1.8x slower; 720p would shrink the arena below the detector's input and blur small units and HUD text.
# usage: scripts/download.sh data/raw/youtube/<queue>.txt <logname>
cd "$(dirname "$0")/.."
for i in 1 2 3; do
  uv run yt-dlp --js-runtimes node --no-warnings -q --continue --retries 15 --fragment-retries 15 --sleep-requests 2 \
    --sleep-interval 20 -S "res:1080" -o "data/raw/%(id)s.%(ext)s" \
    --print-to-file "%(id)s | %(channel)s | %(title)s | %(duration_string)s | %(width)sx%(height)s" data/raw/youtube/downloaded.txt \
    -a "$1" >> "data/logs/yt_download_${2:-queue}.log" 2>&1
  ls data/raw | grep -q "\.part$" || break
  sleep 300
done
