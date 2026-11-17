#!/usr/bin/env bash
# On the Mac: everything the pod needs for the pro-replay board reads (step 5, 2026-11-16), in one folder on the LaCie.
#   bash scripts/pod/pack.sh [JOB_DIR] [--transcode]
# JOB_DIR (default /Volumes/LaCie/bot-77-data/pod_job) gets: code.tgz (src/bot77, the reader script, scripts/pod,
# layouts, HUD assets, the v6l detector weights), manifest.json + cards.json (scripts/pod/make_manifest.py), videos/
# (symlinks to the 484 originals, 86.8 GB; or, with --transcode, 10 fps x264 crf 18 copies, ~29 GB, 96% of unit reads identical to the originals: see decisions.md
# for the read agreement measured on them).
#
# Then (POD = the pod's "SSH over exposed TCP" address from the RunPod console, e.g. root@1.2.3.4 -p 40123):
#   rsync -aL --info=progress2 -e "ssh -p PORT" JOB_DIR/ root@IP:/workspace/pod_job/
#   ssh -p PORT root@IP 'mkdir -p /workspace/bot77 && tar xzf /workspace/pod_job/code.tgz -C /workspace/bot77 &&
#       cd /workspace/bot77 && bash scripts/pod/setup.sh && bash scripts/pod/run.sh 6'
#   (watch)  ssh -p PORT root@IP tail -f /workspace/pod_job/reads.log
#   (back)   rsync -a --info=progress2 -e "ssh -p PORT" root@IP:/workspace/pod_job/reads/ JOB_DIR/reads/
# and on the Mac: uv run --no-sync python scripts/build_bc_live.py --reads JOB_DIR/reads --out
#   /Volumes/LaCie/bot-77-data/bc/shards_t02
set -euo pipefail
cd "$(dirname "$0")/../.."
JOB=${1:-/Volumes/LaCie/bot-77-data/pod_job}
EXTRA=()
[[ "${2:-}" == "--transcode" ]] && EXTRA=(--transcode --crf 18) || EXTRA=(--link)
mkdir -p "$JOB"
uv run --no-sync python scripts/pod/make_manifest.py --out "$JOB" "${EXTRA[@]}"
COPYFILE_DISABLE=1 tar czf "$JOB/code.tgz" --exclude __pycache__ src/bot77 scripts/read_replay_boards.py scripts/pod config/layouts assets/hud \
    data/models/bot77_detector/v6l/best.pt
ls -la "$JOB/code.tgz"
du -shL "$JOB/videos"
