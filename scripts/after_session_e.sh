#!/bin/zsh
# Session E after session D's chain (one writer to the events table at a time).
cd "$(dirname "$0")/.."
until grep -q "SESSION_cole_2026-10-09_d_DONE" data/logs/run_all.log; do sleep 60; done
scripts/after_session.sh cole_2026-10-09_e data/logs/run_all_jobs_c1009e.json
