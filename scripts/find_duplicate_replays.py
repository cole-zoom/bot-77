"""Replays recorded twice (bot77.replay_dedupe) -> config/duplicate_replays.json {later: first}. Run before
build_sources.py, which gives the later copy split 'duplicate'.

  uv run python scripts/find_duplicate_replays.py
"""
import json
from collections import defaultdict
from pathlib import Path

import lancedb

from bot77.replay_dedupe import duplicates

db = lancedb.connect("data/lancedb")
ids = {m["match_id"] for m in db.open_table("matches").search().where(
    "match_id LIKE 'bl_%' OR match_id LIKE 'tv_%'").select(["match_id", "pov_side"]).limit(10**6).to_arrow().to_pylist()
    if "_top_m" not in m["match_id"]}
plays = defaultdict(list)
for p in db.open_table("plays").search().where("source = 'hand'").select(["game_id", "t_match", "card", "side"]) \
        .limit(10**8).to_arrow().to_pylist():
    if p["game_id"] in ids and p["t_match"] is not None:
        plays[p["game_id"]].append((p["t_match"], str(p["card"]), str(p["side"])))
dup = duplicates(plays)
out = Path("config/duplicate_replays.json")
out.write_text(json.dumps({"why": "the same battle recorded twice; the later id is not used (split 'duplicate')",
                           "replays": dict(sorted(dup.items()))}, indent=1) + "\n")
print(f"{len(plays)} replays, {len(dup)} recorded twice -> {out}")
