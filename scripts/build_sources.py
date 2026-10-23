"""Phase 1a of docs/training-architecture.md (2026-10-22): the source columns on `matches`, and the bot's games as
matches / videos.

Adds to every match (the existing columns are kept):
  source        replay_tv / replay_bl / replay_bot / pov_cole / pov_friend / youtube / bot
                (replay_bot: a Battle Log replay of the bot's own game -- its bottom player is the bot, so it is never
                imitation data; the bot's deck on a day the bot played)
  pov_side      bottom / top (a replay is stored twice: once per player's hand, the top one as <id>_top_m01)
  pair_match_id the same game from the other player's side (replays), else null
  truth_set     replay (both hands, exact) / auto (paired POVs) / live (POV keyed by its replay) / none
  split         heldout / test / heldback / dev / train -- once, here, from the config lists (precedence in that order;
                a replay's top side follows its bottom side). Replaces the per-script logic (bot77.holdout, build_*).
  league        TV Royale tab (0 = Ultimate Champion, 1.. down the leagues), -1 for Battle Log games, null otherwise
  result        win / loss for the bot's games (from bot_games / data/live/game_results.json), null otherwise
Bot games (bot_games) become rows of `matches` and `videos` (their screen recordings, layout bluestacks_live).

  uv run python scripts/build_sources.py
"""
import glob, json
from collections import Counter
from pathlib import Path

import lancedb
import pyarrow as pa

db = lancedb.connect("data/lancedb")
cfg = lambda f: json.loads(Path(f"config/{f}").read_text())
live = cfg("live_test_games.json")
HELDOUT = set(cfg("benchmark_holdout.json")["matches"]) | set(live["matches"]) | set(live["replays"])
TEST = set(cfg("replay_test_games.json")["matches"])
HELDBACK = set(cfg("heldback_replays.json")["matches"])
truth = {}
for f in glob.glob("data/ground_truth/opponent_truth_*.json"):
    kind = Path(f).name.split("_")[2]  # replay / auto / live; the benchmark's own truth file is named by video id
    kind = kind if kind in ("replay", "auto", "live") else "benchmark"
    for m in json.loads(Path(f).read_text())["matches"]:
        truth[m] = kind
league = {}
for f in glob.glob("data/raw/tv_royale/*/runner_log.jsonl"):
    for line in open(f):
        r = json.loads(line)
        league[f"tv_{Path(f).parent.name}_r{int(r['i']):02d}"] = r.get("tab", 0)

matches = db.open_table("matches").to_arrow().to_pylist()
videos = db.open_table("videos").to_arrow().to_pylist()
bot_games = db.open_table("bot_games").to_arrow().to_pylist()
results = json.loads(Path("data/live/game_results.json").read_text()) if Path("data/live/game_results.json").exists() else {}
for g in bot_games:
    if g.get("result") in ("win", "loss"):
        results[g["game_id"]] = g["result"]
results = {k: v for k, v in results.items() if v in ("win", "loss")}

# the bot's games as videos + matches
have_v = {v["video_id"] for v in videos}
for g in bot_games:
    gid = g["game_id"]
    rec = Path(f"data/raw/live_games/{gid}/game.mkv")
    if gid not in have_v:
        videos.append({**{k: None for k in videos[0]}, "video_id": gid, "creator": "bot-77", "layout_id": "bluestacks_live",
                       "mode": "ladder", "path": str(rec) if rec.exists() else None, "width": 1080, "height": 1920})
have_m = {m["match_id"] for m in matches}
for g in bot_games:
    if g["game_id"] not in have_m:
        matches.append({**{k: None for k in matches[0]}, "match_id": g["game_id"], "video_id": g["game_id"], "index": 1,
                        "t_start": g.get("t_start"), "t_end": g.get("t_end"),
                        "deck": json.loads(g["deck"]) if g.get("deck") else None})


bot_decks = {frozenset(json.loads(g["deck"])) for g in bot_games if g.get("deck") and json.loads(g["deck"])}
bot_days = {g["game_id"][4:14] for g in bot_games}
deck_of = {m["match_id"]: m.get("deck") for m in matches}


def is_bot_replay(m: str) -> bool:
    b = bottom_of(m)
    d = deck_of.get(b)
    return m.startswith("bl_") and m[3:13] in bot_days and bool(d) and any(set(d) <= dk for dk in bot_decks)


def source(m: str) -> str:
    if is_bot_replay(m):
        return "replay_bot"
    return ("replay_tv" if m.startswith("tv_") else "replay_bl" if m.startswith("bl_") else "pov_cole" if m.startswith("cole_")
            else "pov_friend" if m.startswith("friend_") else "bot" if m.startswith("bot_") else "youtube")


def bottom_of(m: str) -> str:
    return m.replace("_top_m", "_m", 1)


def split(m: str) -> str:
    b = bottom_of(m)
    for name, s in (("heldout", HELDOUT), ("test", TEST), ("heldback", HELDBACK)):
        if m in s or b in s:
            return name
    return "dev" if truth.get(m) == "auto" else "train"


ids = {m["match_id"] for m in matches}
for m in matches:
    mid = m["match_id"]
    top = "_top_m" in mid
    pair = bottom_of(mid) if top else mid.replace("_m", "_top_m", 1)
    m.update({"source": source(mid), "pov_side": "top" if top else "bottom",
              "pair_match_id": pair if pair in ids and pair != mid else None,
              "truth_set": truth.get(mid) or truth.get(bottom_of(mid)) or "none", "split": split(mid),
              "league": league.get(mid.rsplit("_m", 1)[0].replace("_top", "")) if mid.startswith("tv_") else (-1 if mid.startswith("bl_") else None),
              "result": results.get(mid)})

db.create_table("videos", pa.Table.from_pylist(videos), mode="overwrite")
db.create_table("matches", pa.Table.from_pylist(matches), mode="overwrite")

# checks against the config lists
by = Counter((m["split"], m["source"]) for m in matches)
print("matches:", len(matches), " videos:", len(videos))
for k, v in sorted(by.items()):
    print(f"  {k[0]:9s} {k[1]:11s} {v}")
for name, s in (("heldout", HELDOUT), ("test", TEST), ("heldback", HELDBACK)):
    got = {m["match_id"] for m in matches if m["split"] == name and m["pov_side"] == "bottom"}
    print(f"{name}: config {len(s)}, in matches {len(s & ids)}, bottom rows with that split {len(got)}, missing from table {sorted(s - ids)[:5]}")
print("truth sets:", Counter(m["truth_set"] for m in matches), " results:", Counter(m["result"] for m in matches if m["result"]))
