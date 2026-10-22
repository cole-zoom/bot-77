"""Build A's training samples (bot77.moves) from every keyed replay: both players, one row per hand candidate at
each play, into the LanceDB table `move_samples` (2026-10-21). Columns `split` (train / heldback / test / heldout) and
`league` (TV Royale tab: 0 = Ultimate Champion; -1 = Battle Log) let training keep to top ladder.

  uv run python scripts/build_moves.py
"""
import glob, json
from pathlib import Path

import lancedb
import pyarrow as pa

from bot77.holdout import held_out
from bot77.moves import decision_rows

db = lancedb.connect("data/lancedb")
cards_rows = db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).select(
    ["id", "name", "elixir", "type"]).to_arrow().to_pylist()
cards = {c["id"]: c for c in cards_rows}
names = {c["name"]: c["id"] for c in cards_rows}
test = set(json.loads(Path("config/replay_test_games.json").read_text())["matches"])
heldback = set(json.loads(Path("config/heldback_replays.json").read_text())["matches"])
hold = held_out()
league = {}
for f in glob.glob("data/raw/tv_royale/*/runner_log.jsonl"):
    date = Path(f).parent.name
    for line in open(f):
        r = json.loads(line)
        league[f"tv_{date}_r{int(r['i']):02d}"] = r.get("tab", 0)
ev = db.open_table("events"); hud_t = db.open_table("hud_states")


def plays(mid):
    return sorted(({"t": e["t_video"], "card": "__ability" if e["kind"] == "ability" else e["card"]}
                   for e in ev.search().where(f"match_id = '{mid}' AND kind IN ('play', 'ability')").limit(10**4)
                   .select(["t_video", "card", "kind"]).to_arrow().to_pylist()), key=lambda p: p["t"])


def hud(mid):
    return sorted(hud_t.search().where(f"match_id = '{mid}'").limit(10**6).select(
        ["t_video", "t_match", "hand", "next", "elixir", "multiplier"]).to_arrow().to_pylist(), key=lambda r: r["t_video"])


rows = []
files = sorted(glob.glob("data/ground_truth/opponent_truth_replay_*.json"))
for n, f in enumerate(files):
    m = json.loads(Path(f).read_text())["matches"][0]
    top = m.replace("_m", "_top_m", 1) if "_top_" not in m else None
    vid = m.rsplit("_m", 1)[0]
    split = "test" if m in test else "heldout" if m in hold else "heldback" if m in heldback else "train"
    pb, pt = plays(m), plays(top)
    for side, own, opp, h in (("bottom", pb, pt, hud(m)), ("top", pt, pb, hud(top))):
        for r in decision_rows(own, opp, h, cards, names):
            rows.append({**r, "match_id": m, "side": side, "split": split, "league": league.get(vid, -1),
                         "game": f"{m}:{side}"})
    if n % 50 == 0:
        print(f"{n + 1}/{len(files)} games, {len(rows)} rows", flush=True)
db.create_table("move_samples", pa.Table.from_pylist(rows), mode="overwrite")
import collections
dec = {(r["game"], r["decision"]) for r in rows}
print(len(rows), "rows,", len(dec), "decisions;", collections.Counter(r["split"] for r in rows if r["chosen"]))
