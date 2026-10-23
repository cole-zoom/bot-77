"""Pair the bot's games with their Battle Log replays (phase 6, 2026-10-22). A bot game logs its own plays (card, game
clock); its replay (source replay_bot) has the bot's exact hand plays as the bottom player plus the opponent's, and
detections every 0.25 s. A pair is accepted when most of the bot's plays reappear in the replay's bottom hand at the
same game clock (one-to-one, best agreement first). Then every bot play (and its review) can be read on the replay's
complete moment: exact opponent history and the full board, which the bot's own log lacks before 2026-10-22.

-> LanceDB table `bot_pairs` (bot game, replay match, agreement, plays) and `bot_play_moments` (bot game, k -> the
   replay's play moment).

  uv run python scripts/pair_bot_replays.py
"""
from collections import defaultdict

import lancedb
import pyarrow as pa

MIN_AGREE, CLOCK_TOL = 0.7, 1.5
db = lancedb.connect("data/lancedb")
acts = defaultdict(list)
for a in db.open_table("bot_actions").to_arrow().to_pylist():
    if a["time_left_s"] is not None:
        acts[a["game_id"]].append({"k": a["k"], "card": a["card"], "tm": 180 - a["time_left_s"]})
reps = [m["match_id"] for m in db.open_table("matches").search().where("source = 'replay_bot' AND pov_side = 'bottom'")
        .select(["match_id"]).limit(10**5).to_arrow().to_pylist()]
plays = defaultdict(list)
for p in db.open_table("plays").search().where("source = 'hand' AND side = 'bottom'").select(["game_id", "t", "t_match", "card"]) \
        .limit(10**7).to_arrow().to_pylist():
    if p["game_id"] in set(reps):
        plays[p["game_id"]].append(p)


def agree_at(bot, rep, off):
    used, hits = set(), []
    for b in bot:
        for i, p in enumerate(rep):
            if i not in used and p["card"] == b["card"] and p["t_match"] is not None and abs(p["t_match"] - b["tm"] - off) <= CLOCK_TOL:
                used.add(i); hits.append((b, p)); break
    return len(hits) / max(len(bot), 1), hits


def agree(bot, rep):
    """Share of the bot's plays found in the replay (same card, game clock within CLOCK_TOL after the best constant
    offset: the replay's match clock runs ~3-5 s ahead of the bot's clock reads, 2026-10-22), and the matches."""
    return max((agree_at(bot, rep, off / 2) for off in range(-4, 17)), key=lambda x: x[0])


cands = []
for g, bot in acts.items():
    for r in reps:
        a, hits = agree(bot, plays[r])
        if a >= MIN_AGREE:
            cands.append((a, g, r, hits))
pairs, used_g, used_r, links = [], set(), set(), []
for a, g, r, hits in sorted(cands, key=lambda c: -c[0]):
    if g in used_g or r in used_r:
        continue
    used_g.add(g); used_r.add(r)
    pairs.append({"game_id": g, "replay_match_id": r, "agree": a, "bot_plays": len(acts[g]), "matched": len(hits)})
    for b, p in hits:
        links.append({"game_id": g, "k": b["k"], "replay_match_id": r, "replay_t": p["t"]})

# the replay's play moment for each linked bot play
moms = defaultdict(dict)
for m in db.open_table("moments").search().where("source = 'replay_bot' AND kind = 'play' AND side = 'bottom'") \
        .select(["moment_id", "game_id", "t"]).limit(10**6).to_arrow().to_pylist():
    moms[m["game_id"]][round(m["t"], 2)] = m["moment_id"]
for l in links:
    l["replay_moment_id"] = moms[l["replay_match_id"]].get(round(l["replay_t"], 2))
db.create_table("bot_pairs", pa.Table.from_pylist(pairs), mode="overwrite")
db.create_table("bot_play_moments", pa.Table.from_pylist(links), mode="overwrite")
print(f"bot games {len(acts)}, bot replays {len(reps)}: paired {len(pairs)} "
      f"(agreement median {sorted(p['agree'] for p in pairs)[len(pairs) // 2]:.0%}); "
      f"bot plays linked {len(links)}, with a replay moment {sum(1 for l in links if l['replay_moment_id'])}")
