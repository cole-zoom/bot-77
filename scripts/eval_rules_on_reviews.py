"""Offline check of the playbook rules (2026-10-22): replay the reviewed bot plays (bot_review_*_{cole,claude}.json)
through the rules policy and count what it would now do differently.

The states are rebuilt from bot_actions: hand, elixir, time, and the enemy units on our half that were logged with
each play (threats). Our own units and enemies on their half weren't logged, so "support the Giant" can't fire here,
and enemy plays come only from what had crossed.

  uv run python scripts/eval_rules_on_reviews.py [--show]
"""
import argparse, glob, json
from collections import Counter

import lancedb

from bot77.live.policy import Policy
from bot77.live.state import LiveState, Unit

ap = argparse.ArgumentParser(); ap.add_argument("--show", action="store_true", help="print every reviewed play")
a = ap.parse_args()

db = lancedb.connect("data/lancedb")
cards = db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).to_arrow().to_pylist()
rev = {}
for f in sorted(glob.glob("data/ground_truth/reviews/bot_review_*_c*.json")):
    for p in json.loads(open(f).read())["plays"]:
        rev[p["id"]] = p  # Cole's and Claude's (approved) are disjoint
rows = {f"{r['game_id']}:{r['k']}": r for r in db.open_table("bot_actions").to_arrow().to_pylist()}
games = {}
for r in rows.values():
    games.setdefault(r["game_id"], []).append(r)

out = Counter()
for pid, p in sorted(rev.items()):
    r = rows.get(pid)
    if r is None:
        continue
    pol = Policy(cards, playbook=False)
    for q in sorted(games[r["game_id"]], key=lambda q: q["t"]):  # our earlier plays in this game
        if q["t"] < r["t"]:
            pol.record_own(q["t"], q["card"])
    units = [Unit(c, c, "enemy", x, y, 0.9) for c, x, y in json.loads(r["threats"] or "[]")]
    tl = r["time_left_s"]
    st = LiveState(r["t"], json.loads(r["hand"]), None, r["elixir"], tl, None if tl is None else 180 - tl, 1, "regulation", units)
    act = pol.decide(st, [])
    new = "wait (no play)" if act is None else act.card
    moved = act is not None and act.card == r["card"] and ((act.x - r["x_tiles"]) ** 2 + (act.y - r["y_tiles"]) ** 2) ** 0.5 > 2
    if p["verdict"] == "bad":
        fixed = new != r["card"] or moved
        out["bad"] += 1; out["bad changed"] += fixed
        if p.get("card") and p["card"] != r["card"]:  # the reviewer named a different card / wait
            out["bad with a named fix"] += 1; out["new = reviewer's card"] += new == p["card"]
    else:
        out["ok"] += 1; out["ok same card"] += new == r["card"]; out["ok same spot"] += new == r["card"] and not moved
        out["ok -> wait"] += act is None
    if a.show:
        print(f"{pid:28s} {p['verdict']:3s} bot {r['card']:15s} -> {new:15s}{' (moved)' if moved else ''}  "
              f"[{act.reason if act else ''}]  reviewer: {p.get('card')}")
print(f"bad plays the rules change: {out['bad changed']}/{out['bad']}  "
      f"(new move = the reviewer's fix: {out['new = reviewer\'s card']}/{out['bad with a named fix']})")
print(f"ok plays: same card {out['ok same card']}/{out['ok']} (same spot too: {out['ok same spot']}), "
      f"now waits {out['ok -> wait']}")
