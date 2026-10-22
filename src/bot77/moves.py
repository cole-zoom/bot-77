"""Step A, v1 (2026-10-21): the pro-move predictor's data. At every play of either player in a keyed replay, the
player's view right before it (their 4-card hand, next card, elixir, the clock, the multiplier, their own and the
opponent's recent plays, the opponent cards seen so far) and the card they chose. A predicts which of the 4 hand
cards is played; "when" and "where" come later (decisions.md).

Replays show both hands, so both players' sides are exact: the bottom player from the replay processed with
replay_bottom (<id>), the top player from <id>_top. The opponent's history can also be swapped for C's detected
plays (opponent_events) to measure how much C's misses and fakes cost A.
"""
from __future__ import annotations

import bisect

HISTORY = 4  # recent plays kept per side
CAND_FEATURES = ["cand_card", "cand_cost", "cand_type", "cand_slot", "cand_affordable", "cand_waited_s", "cand_is_cheapest",
                 "cand_is_dearest", "cand_last_own_gap"]
CTX_FEATURES = ["elixir", "t_match", "multiplier", "own_since_s", "opp_since_s", "next_card"] + \
    [f"own_prev{k}" for k in range(1, HISTORY + 1)] + [f"opp_prev{k}" for k in range(1, HISTORY + 1)] + \
    [f"opp_prev{k}_dt" for k in range(1, HISTORY + 1)] + ["opp_seen_n"]
TYPES = {"troop": 0, "building": 1, "spell": 2}


def hud_before(hud: list[dict], t: float, lag: float = 0.25) -> dict | None:
    """The last HUD read at or before t - lag (the hand before the card left it)."""
    ts = [h["t_video"] for h in hud]
    i = bisect.bisect_right(ts, t - lag) - 1
    return hud[i] if i >= 0 else None


def decision_rows(own_plays: list[dict], opp_plays: list[dict], hud: list[dict], cards: dict[int, dict],
                  names: dict[str, int]) -> list[dict]:
    """One row per (play, hand candidate). own_plays / opp_plays: [{"t", "card"}] sorted (card names; abilities as
    "__ability"); hud: this player's HUD rows sorted by t_video (hand = 4 card ids, next, elixir, t_match,
    multiplier); cards: id -> {"name", "elixir", "type"}. Rows carry "decision" (play index) and "chosen" (0/1)."""
    rows = []
    last_played: dict[int, float] = {}
    for k, p in enumerate(own_plays):
        if p["card"] in (None, "__ability", "Mirror") or names.get(p["card"]) is None:
            if p["card"] and names.get(p["card"]) is not None:
                last_played[names[p["card"]]] = p["t"]
            continue
        h = hud_before(hud, p["t"])
        cid = names[p["card"]]
        if h is None or not h["hand"] or cid not in h["hand"] or any(c is None or c < 0 for c in h["hand"]):
            last_played[cid] = p["t"]
            continue  # the hand read doesn't show the played card: no clean decision state
        own_prev = [names.get(q["card"], -1) for q in own_plays[:k] if q["card"] not in (None,)][-HISTORY:][::-1]
        opp_before = [q for q in opp_plays if q["t"] < p["t"] - 0.1]
        opp_prev = [names.get(q["card"], -1) for q in opp_before][-HISTORY:][::-1]
        opp_dt = [p["t"] - q["t"] for q in opp_before][-HISTORY:][::-1]
        ctx = {"elixir": h["elixir"] if h["elixir"] is not None else -1.0, "t_match": h["t_match"] if h["t_match"] is not None else -1.0,
               "multiplier": h["multiplier"] or 1, "own_since_s": p["t"] - own_plays[k - 1]["t"] if k else 99.0,
               "opp_since_s": opp_dt[0] if opp_dt else 99.0, "next_card": h["next"] if h["next"] is not None else -1,
               "opp_seen_n": len({q["card"] for q in opp_before if q["card"]})}
        for j in range(HISTORY):
            ctx[f"own_prev{j + 1}"] = own_prev[j] if j < len(own_prev) else -1
            ctx[f"opp_prev{j + 1}"] = opp_prev[j] if j < len(opp_prev) else -1
            ctx[f"opp_prev{j + 1}_dt"] = opp_dt[j] if j < len(opp_dt) else 99.0
        costs = [cards.get(c, {}).get("elixir") or 0 for c in h["hand"]]
        for slot, c in enumerate(h["hand"]):
            info = cards.get(c, {})
            cost = info.get("elixir") or 0
            rows.append({**ctx, "decision": k, "t": p["t"], "chosen": int(c == cid), "cand_card": c, "cand_cost": cost,
                         "cand_type": TYPES.get(info.get("type"), 0), "cand_slot": slot,
                         "cand_affordable": float(ctx["elixir"] >= cost - 0.3),
                         "cand_waited_s": p["t"] - last_played.get(c, -60.0),
                         "cand_is_cheapest": float(cost == min(costs)), "cand_is_dearest": float(cost == max(costs)),
                         "cand_last_own_gap": sum(1 for q in own_plays[:k] if names.get(q["card"]) == c)})
        last_played[cid] = p["t"]
    return rows
