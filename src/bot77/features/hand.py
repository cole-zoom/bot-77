"""Hand and history features (A v1, 2026-10-21): for one decision, the context (elixir, clock, multiplier, next card,
both players' recent plays) and, per card in hand, the candidate features (cost, type, slot, affordable, how long
since it was last played, cheapest / dearest, how often played this game). Shared by training and live play.
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


def candidates(h: dict, now: float, own_before: list[dict], opp_plays: list[dict], last_played: dict[int, float],
               cards: dict[int, dict], names: dict[str, int]) -> list[dict]:
    """The 4 candidate rows for one decision: the player's HUD read `h` (hand, next, elixir, t_match, multiplier), the
    time `now`, their earlier plays, the opponent's plays (those before now - 0.1 s are used) and when each card was
    last played. Shared by training (decision_rows) and live play (bot77.live), so both see the same features."""
    own_prev = [names.get(q["card"], -1) for q in own_before if q["card"] not in (None,)][-HISTORY:][::-1]
    opp_before = [q for q in opp_plays if q["t"] < now - 0.1]
    opp_prev = [names.get(q["card"], -1) for q in opp_before][-HISTORY:][::-1]
    opp_dt = [now - q["t"] for q in opp_before][-HISTORY:][::-1]
    ctx = {"elixir": h["elixir"] if h["elixir"] is not None else -1.0, "t_match": h["t_match"] if h["t_match"] is not None else -1.0,
           "multiplier": h["multiplier"] or 1, "own_since_s": now - own_before[-1]["t"] if own_before else 99.0,
           "opp_since_s": opp_dt[0] if opp_dt else 99.0, "next_card": h["next"] if h["next"] is not None else -1,
           "opp_seen_n": len({q["card"] for q in opp_before if q["card"]})}
    for j in range(HISTORY):
        ctx[f"own_prev{j + 1}"] = own_prev[j] if j < len(own_prev) else -1
        ctx[f"opp_prev{j + 1}"] = opp_prev[j] if j < len(opp_prev) else -1
        ctx[f"opp_prev{j + 1}_dt"] = opp_dt[j] if j < len(opp_dt) else 99.0
    costs = [cards.get(c, {}).get("elixir") or 0 for c in h["hand"]]
    out = []
    for slot, c in enumerate(h["hand"]):
        info = cards.get(c, {})
        cost = info.get("elixir") or 0
        out.append({**ctx, "cand_card": c, "cand_cost": cost, "cand_type": TYPES.get(info.get("type"), 0), "cand_slot": slot,
                    "cand_affordable": float(ctx["elixir"] >= cost - 0.3),
                    "cand_waited_s": now - last_played.get(c, -60.0),
                    "cand_is_cheapest": float(cost == min(costs)), "cand_is_dearest": float(cost == max(costs)),
                    "cand_last_own_gap": sum(1 for q in own_before if names.get(q["card"]) == c)})
    return out


# ---- timing (A v3, 2026-10-22): "wait" as a fifth candidate, and what makes now a good time to play
WAIT_ID = 1  # a pseudo card id for the "wait" candidate (real card ids are 26000000+)
TIMING_FEATURES = ["cand_is_wait", "n_affordable", "min_cost", "elixir_to_full"]


def timing_context(h: dict, cards: dict[int, dict]) -> dict:
    """Decision-level timing features: how many cards in hand are affordable, the cheapest card's cost, elixir to full."""
    costs = [cards.get(c, {}).get("elixir") or 0 for c in h["hand"]]
    el = h["elixir"] if h["elixir"] is not None else -1.0
    return {"n_affordable": float(sum(el >= c - 0.3 for c in costs)), "min_cost": float(min(costs) if costs else 0),
            "elixir_to_full": 10.0 - el if el >= 0 else -1.0}


def wait_row(base: dict) -> dict:
    """The "wait" candidate: the decision's context (and board summary) with neutral card features."""
    r = dict(base)
    r.update({"cand_card": WAIT_ID, "cand_cost": 0, "cand_type": 3, "cand_slot": 4, "cand_affordable": 1.0,
              "cand_waited_s": 0.0, "cand_is_cheapest": 0.0, "cand_is_dearest": 0.0, "cand_last_own_gap": 0,
              "cand_is_wait": 1.0})
    for k in ("c_can_hit_threat", "c_air_answer", "c_spell_value", "c_cost_vs_threat", "c_wincon", "c_tank", "c_ground_only",
              "c_air_unit", "c_swarm", "c_ranged", "c_wincon_under_attack"):
        if k in r:
            r[k] = 0.0
    return r
