"""Placement candidates (2026-10-22): the spots a card could go on a board, and the features of each spot. Fixed spots
(k-means of where players put cards, stored with the model) plus spots anchored on the board: on the main attacker,
just in front of it, on other enemy units, behind our tank, a spell's best group, the enemy towers.
"""
from __future__ import annotations

import math

from bot77.features.board import main_threat
from bot77.features.frames import CROSSED_Y, roles

KINDS = ["fixed", "threat", "threat_ahead", "opp_unit", "behind_tank", "spell_group", "enemy_tower"]
ENEMY_TOWERS = [(3.55, 5.68), (14.5, 5.68)]


def spots(card: str, ctype: str, units: list[dict], fixed: list[tuple[float, float]], elixir: dict[str, int]) -> list[dict]:
    """Candidate spots for this card on this board: [{"kind", "x", "y"}]."""
    R = roles()
    out = [{"kind": "fixed", "x": x, "y": y} for x, y in fixed]
    th = main_threat(units, elixir)
    if th:
        out.append({"kind": "threat", "x": th["x"], "y": th["y"]})
        out.append({"kind": "threat_ahead", "x": th["x"], "y": th["y"] + 1.5})
    opp = sorted((u for u in units if u["team"] == "opp" and u["card"] and u["y"] >= 9.0), key=lambda u: -u["y"])[:6]
    out += [{"kind": "opp_unit", "x": u["x"], "y": u["y"]} for u in opp]
    own_tanks = [u for u in units if u["team"] == "own" and u["card"] in R["tanks"]]
    if own_tanks:
        t = max(own_tanks, key=lambda u: u["y"])
        out.append({"kind": "behind_tank", "x": t["x"], "y": t["y"] + 2.5})
    if ctype == "spell":
        kills = R["spell_kills"].get(card, set())
        us = [u for u in units if u["team"] == "opp" and (u["card"] in kills or not kills)]
        best = max(us, key=lambda u: sum(1 for v in us if math.dist((u["x"], u["y"]), (v["x"], v["y"])) <= 2.5), default=None)
        if best:
            near = [v for v in us if math.dist((best["x"], best["y"]), (v["x"], v["y"])) <= 2.5]
            out.append({"kind": "spell_group", "x": sum(v["x"] for v in near) / len(near), "y": sum(v["y"] for v in near) / len(near)})
        out += [{"kind": "enemy_tower", "x": x, "y": y} for x, y in ENEMY_TOWERS]
    return out


def spot_features(s: dict, card: str, ctype: str, units: list[dict], elixir: dict[str, int], towers: dict | None = None) -> dict:
    R = roles()
    towers = towers or {}
    left = s["x"] < 9
    lane_u = [u for u in units if u["card"] and (u["x"] < 9) == left]

    def val(us):
        return float(sum(elixir.get(c, 0) for c in {u["card"] for u in us}))

    own = [u for u in units if u["team"] == "own" and u["card"]]
    lane = {"s_lane_own_val": val([u for u in lane_u if u["team"] == "own"]),  # which lane: where our push is,
            "s_lane_opp_val_theirs": val([u for u in lane_u if u["team"] == "opp" and u["y"] < 13.0]),  # theirs,
            "s_lane_opp_val_ours": val([u for u in lane_u if u["team"] == "opp" and u["y"] >= 13.0]),  # what's coming,
            "s_lane_opp_tower": float(towers.get("opp_L" if left else "opp_R", 0)),  # the towers still up
            "s_lane_own_tower": float(towers.get("own_L" if left else "own_R", 0)),
            "s_d_own": min((math.dist((s["x"], s["y"]), (u["x"], u["y"])) for u in own), default=30.0)}
    th = main_threat(units, elixir)
    opp = [u for u in units if u["team"] == "opp" and u["card"]]
    near = [u for u in opp if math.dist((s["x"], s["y"]), (u["x"], u["y"])) <= 2.5]
    d_opp = min((math.dist((s["x"], s["y"]), (u["x"], u["y"])) for u in opp), default=30.0)
    return {"s_kind": KINDS.index(s["kind"]), "s_x": s["x"], "s_y": s["y"], "s_lane": float(s["x"] >= 9),
            "s_mid": abs(s["x"] - 9), "s_own_half": float(s["y"] >= CROSSED_Y),
            "s_d_threat": math.dist((s["x"], s["y"]), (th["x"], th["y"])) if th else -1.0,
            "s_dx_threat": s["x"] - th["x"] if th else 0.0, "s_dy_threat": s["y"] - th["y"] if th else 0.0,
            "s_same_lane_threat": float((s["x"] < 9) == (th["x"] < 9)) if th else -1.0,
            "s_d_opp": d_opp, "s_n_opp_near": float(len(near)),
            "s_val_opp_near": float(sum(elixir.get(c, 0) for c in {u["card"] for u in near})),
            "p_type": {"troop": 0, "building": 1, "spell": 2}.get(ctype, 0),
            "p_ranged": float(card in R["ranged"]), "p_tank": float(card in R["tanks"]),
            "p_wincon": float(card in R["win_conditions"]), "p_air": float(card in R["air"]),
            "p_ground_only": float(card in R["ground_only"]), "p_swarm": float(card in R["swarm"]), **lane}
