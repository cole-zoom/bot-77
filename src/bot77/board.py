"""The board for the move predictor (A v2, 2026-10-22): the units on the field at a decision, in the deciding player's
frame, and the features A gets from them. Shared by training (scripts/build_board_states.py, scripts/train_moves.py)
and live play (bot77.live.policy), so both compute the same numbers.

Frame: arena tiles as on the bottom player's screen (x 0-18, y down, box centres), "own" = the deciding player. The
top player's board is turned around: x -> 18 - x, y -> FLIP_Y - y. FLIP_Y is not 2 x the river (29.8): unit boxes
sit ~1 tile above the unit's feet, so the mirror line of box centres is ~13.75 (both players' answer positions in
the interactions table differ by a median 2.3 tiles with 29.8; 2026-10-22).

Two kinds of features:
- board summary (same for the 4 candidates): enemy elixir on our half per lane, how close the nearest attacker is,
  air / tank / swarm on our half, our units, towers standing, the main threat's card;
- candidate vs board: can this card hit the main threat, does it hit air when air is coming, what a spell would hit
  (elixir of the cards in its best target group / its cost), how often players answer the threat's card with it.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

FLIP_Y = 27.5
CROSSED_Y = 15.2  # on our half (as the playbook rules)
ATTACKING_Y = 13.0  # our units past this are on their half
TOWER_CLASSES = ("queen-tower", "cannoneer-tower", "dagger-duchess-tower")


@lru_cache(maxsize=1)
def roles() -> dict:
    r = {k: set(v) for k, v in yaml.safe_load(Path("config/card_roles.yaml").read_text()).items()}
    spells = yaml.safe_load(Path("config/playbook_rules.yaml").read_text())["spells"]
    r["spell_kills"] = {c: set(g["kills"]) for g in spells.values() if isinstance(g, dict) for c in g["cards"]}
    r["spell_radius"] = spells["radius"]
    return r


def to_player_frame(x: float, y: float, side: str, spell: bool = False) -> tuple[float, float]:
    """Spells have no sprite above their spot: they mirror about the river itself (y 14.9)."""
    return (x, y) if side == "bottom" else (18.0 - x, (2 * 14.9 if spell else FLIP_Y) - y)


def summary_features(units: list[dict], towers: dict, elixir: dict[str, int], ids: dict[str, int]) -> dict:
    """units: [{"card", "team": own|opp, "x", "y"}] in the player's frame; towers: {"own_L", "own_R", "opp_L", "opp_R"}
    -> 1 standing / 0 not seen."""
    R = roles()
    opp = [u for u in units if u["team"] == "opp" and u["card"]]
    own = [u for u in units if u["team"] == "own" and u["card"]]
    ours = [u for u in opp if u["y"] >= CROSSED_Y]

    def val(us):  # each card counted once per lane (3 Skeletons are one 1-elixir card)
        return float(sum(elixir.get(card, 0) for card, _ in {(u["card"], u["x"] < 9) for u in us}))

    th = main_threat(units, elixir)
    f = {"b_opp_val_our_half": val(ours), "b_opp_val_our_L": val([u for u in ours if u["x"] < 9]),
         "b_opp_val_our_R": val([u for u in ours if u["x"] >= 9]), "b_opp_val_field": val(opp),
         "b_own_val_field": val(own), "b_own_val_attacking": val([u for u in own if u["y"] < ATTACKING_Y]),
         "b_opp_nearest_y": max((u["y"] for u in opp), default=0.0),
         "b_opp_air_our": float(sum(u["card"] in R["air"] for u in ours)),
         "b_opp_tank_our": float(any(u["card"] in R["tanks"] for u in ours)),
         "b_opp_swarm_our": float(sum(u["card"] in R["swarm"] for u in ours)),
         "b_opp_wincon_field": float(any(u["card"] in R["win_conditions"] for u in opp)),
         "b_own_tank_field": float(any(u["card"] in R["tanks"] for u in own)),
         "b_threat_card": ids.get(th["card"], -1) if th else -1,
         "b_threat_lane": (0.0 if th["x"] < 9 else 1.0) if th else -1.0}
    for k in ("own_L", "own_R", "opp_L", "opp_R"):
        f[f"b_tower_{k}"] = float(towers.get(k, 0))
    return f


def main_threat(units: list[dict], elixir: dict[str, int]) -> dict | None:
    """The attacker to answer: the most expensive enemy card on our half (closest first on ties), else none."""
    ours = [u for u in units if u["team"] == "opp" and u["card"] and u["y"] >= CROSSED_Y]
    return max(ours, key=lambda u: (elixir.get(u["card"], 0), u["y"]), default=None)


def spell_value(spell: str, units: list[dict], elixir: dict[str, int]) -> float:
    """Elixir of the distinct enemy cards in the best group this spell kills, over the spell's cost (-1: unknown
    spell)."""
    R = roles()
    kills = R["spell_kills"].get(spell)
    if kills is None:
        return -1.0
    us = [u for u in units if u["team"] == "opp" and u["card"] in kills]
    best = 0.0
    for u in us:
        near = {v["card"] for v in us if (v["x"] - u["x"]) ** 2 + (v["y"] - u["y"]) ** 2 <= R["spell_radius"] ** 2}
        best = max(best, sum(elixir.get(c, 0) for c in near))
    return best / max(elixir.get(spell, 1), 1)


def candidate_features(card: str, ctype: str, units: list[dict], elixir: dict[str, int]) -> dict:
    R = roles()
    th = main_threat(units, elixir)
    air_coming = any(u["team"] == "opp" and u["card"] in R["air"] and u["y"] >= CROSSED_Y for u in units)
    hits_air = card not in R["ground_only"]
    return {"c_can_hit_threat": -1.0 if th is None else float(not (th["card"] in R["air"] and not hits_air)),
            "c_air_answer": float(air_coming and hits_air and ctype != "spell"),
            "c_spell_value": spell_value(card, units, elixir) if ctype == "spell" else 0.0,
            "c_cost_vs_threat": (elixir.get(card, 0) - elixir.get(th["card"], 0)) if th else 0.0,
            "c_wincon": float(card in R["win_conditions"]), "c_tank": float(card in R["tanks"]),
            "c_ground_only": float(not hits_air), "c_air_unit": float(card in R["air"]),
            "c_swarm": float(card in R["swarm"]), "c_ranged": float(card in R["ranged"]),
            "c_wincon_under_attack": float(card in R["win_conditions"] and th is not None)}


SUMMARY_FEATURES = ["b_opp_val_our_half", "b_opp_val_our_L", "b_opp_val_our_R", "b_opp_val_field", "b_own_val_field",
                    "b_own_val_attacking", "b_opp_nearest_y", "b_opp_air_our", "b_opp_tank_our", "b_opp_swarm_our",
                    "b_opp_wincon_field", "b_own_tank_field", "b_threat_card", "b_threat_lane",
                    "b_tower_own_L", "b_tower_own_R", "b_tower_opp_L", "b_tower_opp_R"]
CAND_BOARD_FEATURES = ["c_can_hit_threat", "c_air_answer", "c_spell_value", "c_cost_vs_threat", "c_wincon", "c_tank",
                       "c_ground_only", "c_air_unit", "c_swarm", "c_ranged", "c_wincon_under_attack"]


def live_board(units) -> tuple[list[dict], dict]:
    """bot77.live Units (the bot is always the bottom player) -> (units, towers) in the board format."""
    out, towers = [], {}
    for u in units:
        team = "own" if u.team == "friendly" else "opp"
        if u.cls in TOWER_CLASSES:
            towers[f"{team}_{'L' if u.x < 9 else 'R'}"] = 1
        elif u.card:
            out.append({"card": u.card, "team": team, "x": u.x, "y": u.y})
    return out, towers


def add_board_features(rows: list[dict], units: list[dict], towers: dict, cards: dict[int, dict]) -> list[dict]:
    """Add the board features to one decision's candidate rows (bot77.moves.candidates / decision_rows; cand_card is a
    card id). cards: id -> {"name", "elixir", "type"}."""
    elixir = {c["name"]: c.get("elixir") or 0 for c in cards.values()}
    ids = {c["name"]: i for i, c in cards.items()}
    summ = summary_features(units, towers, elixir, ids)
    for r in rows:
        c = cards.get(r["cand_card"], {})
        r.update(summ)
        r.update(candidate_features(c.get("name"), c.get("type"), units, elixir))
    return rows
