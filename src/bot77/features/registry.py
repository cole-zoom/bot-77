"""Every feature the models see, with its group, level and meaning (2026-10-22). Level: `decision` (one value per
moment, shared by its candidates), `card` (per card in hand) or `spot` (per candidate spot). docs/features.md is
generated from this (scripts/features_doc.py) and tests/test_features.py checks that every feature the code produces
is listed here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    name: str
    group: str
    level: str
    doc: str
    since: str  # model version that added it


def _f(group, level, since, items):
    return [Feature(n, group, level, d, since) for n, d in items]


HISTORY = 4
FEATURES: list[Feature] = (
    _f("hand", "card", "A v1", [
        ("cand_card", "the card (id; categorical)"),
        ("cand_cost", "its elixir cost"),
        ("cand_type", "troop 0 / building 1 / spell 2"),
        ("cand_slot", "hand slot 0-3"),
        ("cand_affordable", "elixir >= cost - 0.3"),
        ("cand_waited_s", "seconds since this card was last played (60+ if never)"),
        ("cand_is_cheapest", "cheapest card in hand"),
        ("cand_is_dearest", "most expensive card in hand"),
        ("cand_last_own_gap", "times this player has played the card so far this game"),
    ])
    + _f("elixir / clock", "decision", "A v1", [
        ("elixir", "elixir now"),
        ("t_match", "seconds since the match started"),
        ("multiplier", "elixir multiplier (1 / 2 / 3)"),
        ("next_card", "the next card (id; categorical)"),
    ])
    + _f("history", "decision", "A v1", [
        ("own_since_s", "seconds since the player's last play"),
        ("opp_since_s", "seconds since the opponent's last play"),
        ("opp_seen_n", "different opponent cards seen so far"),
    ] + [(f"own_prev{k}", f"the player's {k}{'st' if k == 1 else 'nd' if k == 2 else 'rd' if k == 3 else 'th'}-last card (id)") for k in range(1, HISTORY + 1)]
      + [(f"opp_prev{k}", f"the opponent's {k}{'st' if k == 1 else 'nd' if k == 2 else 'rd' if k == 3 else 'th'}-last card (id)") for k in range(1, HISTORY + 1)]
      + [(f"opp_prev{k}_dt", f"seconds since the opponent's {k}{'st' if k == 1 else 'nd' if k == 2 else 'rd' if k == 3 else 'th'}-last play") for k in range(1, HISTORY + 1)])
    + _f("rates", "card", "A v1", [
        ("resp_rate", "how often players pick this card right after the opponent's last card (smoothed)"),
        ("own_rate_after", "how often players pick this card right after their own last card (smoothed)"),
        ("elixir_after", "elixir left after playing it"),
    ])
    + _f("rates", "card", "A v2", [
        ("resp_threat", "how often players answer the main threat on the board with this card (smoothed)"),
    ])
    + _f("board", "decision", "A v2", [
        ("b_opp_val_our_half", "elixir of the enemy cards on our half (each card once per lane)"),
        ("b_opp_val_our_L", "... in our left lane"),
        ("b_opp_val_our_R", "... in our right lane"),
        ("b_opp_val_field", "elixir of all enemy cards on the field"),
        ("b_own_val_field", "elixir of our cards on the field"),
        ("b_own_val_attacking", "elixir of our cards on their half"),
        ("b_opp_nearest_y", "how far down the nearest enemy unit is (tiles; 0 = none)"),
        ("b_opp_air_our", "enemy air units on our half"),
        ("b_opp_tank_our", "an enemy tank on our half"),
        ("b_opp_swarm_our", "enemy swarm units on our half"),
        ("b_opp_wincon_field", "an enemy win condition on the field"),
        ("b_own_tank_field", "our tank on the field"),
        ("b_threat_card", "the main threat: the dearest enemy card on our half (id; categorical; -1 none)"),
        ("b_threat_lane", "its lane: left 0 / right 1 (-1 none)"),
        ("b_tower_own_L", "our left princess tower standing"),
        ("b_tower_own_R", "our right princess tower standing"),
        ("b_tower_opp_L", "their left princess tower standing (our left)"),
        ("b_tower_opp_R", "their right princess tower standing"),
    ])
    + _f("card vs board", "card", "A v2", [
        ("c_can_hit_threat", "this card can hit the main threat (air vs ground-only; -1 no threat)"),
        ("c_air_answer", "air is coming and this troop / building hits air"),
        ("c_spell_value", "for spells: elixir of the cards in its best target group / its cost"),
        ("c_cost_vs_threat", "this card's cost minus the threat's"),
        ("c_wincon", "a win condition (the card you rely on for tower damage)"),
        ("c_tank", "a tank"),
        ("c_ground_only", "can't hit air"),
        ("c_air_unit", "a flying unit"),
        ("c_swarm", "many small units"),
        ("c_ranged", "shoots from range"),
        ("c_wincon_under_attack", "a win condition while there's a threat on our half"),
    ])
    + _f("timing", "card", "A v3", [
        ("cand_is_wait", "this candidate is \"wait\" (don't play now)"),
    ])
    + _f("timing", "decision", "A v3", [
        ("n_affordable", "cards in hand affordable now"),
        ("min_cost", "the cheapest card in hand"),
        ("elixir_to_full", "elixir until the bar is full (10)"),
    ])
    + _f("spot", "spot", "place v1", [
        ("s_kind", "what the spot is anchored on: fixed / threat / threat_ahead / opp_unit / behind_tank / spell_group / enemy_tower (categorical)"),
        ("s_x", "spot x (tiles)"), ("s_y", "spot y (tiles)"), ("s_lane", "left 0 / right 1"),
        ("s_mid", "distance from the middle of the arena (x)"), ("s_own_half", "on our half"),
        ("s_d_threat", "distance to the main threat (-1 none)"), ("s_dx_threat", "x offset from the threat"),
        ("s_dy_threat", "y offset from the threat"), ("s_same_lane_threat", "same lane as the threat (-1 none)"),
        ("s_d_opp", "distance to the nearest enemy unit"), ("s_n_opp_near", "enemy units within 2.5 tiles"),
        ("s_val_opp_near", "elixir of the enemy cards within 2.5 tiles"),
        ("s_lane_own_val", "elixir of our cards in this lane"), ("s_lane_opp_val_theirs", "enemy cards in this lane on their half"),
        ("s_lane_opp_val_ours", "enemy cards in this lane on our half"), ("s_lane_opp_tower", "their tower in this lane standing"),
        ("s_lane_own_tower", "our tower in this lane standing"), ("s_d_own", "distance to our nearest unit"),
        ("p_type", "the card's type"), ("p_ranged", "the card shoots from range"), ("p_tank", "the card is a tank"),
        ("p_wincon", "the card is a win condition"), ("p_air", "the card flies"), ("p_ground_only", "the card can't hit air"),
        ("p_swarm", "the card is a swarm"), ("p_card", "the card (id; categorical)"),
    ])
)
BY_NAME = {f.name: f for f in FEATURES}
