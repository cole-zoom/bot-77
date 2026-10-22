"""Board features (bot77.board) and placement spots (bot77.place) on synthetic boards."""
from bot77.board import (FLIP_Y, candidate_features, live_board, main_threat, spell_value, summary_features,
                         to_player_frame)
from bot77.live.state import Unit
from bot77.place import spots

EL = {"Giant": 5, "Minions": 3, "Knight": 3, "Arrows": 3, "Skeletons": 1, "Spear Goblins": 2, "Mini P.E.K.K.A": 4,
      "Musketeer": 4}


def u(card, team, x, y):
    return {"card": card, "team": team, "x": x, "y": y}


def test_top_player_frame_mirrors():
    assert to_player_frame(3.0, 10.0, "top") == (15.0, FLIP_Y - 10.0)
    assert to_player_frame(3.0, 10.0, "top", spell=True) == (15.0, 29.8 - 10.0)
    assert to_player_frame(3.0, 10.0, "bottom") == (3.0, 10.0)


def test_main_threat_is_the_dearest_on_our_half():
    units = [u("Skeletons", "opp", 4, 20), u("Giant", "opp", 4, 17), u("Knight", "opp", 14, 8)]
    assert main_threat(units, EL)["card"] == "Giant"
    assert main_threat([u("Knight", "opp", 14, 8)], EL) is None  # still on their side


def test_summary_counts_each_card_once_per_lane():
    units = [u("Skeletons", "opp", 4, 18), u("Skeletons", "opp", 4.5, 18), u("Skeletons", "opp", 4, 19),
             u("Giant", "opp", 14, 17), u("Minions", "own", 4, 10)]
    f = summary_features(units, {"own_L": 1, "own_R": 1}, EL, {})
    assert f["b_opp_val_our_L"] == 1 and f["b_opp_val_our_R"] == 5 and f["b_opp_tank_our"] == 1
    assert f["b_own_val_attacking"] == 3 and f["b_tower_own_L"] == 1 and f["b_tower_opp_L"] == 0


def test_candidate_vs_board():
    air = [u("Minions", "opp", 4, 18)]
    assert candidate_features("Knight", "troop", air, EL)["c_can_hit_threat"] == 0
    assert candidate_features("Musketeer", "troop", air, EL)["c_can_hit_threat"] == 1
    gobs = [u("Spear Goblins", "opp", 4, 18), u("Spear Goblins", "opp", 4.5, 18)]
    assert spell_value("Arrows", gobs, EL) == 2 / 3
    assert spell_value("Arrows", [u("Knight", "opp", 4, 18)], EL) == 0


def test_live_board_teams_and_towers():
    units, towers = live_board([Unit("Giant", "giant", "enemy", 4, 18, 0.9), Unit(None, "queen-tower", "friendly", 3.5, 20.8, 0.9)])
    assert units == [{"card": "Giant", "team": "opp", "x": 4, "y": 18}] and towers == {"own_L": 1}


def test_spots_anchor_on_the_threat_and_behind_our_tank():
    units = [u("Giant", "opp", 4, 18), u("Giant", "own", 14, 19)]
    kinds = {s["kind"]: (s["x"], s["y"]) for s in spots("Mini P.E.K.K.A", "troop", units, [(9, 20)], EL)}
    assert kinds["threat"] == (4, 18) and kinds["behind_tank"] == (14, 21.5) and "enemy_tower" not in kinds
