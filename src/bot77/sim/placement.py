"""Where a card can go on the engine board, and how big it is there, read from the engine (P0.1 of
docs/plan-sim-1to1.md, 2026-10-29: the play page's tile grid).

Engine board: 18 x 34 cells, cell i covers [i - 0.5, i + 0.5] (bot77.sim.geometry). A play goes to exactly the
(x, y) it is given (ClashEnv::step / runSelfPlayTicks pass team 0's target straight to GameManager::playCard), so the
play page sends integer cell centres and the legality of a cell is the engine's own predicate at that centre:
`ClashRoyaleEnv.is_valid_placement` (GameManager::isValidPlacement: bounds, the back-row dead zone, the footprint
against the arena edge, own half for non-spells, clear of buildings and towers).

What the engine does NOT have: the pocket. Taking an enemy princess tower does not open any cell
(fork test python_ai/tests/test_placement_legality_is_static.py pins that), so the page shows only what the engine
will accept, and the pocket is a known gap of the engine, not of the page.
"""
from __future__ import annotations

import clash_royale_env as cre
import numpy as np

W, H = cre.ClashRoyaleEnv.BOARD_WIDTH, cre.ClashRoyaleEnv.BOARD_HEIGHT
OWN_HALF_RIVER_BUFFER = 0.5  # GameManager::OWN_HALF_RIVER_BUFFER (not bound): own-half limit = river start - 0.5
BRIDGE_HALF_WIDTH = 1.0  # Board::BRIDGE_HALF_WIDTH (not bound): a bridge spans its centre x +- 1.0
KING_RADIUS, PRINCESS_RADIUS = 2.0, 1.5  # Tower::getCollisionRadius ('R' king 2.0, princess 1.5)
FIREBALL = 7  # a plain disc spell: legal anywhere on the board but the dead back-row corners
# Seconds per engine tick. The bindings carry no tick length, but they do carry the tick double elixir starts on, and
# that is 2:00 into the match by the game's rules, so the engine's own number fixes it (0.1 s at 1200; a 50 ms engine
# would bind 2400 and this follows). The page paces its real-time loop and its clock with it.
TICK_S = 120.0 / cre.ClashRoyaleEnv.DOUBLE_ELIXIR_TICK
REGULATION_END_S = 180.0  # MatchRules::REGULATION_END_TICK (not bound): 3:00 of play, then overtime
CLOCK = {"regulation_end": REGULATION_END_S, "double_at": cre.ClashRoyaleEnv.DOUBLE_ELIXIR_TICK * TICK_S,
         "triple_at": cre.ClashRoyaleEnv.TRIPLE_ELIXIR_TICK * TICK_S, "tick_s": TICK_S}  # seconds of engine clock
# Crown tower reach (not bound): GameManager::addTower builds the King with attack and sight 7.0, the Princess Tower
# from TowerTroops.h (attackRange 7.5 for the original and every tower troop). For the page's hover circles only.
TOWER_RANGE = {"King Tower": 7.0, "Princess Tower": 7.5}


def legal_mask(game, card_id: int, team: int = 0) -> str:
    """'1'/'0' per cell, row-major from engine y = 0 (team 0's back row), x = 0 first: would playCard accept
    `card_id` at this cell's centre right now? Buildings on the board change it, so it is asked every tick."""
    v = game.is_valid_placement
    return "".join("1" if v(card_id, float(x), float(y), team) else "0" for y in range(H) for x in range(W))


def board_geometry(game) -> dict:
    """The terrain as the engine has it: the river band, the bridge columns (the observation's river row, which is
    Board::isOnBridge), the own-half limit, the dead back-row corners (where even a spell is refused)."""
    river_start = game.get_own_half_max_y() + OWN_HALF_RIVER_BUFFER  # Board::riverY_start (15.5)
    river_end = 2 * cre.ARENA_BRIDGE_Y - river_start  # Board::riverY_end (17.5), symmetric about the bridge row
    obs = np.asarray(game.get_observation_for_team(0), dtype=np.float32)
    river_row = obs[8 * W * H + 17 * W: 8 * W * H + 18 * W]  # channel 8, row 17: +1 bridge, -1 river
    bridges = [int(x) for x in range(W) if river_row[x] > 0]
    # where even a spell is refused: bounds and Board::isBackRowDeadZone (Fireball, which has no other restriction)
    dead = "".join("0" if c == "1" else "1" for c in legal_mask(game, FIREBALL))
    return {"w": W, "h": H, "river": [river_start, river_end], "bridge_cols": bridges,
            "bridge_x": [cre.ARENA_LEFT_BRIDGE_X, cre.ARENA_RIGHT_BRIDGE_X], "bridge_half_width": BRIDGE_HALF_WIDTH,
            "own_half_max_y": game.get_own_half_max_y(), "dead": dead, "center_x": cre.ARENA_CENTER_X,
            "king_radius": KING_RADIUS, "princess_radius": PRINCESS_RADIUS}


def card_spec(card_id: int) -> dict:
    """What the page draws for a card's ghost: its placement footprint (CardDefinition::placementRadius: 1.0 for
    every building, 0.4 for a troop), the spell radius, and the bodies' spawn offsets (team 0's frame)."""
    info, st = cre.get_card_info(card_id), cre.get_card_stats(card_id)
    return {"id": card_id, "name": info["name"], "cost": info["cost"], "spell": bool(info["is_spell"]),
            "building": bool(info["is_building"]), "placement_radius": round(float(info["placement_radius"]), 3),
            "spell_radius": float(st["spell_radius"]), "hp": int(st["hp"]),
            "offsets": [[round(float(a), 3), round(float(b), 3)] for a, b in st["spawn_offsets"]]}


def max_hp(game, unit) -> int | None:
    """A unit's full HP: a tower's from the engine (slot 0 king, 1 left, 2 right), else its card's stats."""
    if unit.tower:
        slot = 0 if abs(unit.x - cre.ARENA_CENTER_X) < 1 else (1 if unit.x < cre.ARENA_CENTER_X else 2)
        return int(game.get_tower_max_hp(unit.team, slot))
    try:
        return int(cre.get_card_stats(int(unit.card_id))["hp"]) if unit.card_id >= 0 else None
    except (ValueError, TypeError):
        return None


def catalog() -> dict:
    """Every card in the engine's registry, for the page's sprites, ghosts and hover circles: name, cost, kind, air,
    attack and sight range, spell radius, HP, bodies. Bodies a card spawns (Golemites, the Hut's Spear Goblins) carry
    a negative card id and no stats of their own; the page finds them by name."""
    out = {}
    for i in cre.get_all_card_ids():
        info, st = cre.get_card_info(i), cre.get_card_stats(i)
        out[int(i)] = {"name": info["name"], "cost": info["cost"], "spell": bool(info["is_spell"]),
                       "building": bool(info["is_building"]), "flying": bool(st["flying"]),
                       "air": bool(st["targets_air"]), "range": round(float(st["attack_range"]), 2),
                       "sight": round(float(st["sight_range"]), 2), "spell_radius": round(float(st["spell_radius"]), 2),
                       "hp": int(st["hp"]), "bodies": int(st["bodies"])}
    return out


def next_card(game, team: int = 0) -> int | None:
    """The card that comes into the hand after the next play, read from the engine: the deck queue is not bound, but it
    is first-in first-out, so on a throwaway copy (ClashEnv::snapshot) with a full bar, play any slot that will go and
    see what replaces it. The real game is untouched. None if no slot can be played on the copy."""
    if team != 0:
        return None
    hand = list(game.get_hand())
    for k, c in enumerate(hand):
        s = game.snapshot()
        s.set_elixir_for_team(0, 10.0)
        cell = legal_mask(s, c).find("1")
        if cell < 0:
            continue
        s.step(k, float(cell % W), float(cell // W), 1)
        after = list(s.get_hand())
        if after != hand:
            return int(after[k])
    return None
