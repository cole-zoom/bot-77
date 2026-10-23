"""Shared frame conventions and card roles for every feature (bot77.features, 2026-10-22).

Frame: arena tiles as on the bottom player's screen (x 0-18, y down, unit box centres); every feature is computed in
the deciding player's frame. The top player's board is turned around: x -> 18 - x, y -> FLIP_Y - y. FLIP_Y is not 2 x
the river (29.8): unit boxes sit ~1.3 tiles above the unit's feet, so the mirror line of box centres is ~13.75 (both
players' answer positions differ by a median 2.3 tiles with 29.8). Spells have no sprite: they mirror about the river.
A tap (where a card is dropped) is FEET_DY below the box centre of the unit it makes.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

RIVER_Y = 14.9
FEET_DY = 1.27
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
    return (x, y) if side == "bottom" else (18.0 - x, (2 * RIVER_Y if spell else FLIP_Y) - y)
