"""Our tile frame <-> the engine's board, and which engine team is which (docs/simulator.md S1, 2026-10-23).

Ours: the bottom player's (or the deciding player's) frame, x 0-18 left to right, y down the screen, river at
RIVER_Y = 14.9 (bot77.features.frames). y is in screen units of one x-tile, so it is foreshortened: the arena's 32
rows span ~23 of our units. Detections are box centres; a unit's feet are FEET_DY below. Taps land at the feet.

Engine (ClashRoyaleEnv): 18 x 34 cells, cell i covers [i - 0.5, i + 0.5], river band y 15.5-17.5, team 0 defends low y.
Positions are where a unit stands (its feet).

The map, measured 2026-10-23 on the v5l_e15 detections of the human replays:
- x: princess towers at 3.45 / 14.50 in ours (both teams; no perspective in x), 3.0 / 14.0 in the engine: same 11-tile
  spread, offset 0.48. Taken as the cell convention's 0.5 (engine [-0.5, 17.5] = ours [0, 18]).
- y: linear, Y_SCALE of our units per engine row, through the river centre (ours 14.9 = engine 16.5). Y_SCALE from
  the princess towers' box centres (5.94 / 21.28, 21 engine rows apart: 0.7305), which also puts both princess boxes
  1.29 above their tower's centre, as FEET_DY (1.27) is for units. Walking speeds agree: Giant, Golem, Knight,
  Valkyrie and Mini P.E.K.K.A walk 0.73-0.74 of our units per engine tile between y 8 and 18, no trend with y.
  The king towers don't fit the same line (0.76; ~0.45 of our units off behind each king): see
  docs/simulator-validation.md once S2 has measured it on tracks.
- Bridges: units with their feet in the river cross at x 3.7 / 14.2 in ours, inside the engine's bridge cells
  ([2, 4] / [14, 16] in ours).
"""
from __future__ import annotations

import clash_royale_env as cre

from bot77.features.frames import FEET_DY, RIVER_Y

X_OFFSET = 0.5
Y_SCALE = 0.7305
ENGINE_RIVER_Y = cre.ARENA_BRIDGE_Y  # 16.5, the river band's centre
ENGINE_H = cre.ARENA_HEIGHT  # 34

# The bottom player (ours, or the deciding player of a moment) is engine team 0, which defends low engine y. Kept
# here and nowhere else.
OWN_TEAM, OPP_TEAM = 0, 1
TEAMS = {"own": OWN_TEAM, "opp": OPP_TEAM, "bottom": OWN_TEAM, "top": OPP_TEAM}


def ours_to_engine(x: float, y: float) -> tuple[float, float]:
    """A ground point (feet, a tap, a spell's centre) in our frame -> engine board coordinates."""
    return x - X_OFFSET, ENGINE_RIVER_Y + (RIVER_Y - y) / Y_SCALE


def engine_to_ours(x: float, y: float) -> tuple[float, float]:
    """Engine board coordinates -> the ground point in our frame."""
    return x + X_OFFSET, RIVER_Y - (y - ENGINE_RIVER_Y) * Y_SCALE


def box_to_feet(x: float, y: float) -> tuple[float, float]:
    """A detection's box centre -> where the unit stands (bot77.place.FEET_DY)."""
    return x, y + FEET_DY


def feet_to_box(x: float, y: float) -> tuple[float, float]:
    return x, y - FEET_DY


def engine_team(team: str) -> int:
    """'own' / 'opp' (moments) or 'bottom' / 'top' (plays) -> engine team."""
    return TEAMS[team]


def mirror_engine_y(y: float) -> float:
    """step_self_play takes team 1's target y in team 1's own frame (ClashEnv::runSelfPlayTicks: 33 - y)."""
    return (ENGINE_H - 1) - y
