"""Guards on the network's actions (docs/plan-elixir-spells.md P1, 2026-10-29).

Spell guard (live and in the sim): Fireball or Arrows only go down if at least one enemy non-tower unit, or an enemy
tower, is within the spell's reach of the target; otherwise the play becomes a wait. Cole, watching the defence school
live: "sometimes it fireballs or arrows nothing like it's cycling it in the back" (17 of 52 spells at log-20, ~26 of
70 at log-50 landed behind our towers or in the centre with nothing there).

The reach is the engine's own hit test (include/entities/AreaSpell.h: a disc spell hits what its hitbox overlaps,
centre distance <= spell radius + the target's collision radius). The spell radius comes from the bindings
(get_card_stats: Fireball 2.5, Arrows 3.5), never a copy here. The target's radius isn't bound for troops: towers are
1.5 (Princess) / 2.0 (King) (include/entities/Tower.h), troops get the engine's implicit 0.4
(Entity::IMPLICIT_TROOP_RADIUS), a lower bound (a Giant's hitbox is bigger), so a spell grazing a big body's edge can
be vetoed; a spell anywhere near a body's centre never is.

One function for both callers, on engine units (UnitView-like: team, x, y, hp, tower), in engine coordinates:
- CloneEnv.step (training, the exam, the play page's sim) on our action;
- bot77.live.selfplay decide() on the mirrored engine (the screen read -> bot77.sim.mirror), so live and sim agree.
"""
from __future__ import annotations

from functools import lru_cache

FIREBALL, ARROWS = 7, 3
GUARDED = (FIREBALL, ARROWS)
TROOP_RADIUS = 0.4  # include/entities/Entity.h IMPLICIT_TROOP_RADIUS (not bound)
PRINCESS_RADIUS, KING_RADIUS = 1.5, 2.0  # include/entities/Tower.h getCollisionRadius (not bound)
KING_X = 8.5  # include/core/ArenaLayout.h CENTER_X


@lru_cache(maxsize=None)
def spell_radius(card_id: int) -> float:
    import clash_royale_env as cre
    return float(dict(cre.get_card_stats(int(card_id))).get("spell_radius", 0.0) or 0.0)


def _body_radius(u) -> float:
    if getattr(u, "tower", False):
        return KING_RADIUS if abs(float(u.x) - KING_X) < 1.0 else PRINCESS_RADIUS
    return TROOP_RADIUS


def spell_hits(card_id: int, x: float, y: float, units, enemy_team: int = 1) -> bool:
    """True if a spell cast at (x, y) would touch at least one live enemy body or tower."""
    r = spell_radius(card_id)
    for u in units:
        if int(u.team) != enemy_team or float(getattr(u, "hp", 1)) <= 0:
            continue
        reach = r + _body_radius(u)
        if (float(u.x) - x) ** 2 + (float(u.y) - y) ** 2 <= reach * reach:
            return True
    return False


def spell_vetoed(card_id: int, x: float, y: float, units, enemy_team: int = 1) -> bool:
    """The spell guard: a guarded spell (Fireball, Arrows) aimed where it touches nothing of theirs."""
    return int(card_id) in GUARDED and not spell_hits(card_id, x, y, units, enemy_team)
