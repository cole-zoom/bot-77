"""Our cards (`cards` table: Supercell ids and names) <-> the engine's card ids (docs/simulator.md S1, 2026-10-23).

Matched by name, ignoring case and punctuation ("Mini P.E.K.K.A" = "Mini PEKKA"). A play's `form` picks the
variant: 'evo' -> the engine's Evolution of that card (a separate id, same name), 'hero' -> "Hero <name>". Tower troops
are not cards in the engine but a TowerTroopType per team. What the engine lacks is listed in
docs/simulator-cards.md (`scripts/sim_cards.py`).
"""
from __future__ import annotations

import re
from functools import lru_cache

import clash_royale_env as cre
import lancedb

TOWER_TROOPS = {  # our detector's tower class / card name -> the engine's tower troop
    "queen-tower": cre.TowerTroopType.NONE, "Tower Princess": cre.TowerTroopType.NONE,  # the engine default
    "cannoneer-tower": cre.TowerTroopType.CANNONEER, "Cannoneer": cre.TowerTroopType.CANNONEER,
    "dagger-duchess-tower": cre.TowerTroopType.DAGGER_DUCHESS, "Dagger Duchess": cre.TowerTroopType.DAGGER_DUCHESS,
    "Royal Chef": cre.TowerTroopType.ROYAL_CHEF,
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


@lru_cache(maxsize=1)
def engine_cards() -> dict[int, dict]:
    """Every engine card id (playable, Evolutions and Heroes) -> get_card_info, plus 'evolution' (bool)."""
    playable = set(cre.get_all_card_ids())
    out = {}
    for i in range(cre.ClashRoyaleEnv.NUM_CARD_IDS):
        try:
            info = dict(cre.get_card_info(i))
        except ValueError:
            continue
        info["evolution"] = i not in playable
        out[i] = info
    return out


@lru_cache(maxsize=1)
def _by_name() -> dict[tuple[str, str], int]:
    """(normalised base name, form) -> engine id."""
    out = {}
    for i, info in engine_cards().items():
        n = _norm(info["name"])
        if info["evolution"]:
            out[(n, "evo")] = i
        elif info["is_hero"] and n.startswith("hero"):
            out[(n[4:], "hero")] = i
        else:
            out[(n, "normal")] = i
    return out


@lru_cache(maxsize=1)
def our_cards(db_path: str = "data/lancedb") -> dict[int, str]:
    rows = lancedb.connect(db_path).open_table("cards").search().select(["id", "name"]).limit(10_000).to_arrow()
    return dict(zip(rows["id"].to_pylist(), rows["name"].to_pylist()))


def engine_id(card: int | str, form: str | None = "normal") -> int | None:
    """Our card (Supercell id or name) in a form ('normal' / 'evo' / 'hero') -> engine id, None if the engine lacks
    it. An Evolution or Hero the engine lacks falls back to the plain card (recorded by the caller)."""
    name = our_cards()[card] if isinstance(card, int) else card
    n = _norm(name)
    form = form or "normal"
    return _by_name().get((n, form))


def our_name(engine_card_id: int) -> str:
    """Engine id -> our card name (Evolutions and Heroes -> the base card's name, as our tables have them)."""
    info = engine_cards()[engine_card_id]
    n = _norm(info["name"])
    if info["is_hero"] and n.startswith("hero"):
        n = n[4:]
    names = {_norm(v): v for v in our_cards().values()}
    return names.get(n, info["name"])


def cost(engine_card_id: int) -> float:
    return engine_cards()[engine_card_id]["cost"]
