"""Map KataCR detector classes [Wu et al. 2025] to our card names."""

from __future__ import annotations

import re

# Classes that belong to a card but aren't the card's own unit: sub-units, death spawns,
# projectiles and ability summons. Spawns of these are usually *not* a card being played.
SUBUNITS = {
    "elixir-golem-big": "Elixir Golem", "elixir-golem-mid": "Elixir Golem", "elixir-golem-small": "Elixir Golem",
    "golemite": "Golem", "lava-pup": "Lava Hound", "phoenix-big": "Phoenix", "phoenix-egg": "Phoenix",
    "phoenix-small": "Phoenix", "rascal-boy": "Rascals", "rascal-girl": "Rascals", "royal-guardian": "Little Prince",
    "zappy": "Zappies", "hog": "Royal Hogs", "goblin-brawler": "Goblin Cage",
    # Cards made of two entities (not in the 2024 KataCR classes; for our fine-tuned detector)
    "goblinstein-doctor": "Goblinstein", "goblinstein-monster": "Goblinstein",
    "spirit-empress-ground": "Spirit Empress", "spirit-empress-air": "Spirit Empress",  # two forms (Cole)
    "tombstone-hero": "Tombstone", "tomb-queen": "Tombstone",  # hero form + its ability summon
}
DEATH_OR_SUMMON = {"golemite", "lava-pup", "phoenix-egg", "phoenix-small", "royal-guardian", "goblin-brawler",
                   "elixir-golem-mid", "elixir-golem-small"}
# Not units: HUD-like arena elements and effects.
NON_UNITS = {"bar", "bar-level", "clock", "dirt", "elixir", "emote", "text", "selected", "axe", "bomb", "goblin-ball",
             "king-tower", "king-tower-bar", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
             "dagger-duchess-tower-bar", "tower-bar", "skeleton-king-bar", "skeleton-king-skill", "padding_0",
             "evolution-symbol", "ice-spirit-evolution-symbol"}
EVO_SYMBOLS = {"evolution-symbol", "ice-spirit-evolution-symbol"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def class_to_card(cls: str, card_names: list[str]) -> tuple[str | None, str]:
    """(card name, form) for a detector class; (None, "") for non-units."""
    if cls in NON_UNITS:
        return None, ""
    if cls in SUBUNITS:
        return SUBUNITS[cls], "normal"
    form = "evo" if cls.endswith("-evolution") else "normal"
    base = _norm(cls.replace("-evolution", ""))
    by_norm = {_norm(c): c for c in card_names}
    for cand in (base, base + "s", base.rstrip("s"), base + "es"):
        if cand in by_norm:
            return by_norm[cand], form
    return None, ""
