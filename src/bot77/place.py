"""Placement model (2026-10-22): where to put a chosen card, given the board. Like A, it ranks candidates: a set of
spots built from the board and fixed spots learned from where players put their cards (k-means of the training
placements); a gradient-boosted model scores each spot and the best one wins. Spots are anchored to the board (on the
main attacker, just in front of it, on any enemy unit, behind our tank, on the best spell target, the enemy towers),
so "Mini P.E.K.K.A on the Giant" is one anchor and not a learned coordinate.

Positions are in the player's frame of bot77.board (box centres). Taps go where the unit's feet land, ~FEET_DY tiles
below its box centre (the two players' placements line up with that offset; 2026-10-22).
"""
from __future__ import annotations

from bot77.features.frames import FEET_DY  # noqa: F401  (spot code: bot77.features.spots)
from bot77.features.spots import ENEMY_TOWERS, KINDS, spot_features, spots  # noqa: F401


class Placer:
    """The trained placement model at play time (data/models/place_v2/model.pkl from scripts/train_place.py)."""

    def __init__(self, path: str = "data/models/place_v2/model.pkl"):
        import pickle
        from pathlib import Path
        art = pickle.loads(Path(path).read_bytes())
        self.model, self.feats, self.code, self.fixed = art["model"], art["features"], art["code"], art["fixed"]

    def place(self, card: str, ctype: str, card_id: int, units: list[dict], towers: dict, summary: dict,
              elixir: dict[str, int]) -> tuple[float, float]:
        """The best spot, as a tap position (feet) in the player's frame."""
        import numpy as np
        cand = spots(card, ctype, units, self.fixed, elixir)
        rows = [{**summary, **spot_features(s, card, ctype, units, elixir, towers), "p_card": self.code.get(card_id, np.nan)}
                for s in cand]
        X = np.array([[r.get(f, np.nan) for f in self.feats] for r in rows], dtype=np.float64)
        best = cand[int(np.argmax(self.model.predict_proba(X)[:, 1]))]
        return best["x"], best["y"] + (FEET_DY if ctype != "spell" else 0.0)
