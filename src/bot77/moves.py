"""Step A, v1 (2026-10-21): the pro-move predictor's data. At every play of either player in a keyed replay, the
player's view right before it (their 4-card hand, next card, elixir, the clock, the multiplier, their own and the
opponent's recent plays, the opponent cards seen so far) and the card they chose. A predicts which of the 4 hand
cards is played; "when" and "where" come later (decisions.md).

Replays show both hands, so both players' sides are exact: the bottom player from the replay processed with
replay_bottom (<id>), the top player from <id>_top. The opponent's history can also be swapped for C's detected
plays (opponent_events) to measure how much C's misses and fakes cost A.
"""
from __future__ import annotations

# the features live in bot77.features (2026-10-22); re-exported here for older imports
from bot77.features.hand import CAND_FEATURES, CTX_FEATURES, HISTORY, TYPES, candidates, hud_before  # noqa: F401
from bot77.features.rates import RateTables


def decision_rows(own_plays: list[dict], opp_plays: list[dict], hud: list[dict], cards: dict[int, dict],
                  names: dict[str, int]) -> list[dict]:
    """One row per (play, hand candidate). own_plays / opp_plays: [{"t", "card"}] sorted (card names; abilities as
    "__ability"); hud: this player's HUD rows sorted by t_video (hand = 4 card ids, next, elixir, t_match,
    multiplier); cards: id -> {"name", "elixir", "type"}. Rows carry "decision" (play index) and "chosen" (0/1)."""
    rows = []
    last_played: dict[int, float] = {}
    for k, p in enumerate(own_plays):
        if p["card"] in (None, "__ability", "Mirror") or names.get(p["card"]) is None:
            if p["card"] and names.get(p["card"]) is not None:
                last_played[names[p["card"]]] = p["t"]
            continue
        h = hud_before(hud, p["t"])
        cid = names[p["card"]]
        if h is None or not h["hand"] or cid not in h["hand"] or any(c is None or c < 0 for c in h["hand"]):
            last_played[cid] = p["t"]
            continue  # the hand read doesn't show the played card: no clean decision state
        for r in candidates(h, p["t"], own_plays[:k], opp_plays, last_played, cards, names):
            rows.append({**r, "decision": k, "t": p["t"], "chosen": int(r["cand_card"] == cid)})
        last_played[cid] = p["t"]
    return rows



class MovePredictor:
    """A v1 at play time: scores the 4 candidates of a decision (data/models/moves_v1/model.pkl from
    scripts/train_moves.py, which stores the model, card coding and response-rate tables)."""

    def __init__(self, path: str = "data/models/moves_v1/model.pkl"):
        import pickle
        from pathlib import Path
        art = pickle.loads(Path(path).read_bytes())
        self.model, self.feats, self.cats, self.code = art["model"], art["features"], art["cats"], art["code"]
        self.rates = RateTables.from_dict(art["rates"])

    @property
    def uses_board(self) -> bool:
        return "resp_threat" in self.feats

    def scores(self, rows: list[dict]) -> list[float]:
        import numpy as np
        self.rates.apply(rows)  # resp_rate, own_rate_after, elixir_after (+ resp_threat when rows carry the board)
        X = np.array([[(self.code.get(r[f], np.nan) if f in self.cats else r[f]) for f in self.feats] for r in rows],
                     dtype=np.float64)
        return list(self.model.predict_proba(X)[:, 1])


class TimingPredictor:
    """A v3 / v4 at play time (data/models/moves_v4/model.pkl from scripts/train_timing.py): scores the 4 cards in hand
    and "wait" from the same features as training; plays the best card when (best card - wait) clears the threshold
    tuned on held-back."""

    def __init__(self, path: str = "data/models/moves_v4/model.pkl"):
        import pickle
        from pathlib import Path
        art = pickle.loads(Path(path).read_bytes())
        self.model, self.feats, self.cats, self.code = art["model"], art["features"], art["cats"], art["code"]
        self.rates, self.threshold = RateTables.from_dict(art["rates"]), art["play_threshold"]

    def decide(self, h: dict, now: float, own_before: list[dict], opp_plays: list[dict], last_played: dict[int, float],
               cards: dict[int, dict], names: dict[str, int], units: list[dict], towers: dict) -> dict:
        """{"play": bool, "slot": best card's slot, "card": its name, "margin": best card - wait, "scores": [5]}."""
        import numpy as np
        from bot77.features.board import add_board_features
        from bot77.features.hand import timing_context, wait_row
        rows = candidates(h, now, own_before, opp_plays, last_played, cards, names)
        add_board_features(rows, units, towers, cards)
        for r in rows:
            r["cand_is_wait"] = 0.0
        rows.append(wait_row(rows[0]))
        tc = timing_context(h, cards)
        for r in rows:
            r.update(tc)
        self.rates.apply(rows)
        X = np.array([[(self.code.get(r[f], np.nan) if f in self.cats else r.get(f, np.nan)) for f in self.feats] for r in rows],
                     dtype=np.float64)
        s = self.model.predict_proba(X)[:, 1]
        best = int(np.argmax(s[:4]))
        margin = float(s[best] - s[4])
        return {"play": margin > self.threshold, "slot": best, "card": cards.get(rows[best]["cand_card"], {}).get("name"),
                "margin": margin, "scores": [round(float(v), 4) for v in s]}
