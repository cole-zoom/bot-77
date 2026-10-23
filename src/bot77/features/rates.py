"""Response-rate features (A v1, 2026-10-21): how often a card is chosen, when it's in hand, right after the opponent's
last card (resp_rate), after the player's own last card (own_rate_after), and (A v2) as the answer to the main threat
on the board (resp_threat), each smoothed toward the card's overall rate. One class for training (cross-fitted, so a
game never informs its own rows) and play time (the tables saved with the model).
"""
from __future__ import annotations

import zlib
from collections import defaultdict

PRIOR = 20


class RateTables:
    def __init__(self, pair: dict, base: dict, prior: int = PRIOR):
        self.pair, self.base, self.prior = pair, base, prior

    @classmethod
    def fit(cls, rows: list[dict], prior: int = PRIOR) -> "RateTables":
        pair, base = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        for r in rows:
            keys = [((r["opp_prev1"], r["cand_card"]), pair), (("own", r["own_prev1"], r["cand_card"]), pair),
                    (r["cand_card"], base)]
            if "b_threat_card" in r:  # A v2: answers to the attacker on the board, not just the last play
                keys.append((("threat", r["b_threat_card"], r["cand_card"]), pair))
            for key, tab in keys:
                tab[key][0] += r["chosen"]; tab[key][1] += 1
        return cls(dict(pair), dict(base), prior)

    def rate(self, key, card) -> float:
        b = self.base.get(card)
        b = b[0] / b[1] if b and b[1] else 0.25
        c, n = self.pair.get(key, (0, 0))
        return (c + self.prior * b) / (n + self.prior)

    def apply(self, rows: list[dict]) -> list[dict]:
        for r in rows:
            r["resp_rate"] = self.rate((r["opp_prev1"], r["cand_card"]), r["cand_card"])
            r["own_rate_after"] = self.rate(("own", r["own_prev1"], r["cand_card"]), r["cand_card"])
            r["elixir_after"] = r["elixir"] - r["cand_cost"]
            if "b_threat_card" in r:
                r["resp_threat"] = self.rate(("threat", r["b_threat_card"], r["cand_card"]), r["cand_card"])
        return rows

    def to_dict(self) -> dict:  # the form saved in model.pkl ("rates")
        return {"pair": dict(self.pair), "base": dict(self.base), "prior": self.prior}

    @classmethod
    def from_dict(cls, d: dict) -> "RateTables":
        return cls(d["pair"], d["base"], d["prior"])


def add_response_rates(rows: list[dict], folds: int = 5, prior: int = PRIOR, full: RateTables | None = None) -> RateTables:
    """Training: train rows get rates cross-fitted by game (5 folds), every other split the tables of all train games
    (or `full`, tables fitted earlier). Returns those full tables (saved with the model)."""
    train_games = sorted({r["game"] for r in rows if r["split"] == "train"})
    fold = {g: zlib.crc32(g.encode()) % folds for g in train_games}
    tr = [r for r in rows if r["split"] == "train"]
    full = full or RateTables.fit(tr, prior)
    per_fold = {k: RateTables.fit([r for r in tr if fold[r["game"]] != k], prior) for k in range(folds)} if tr else {}
    for r in rows:
        (per_fold[fold[r["game"]]] if r["split"] == "train" else full).apply([r])
    return full
