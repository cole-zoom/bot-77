"""Win model (docs/simulator.md S3): P(the deciding player wins | the state), the search's score.

Cole, 2026-10-23: "there is no right answer, it's what will win you the game". A candidate play is scored by the change
in this probability over the next ~10 s, so damage, elixir and time are weighed by what they did to winning, not by
hand-set points.

State (`state_features`): the clock (time left, overtime, multiplier), towers (each one's HP fraction, crowns), both
elixir bars, and the board (elixir value of each side's units, split by half). Built the same way from a replay moment
(+ tower_hp) and from a running Sim, so the search can score simulated states. HP of units is left out: replays don't
show it.

Trained on the human replays' train split (both players' moments, label = matches.result for that side), calibration
checked on held-back. Gradient boosting (scikit-learn HistGradientBoostingClassifier).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from bot77.features.frames import RIVER_Y
from bot77.sim import cards as sc

REGULATION_S = 180.0
FEATURES = [
    "t_left_reg", "overtime", "t_left_ot", "multiplier",
    "own_king", "own_l", "own_r", "opp_king", "opp_l", "opp_r",
    "own_min", "opp_min", "own_crowns", "opp_crowns", "crown_diff",
    "elixir_own", "elixir_opp", "elixir_diff",
    "own_val_own_half", "own_val_opp_half", "opp_val_own_half", "opp_val_opp_half", "val_diff", "threat_diff",
]
MODEL_PATH = Path("data/models/sim_value_v1/model.pkl")


def unit_value(card: str) -> float:
    """Elixir per body (a squad's cost split over its bodies)."""
    from bot77.sim.scenario import _squad_size
    e = sc.engine_id(card)
    return sc.cost(e) / _squad_size(e) if e is not None else 0.0


def state_features(t_match: float, towers: dict[str, float], elixir_own: float, elixir_opp: float,
                   units: list[dict]) -> dict[str, float]:
    """towers: HP fractions {own_king, own_l, own_r, opp_king, opp_l, opp_r} (0 = destroyed). units: card, team
    (own / opp), y (our frame of the deciding player: own half is y > river)."""
    ot = t_match >= REGULATION_S
    f = {"t_left_reg": max(0.0, REGULATION_S - t_match), "overtime": float(ot),
         "t_left_ot": max(0.0, 300.0 - t_match) if ot else 120.0,
         "multiplier": 1.0 if t_match < 120 else 2.0 if t_match < 240 else 3.0}
    f.update({k: float(towers.get(k, 1.0)) for k in ("own_king", "own_l", "own_r", "opp_king", "opp_l", "opp_r")})
    f["own_min"] = min(f["own_l"], f["own_r"], f["own_king"])
    f["opp_min"] = min(f["opp_l"], f["opp_r"], f["opp_king"])
    f["own_crowns"] = float((f["opp_l"] <= 0) + (f["opp_r"] <= 0) + 3 * (f["opp_king"] <= 0))
    f["opp_crowns"] = float((f["own_l"] <= 0) + (f["own_r"] <= 0) + 3 * (f["own_king"] <= 0))
    f["crown_diff"] = f["own_crowns"] - f["opp_crowns"]
    f["elixir_own"], f["elixir_opp"] = float(elixir_own), float(elixir_opp)
    f["elixir_diff"] = f["elixir_own"] - f["elixir_opp"]
    v = {(t, h): 0.0 for t in ("own", "opp") for h in ("own", "opp")}
    for u in units:
        v[(u["team"], "own" if u["y"] > RIVER_Y else "opp")] += unit_value(u["card"])
    f["own_val_own_half"], f["own_val_opp_half"] = v[("own", "own")], v[("own", "opp")]
    f["opp_val_own_half"], f["opp_val_opp_half"] = v[("opp", "own")], v[("opp", "opp")]
    f["val_diff"] = sum(v[("own", h)] for h in ("own", "opp")) - sum(v[("opp", h)] for h in ("own", "opp"))
    f["threat_diff"] = v[("own", "opp")] - v[("opp", "own")]  # attacking value: ours on their half minus theirs on ours
    return f


def sim_features(sim) -> dict[str, float]:
    """The same features from a running bot77.sim.engine.Sim (towers as engine HP fractions)."""
    tw = {k: v["frac"] for k, v in sim.towers().items()}
    units = [u for u in sim.units() if not u["spell"]]
    return state_features(sim.t, tw, sim.elixir("own"), sim.elixir("opp"), units)


class WinModel:
    def __init__(self, model=None):
        self.model = model

    def p(self, feats: list[dict] | dict) -> np.ndarray | float:
        one = isinstance(feats, dict)
        X = np.array([[f[k] for k in FEATURES] for f in ([feats] if one else feats)])
        p = self.model.predict_proba(X)[:, 1]
        return float(p[0]) if one else p

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self.model))

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "WinModel":
        return cls(pickle.loads(Path(path).read_bytes()))
