"""Train the win model bot77.sim.value (docs/simulator.md S3) -> data/models/sim_value_v1, `models` role sim_value.

  uv run python scripts/train_sim_value.py

Rows: every moment of the human replays (replay_tv, replay_bl; both players) whose tower HP was read (tower_hp),
features from the moment + tower_hp, label = that player's matches.result (draws and unknown results left out). Fit on train, calibration reported on
held-back (log loss, Brier, AUC, reliability) next to two baselines: crowns + clock only, and a constant.
"""
import bisect
import collections

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from bot77 import models
from bot77.features.frames import FEET_DY
from bot77.sim import store
from bot77.sim.scenario import REAL_ELIXIR_S, real_multiplier
from bot77.sim.value import FEATURES, MODEL_PATH, WinModel, state_features

SWAP = {"own_king": "opp_king", "own_l": "opp_r", "own_r": "opp_l", "opp_king": "own_king", "opp_l": "own_r",
        "opp_r": "own_l"}  # bottom-frame tower names -> the top player's own frame


def tower_series(match_id: str) -> dict[str, tuple[list[float], list[float]]]:
    out = collections.defaultdict(lambda: ([], []))
    for r in sorted(store.read("tower_hp", f"match_id = '{match_id}' AND accepted = true"), key=lambda r: r["t"]):
        if r["frac"] is not None:
            out[r["tower"]][0].append(r["t"])
            out[r["tower"]][1].append(r["frac"])
    return out


def at(series, t, default=1.0):
    ts, fs = series
    i = bisect.bisect_right(ts, t) - 1
    return fs[i] if i >= 0 else default


def opp_elixir(plays: list[dict], t_match: float) -> float:
    before = [p for p in plays if p["t_match"] <= t_match and p["elixir_after"] is not None]
    if not before:
        return min(10.0, 5.0 + t_match / REAL_ELIXIR_S)
    last = max(before, key=lambda p: p["t_match"])
    e, t = last["elixir_after"], last["t_match"]
    while t < t_match:
        e += 0.1 * real_multiplier(t) / REAL_ELIXIR_S
        t += 0.1
    return min(10.0, e)


def rows_for(match: dict, result_top: str | None) -> list[tuple[dict, int, str]]:
    gid = match["match_id"]
    res = {"bottom": match["result"], "top": result_top}
    series = tower_series(gid)
    if not series:  # tower HP not read for this game: every tower would look full
        return []
    plays = store.read("plays", f"game_id = '{gid}' AND source = 'hand'")
    out = []
    for m in store.read("moments", f"game_id = '{gid}'"):
        r = res.get(m["side"])
        if r not in ("win", "loss"):
            continue
        tw_flags = m["towers"] or {}
        towers = {}
        for k in ("own_king", "own_l", "own_r", "opp_king", "opp_l", "opp_r"):
            bottom_name = k if m["side"] == "bottom" else SWAP[k]
            towers[k] = at(series[bottom_name], m["t"]) if bottom_name in series else 1.0
            if tw_flags.get(k) is False:
                towers[k] = 0.0
        opp = "top" if m["side"] == "bottom" else "bottom"
        units = [{"card": u["card"], "team": u["team"], "y": u["y"] + FEET_DY} for u in (m["units"] or [])
                 if u["card"] and "tower" not in (u["cls"] or "") and "bar" not in (u["cls"] or "")]
        f = state_features(m["t_match"], towers, m["elixir"] or 0.0,
                           opp_elixir([p for p in plays if p["side"] == opp], m["t_match"]), units)
        out.append((f, int(r == "win"), gid))
    return out


def main():
    ms = store.read("matches", "source IN ('replay_tv', 'replay_bl')")
    by_id = {m["match_id"]: m for m in ms}
    data = collections.defaultdict(list)
    for m in ms:
        if m["pov_side"] != "bottom" or m["split"] not in ("train", "dev", "heldback"):
            continue
        top = by_id.get(m["pair_match_id"])
        data["heldback" if m["split"] == "heldback" else "train"] += rows_for(m, top["result"] if top else None)
    X = {k: np.array([[f[c] for c in FEATURES] for f, _, _ in v]) for k, v in data.items()}
    y = {k: np.array([lab for _, lab, _ in v]) for k, v in data.items()}
    games = {k: len({g for _, _, g in v}) for k, v in data.items()}
    print({k: (len(v), games[k]) for k, v in data.items()}, "rows, games")

    # strongly regularised: 382 games behind 105k correlated moments; the default settings memorised games (held-back
    # log loss 0.750, worse than a constant 0.693; this: 0.686)
    gb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.03, max_depth=3, min_samples_leaf=2000,
                                        l2_regularization=5.0, random_state=77).fit(X["train"], y["train"])
    base_cols = [FEATURES.index(c) for c in ("crown_diff", "t_left_reg", "overtime", "own_min", "opp_min")]
    lr = LogisticRegression(max_iter=1000).fit(X["train"][:, base_cols], y["train"])
    metrics = {}
    for name, p in (("model", gb.predict_proba(X["heldback"])[:, 1]),
                    ("crowns+clock+min tower (logistic)", lr.predict_proba(X["heldback"][:, base_cols])[:, 1]),
                    ("constant", np.full(len(y["heldback"]), y["train"].mean()))):
        yy = y["heldback"]
        auc = roc_auc_score(yy, p) if name != "constant" else 0.5
        metrics[name] = {"log_loss": log_loss(yy, p), "brier": brier_score_loss(yy, p), "auc": auc}
        print(f"held-back {name:36s} log loss {metrics[name]['log_loss']:.3f}  Brier {metrics[name]['brier']:.3f}  "
              f"AUC {auc:.3f}")
    p = gb.predict_proba(X["heldback"])[:, 1]
    print("reliability (held-back): predicted -> observed (n)")
    for lo in np.arange(0, 1, 0.1):
        sel = (p >= lo) & (p < lo + 0.1)
        if sel.sum():
            print(f"  {lo:.1f}-{lo + 0.1:.1f}: {y['heldback'][sel].mean():.2f} ({sel.sum()})")
    model = WinModel(gb)
    model.save()
    models.register("sim_value_v1", "sim_value", "v1", str(MODEL_PATH.parent), role="sim_value",
                    what="P(the deciding player wins | towers, elixir, clock, board) for the simulator search",
                    data={"train": f"human replays train+dev: {games['train']} games, {len(y['train'])} moments",
                          "heldback": f"{games['heldback']} games, {len(y['heldback'])} moments"},
                    features=FEATURES, params={k: str(v) for k, v in gb.get_params().items()},
                    metrics={"heldback": metrics}, notes="docs/simulator.md S3", adopt=True)
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
