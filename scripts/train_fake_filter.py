"""Train the fake-play filter (bot77.fakefilter) on the train-split replays and sweep its threshold on held-back
replays + dev recordings (decisions.md, 2026-10-16). Test and benchmark games are never read here.

  uv run python scripts/train_fake_filter.py --train opponent_events_dev_ff_s0,... --heldback opponent_events_dev_va_def6
        --dev opponent_events_dev_dev_def6   -> data/models/fake_filter/model.pkl
"""
import argparse, glob, json, pickle, sys
from pathlib import Path

import lancedb
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).parent))
from render_game import match_plays  # noqa: E402
from bot77.fakefilter import FEATURES, features  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True); ap.add_argument("--heldback", required=True); ap.add_argument("--dev", required=True)
ap.add_argument("--out", default="data/models/fake_filter")
ap.add_argument("--model-in", default="", help="score this existing model.pkl instead of training (comparisons)")
ap.add_argument("--pov-weight", type=float, default=1.0, help="sample weight of plays from POV tables (not replays)")
a = ap.parse_args()

db = lancedb.connect("data/lancedb")
cards = {c["name"]: c for c in db.open_table("cards").search().where("kind = 'deck_card'").limit(1000)
         .select(["name", "type", "elixir", "ability_cost"]).to_arrow().to_pylist()}
hud = db.open_table("hud_states")
excluded = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])
excluded |= set(json.loads(Path("config/replay_test_games.json").read_text())["matches"])
from bot77.holdout import held_out  # noqa: E402
excluded |= held_out()
truth = {}
for f in glob.glob("data/ground_truth/opponent_truth_*.json"):
    t = json.loads(Path(f).read_text())
    if t["matches"][0] not in excluded:
        truth[t["matches"][0]] = t


def load(tables: str):
    """X, y (1 = fake), and per-match (events, X, y) for re-scoring at a threshold."""
    Xs, ys, per, ws = [], [], [], []
    for tn in tables.split(","):
        tbl = db.open_table(tn)
        ms = sorted(set(tbl.search().select(["match_id"]).limit(10**7).to_arrow().column("match_id").to_pylist()) & set(truth))
        for m in ms:
            ev = sorted(tbl.search().where(f"match_id = '{m}'").limit(10**4).select(
                ["t_video", "card", "votes", "units", "t_seen", "x_tiles", "y_tiles", "classes", "subunit_only", "evo",
                 "deck"]).to_arrow().to_pylist(), key=lambda r: r["t_video"])
            if not ev:
                continue
            items = [i for i in truth[m]["items"] if i["match_id"] == m]
            fake = {id(e) for t, kind, _, e, _ in match_plays(items, ev) if kind == "FAKE"}
            tl = [(r["t_video"], r["t_match"], r["multiplier"]) for r in hud.search().where(f"match_id = '{m}'")
                  .limit(10**6).select(["t_video", "t_match", "multiplier"]).to_arrow().to_pylist()]
            own = [(x["t_video"], x["card"]) for x in db.open_table("events").search().where(f"match_id = '{m}' AND kind = 'play'")
                   .limit(10**4).select(["t_video", "card"]).to_arrow().to_pylist()]
            X = features(ev, ev[0]["deck"] or [], tl, cards, own)
            y = np.array([1 if id(e) in fake else 0 for e in ev])
            Xs.append(X); ys.append(y); per.append((m, ev, items, X, y))
            ws.append(np.full(len(y), 1.0 if m.startswith(("bl_", "tv_")) else a.pov_weight))
    load.w = np.concatenate(ws)
    return np.vstack(Xs), np.concatenate(ys), per


def rescore(per, model, thr):
    right = wrong = fakes = 0
    for m, ev, items, X, _ in per:
        p = model.predict_proba(X[:, :model.n_features_in_])[:, 1]
        kept = [e for e, q in zip(ev, p) if q < thr]
        for t, kind, _, e, _ in match_plays(items, kept):
            right += kind == "RIGHT"; wrong += kind == "WRONG"; fakes += kind == "FAKE"
    return right, wrong, fakes


if a.model_in:
    model = pickle.loads(Path(a.model_in).read_bytes())["model"]
else:
  Xtr, ytr, _ = load(a.train)
  print(f"train: {len(ytr)} plays, {ytr.mean():.1%} fake", flush=True)
  model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                       min_samples_leaf=40, random_state=0).fit(Xtr, ytr, sample_weight=load.w)
  out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
  (out / "model.pkl").write_bytes(pickle.dumps({"model": model, "features": FEATURES}))
for name, tables in (("heldback", a.heldback), ("dev", a.dev)):
    X, y, per = load(tables)
    X = X[:, :model.n_features_in_]  # models from before the own-play features
    p = model.predict_proba(X)[:, 1]
    from sklearn.metrics import roc_auc_score
    print(f"\n{name}: {len(y)} plays, {y.mean():.1%} fake, AUC {roc_auc_score(y, p):.3f}")
    for thr in (1.01, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
        print(f"  thr {thr:4.2f}: right / wrong / fakes = {rescore(per, model, thr)}", flush=True)
