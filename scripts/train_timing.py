"""A v3 / v4 (phase 6 of docs/training-architecture.md, 2026-10-22): card + timing in one model. At every play and
could-have-played moment the candidates are the 4 cards in hand and "wait"; a gradient-boosted model scores each.
The bot plays the best card when (best card's score - wait's score) clears a threshold tuned on held-back so the model
plays as often as real players do; otherwise it waits.

Stage 1 (default): human replays only.
Stage 2 (--reviews): + Cole's reviews as labels on the bot's moments, read on the bot's paired replays
(pair_bot_replays.py: exact opponent history and full board). A reviewed play with a fix (another card / wait) makes
the fix the chosen candidate and the bot's card wrong; an OK play confirms the bot's card; a "right card, wrong spot"
mistake doesn't speak to the card and is left out. Review moments get weight w. With --cv, each weight in
--review-weights is scored by holding out a third of the reviewed games at a time: agreement with the reviewer on the
held-out games, and held-back replay accuracy (the cost to general play).

  uv run python scripts/train_timing.py                                   -> data/models/moves_v3 (stage 1)
  uv run python scripts/train_timing.py --reviews --cv                    (choose the weight)
  uv run python scripts/train_timing.py --reviews --review-weight 20 --out data/models/moves_v4
"""
import argparse, pickle, re, time, zlib
from collections import defaultdict
from pathlib import Path

import lancedb
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from bot77.features import (CAND_BOARD_FEATURES, CAND_FEATURES, CTX_FEATURES, SUMMARY_FEATURES, TIMING_FEATURES, WAIT_ID,
                            add_response_rates)
from bot77.features.datasets import HUMAN_REPLAYS, timing_moments

ap = argparse.ArgumentParser()
ap.add_argument("--eval", default="heldback"); ap.add_argument("--out", default="data/models/moves_v3")
ap.add_argument("--what", default="")
ap.add_argument("--reviews", action="store_true", help="stage 2: add the review labels")
ap.add_argument("--review-weight", type=float, default=20.0)
ap.add_argument("--cv", action="store_true", help="score --review-weights by holding out reviewed games")
ap.add_argument("--review-weights", default="0,5,20,50,100")
ap.add_argument("--max-iter", type=int, default=400)
a = ap.parse_args()

db = lancedb.connect("data/lancedb")
cards = {c["id"]: c for c in db.open_table("cards").search().where("kind = 'deck_card'").limit(1000)
         .select(["id", "name", "elixir", "type"]).to_arrow().to_pylist()}
names = {c["name"]: i for i, c in cards.items()}
BASE = CAND_FEATURES + CTX_FEATURES + SUMMARY_FEATURES + CAND_BOARD_FEATURES + TIMING_FEATURES
RATE = ["resp_rate", "own_rate_after", "elixir_after", "resp_threat"]
FEATS = BASE + RATE
CATS = ["cand_card", "next_card", "b_threat_card"] + [f for f in CTX_FEATURES if f.startswith(("own_prev", "opp_prev")) and not f.endswith("_dt")]
SLIM = ["opp_prev1", "own_prev1", "cand_card", "b_threat_card", "chosen", "elixir", "cand_cost", "game", "split"]

# ---- the review labels, on the bot's paired replay moments
review = {}  # replay moment id -> (action the reviewer wants, verdict, review game)
if a.reviews:
    link = {(l["game_id"], l["k"]): l["replay_moment_id"] for l in db.open_table("bot_play_moments").to_arrow().to_pylist()
            if l["replay_moment_id"]}
    for l in db.open_table("labels").search().where("source LIKE 'review_%'").limit(10**5).to_arrow().to_pylist():
        g, k = re.match(r"review:[^:]+:(bot_[^:]+):(\d+)$", l["label_id"]).groups()
        mid = link.get((g, int(k)))
        bot_card = next((x["card"] for x in db.open_table("bot_actions").search().where(f"game_id = '{g}' AND k = {k}")
                         .limit(1).to_arrow().to_pylist()), None)
        if mid is None:
            continue
        if l["verdict"] == "ok":
            review[mid] = (bot_card, "ok", g)
        elif l["action"] and l["action"] != bot_card:  # a fix: another card or wait
            review[mid] = (l["action"], "bad", g)
    print(f"review moments: {len(review)} ({sum(v[1] == 'bad' for v in review.values())} mistakes with a fix)")

# ---- rows: human replay moments + the reviewed bot moments
t0 = time.time()
X, slim, meta = [], [], []  # meta: moment id, kind, split, is_wait, chosen, elixir, threat value, review game


def add(rows, split, chosen_fn, rgame):
    for r in rows:
        X.append([r.get(f, np.nan) for f in BASE])
        ch = chosen_fn(r)
        slim.append({**{k: r[k] for k in SLIM}, "chosen": ch, "split": split})
        meta.append((r["moment_id"], r["kind"], split, r["cand_is_wait"], ch, r["elixir"], r["b_opp_val_our_half"], rgame))


for rows in timing_moments(db, cards, HUMAN_REPLAYS):
    add(rows, rows[0]["split"], lambda r: r["chosen"], None)
if review:
    ids = ",".join(f"'{m}'" for m in review)
    for rows in timing_moments(db, cards, f"moment_id IN ({ids})"):
        want, verdict, g = review[rows[0]["moment_id"]]
        hand = {r["cand_card"] for r in rows}
        if want != "wait" and names.get(want) not in hand:
            continue  # the reviewer's card wasn't in hand at the replay's moment
        add(rows, "review", lambda r, w=want: int(r["cand_is_wait"] == 1.0) if w == "wait" else int(r["cand_card"] == names[w]), g)
X = np.array(X, dtype=np.float64)
print(f"{len(meta):,} candidate rows from {len(meta) // 5:,} moments in {time.time() - t0:.0f} s", flush=True)
RATES = add_response_rates(slim)  # cross-fitted for train rows; review / eval rows use the train tables
X = np.hstack([X, np.array([[s[k] for k in RATE] for s in slim], dtype=np.float64)])
ids = sorted({int(v) for f in CATS for v in X[:, FEATS.index(f)] if v == v and v >= 0})
code = {c: i for i, c in enumerate(ids)}
assert len(code) < 255, len(code)
for f in CATS:
    j = FEATS.index(f)
    X[:, j] = [code.get(int(v), np.nan) if v == v and v >= 0 else np.nan for v in X[:, j]]
split = np.array([m[2] for m in meta]); y = np.array([m[4] for m in meta]); rgame = np.array([m[7] or "" for m in meta])


def fit(mask, weights):
    m = HistGradientBoostingClassifier(max_iter=a.max_iter, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=40,
                                       l2_regularization=1.0, categorical_features=[f in CATS for f in FEATS], random_state=0)
    m.fit(X[mask], y[mask], sample_weight=weights[mask])
    return m


def moments_of(mask):
    by = defaultdict(list)
    for i in np.where(mask)[0]:
        by[meta[i][0]].append(i)
    return by


def evaluate(model, mask):
    """Per moment: margin (best card - wait), the best card, the label."""
    p = model.predict_proba(X[mask])[:, 1]
    score = dict(zip(np.where(mask)[0], p))
    out = []
    for mid, idx in moments_of(mask).items():
        cards_ = [i for i in idx if meta[i][3] == 0.0]
        wait = next(i for i in idx if meta[i][3] == 1.0)
        best = max(cards_, key=lambda i: score[i])
        out.append({"margin": score[best] - score[wait], "best_right": meta[best][4] == 1, "play": meta[wait][4] == 0,
                    "wait_right": meta[wait][4] == 1, "elixir": meta[idx[0]][5], "threat": meta[idx[0]][6] > 0,
                    "verdict": review.get(mid, (None, None))[1]})
    return out


def threshold(res):
    """The margin above which the model plays, so that it plays as often as the players did."""
    rate = np.mean([r["play"] for r in res])
    return float(np.quantile([r["margin"] for r in res], 1 - rate))


def replay_metrics(res, thr, tag):
    lab = np.array([r["play"] for r in res]); mg = np.array([r["margin"] for r in res])
    el = np.array([r["elixir"] for r in res], dtype=float); th = np.array([r["threat"] for r in res])
    rule = th | (el >= 9.5)
    # at the rules' false-play rate: how many plays the model catches
    fpr_rule = rule[~lab].mean()
    thr_rule = float(np.quantile(mg[~lab], 1 - fpr_rule))
    plays = [r for r in res if r["play"]]
    return {f"{tag} card top-1 (play moments)": round(np.mean([r["best_right"] for r in plays]), 4),
            f"{tag} timing AUC": round(roc_auc_score(lab, mg), 4),
            f"{tag} timing AUC, elixir alone": round(roc_auc_score(lab, el), 4),
            f"{tag} play threshold (margin)": round(thr, 4),
            f"{tag} plays caught at that threshold": round((mg[lab] > thr).mean(), 4),
            f"{tag} waits kept at that threshold": round((mg[~lab] <= thr).mean(), 4),
            f"{tag} rules trigger: plays caught": round(rule[lab].mean(), 4),
            f"{tag} rules trigger: waits kept": round((~rule[~lab]).mean(), 4),
            f"{tag} model at the rules' wait rate: plays caught": round((mg[lab] > thr_rule).mean(), 4),
            f"{tag} moments": len(res), f"{tag} play moments": len(plays)}


def review_agreement(res, thr):
    """The model's decision (best card if margin > thr, else wait) = the reviewer's."""
    ok = [r for r in res if r["verdict"]]
    agree = [(r["wait_right"] if r["margin"] <= thr else r["best_right"]) for r in ok]
    bad = [x for x, r in zip(agree, ok) if r["verdict"] == "bad"]
    good = [x for x, r in zip(agree, ok) if r["verdict"] == "ok"]
    return {"agree": float(np.mean(agree)) if agree else None, "agree on mistakes": float(np.mean(bad)) if bad else None,
            "agree on OK plays": float(np.mean(good)) if good else None, "n": len(ok)}


tr, ev = split == "train", split == a.eval
if a.cv:
    games = sorted(set(rgame[split == "review"]))
    fold = {g: zlib.crc32(g.encode()) % 3 for g in games}
    print(f"\nreview weight CV: {len(games)} reviewed games in 3 folds")
    for wt in [float(w) for w in a.review_weights.split(",")]:
        agg, rep = defaultdict(list), []
        for k in range(3):
            test_rev = np.array([split[i] == "review" and fold.get(rgame[i]) == k for i in range(len(meta))])
            train_rev = (split == "review") & ~test_rev
            w = np.where(split == "review", wt, 1.0)
            model = fit(tr | train_rev, w) if wt > 0 else fit(tr, w)
            res_ev = evaluate(model, ev)
            thr = threshold(res_ev)
            ra = review_agreement(evaluate(model, test_rev), thr)
            for key in ("agree", "agree on mistakes", "agree on OK plays"):
                if ra[key] is not None:
                    agg[key].append((ra[key], ra["n"]))
            rep.append(replay_metrics(res_ev, thr, a.eval)[f"{a.eval} card top-1 (play moments)"])
            if wt == 0:
                break  # weight 0 doesn't depend on the fold: one fit scores every reviewed game
        fmt = {k: np.average([x for x, _ in v], weights=[n for _, n in v]) for k, v in agg.items()}
        print(f"  weight {wt:5g}: review agreement {fmt.get('agree', 0):.3f} (mistakes {fmt.get('agree on mistakes', 0):.3f}, "
              f"OK plays {fmt.get('agree on OK plays', 0):.3f}) | {a.eval} card top-1 {np.mean(rep):.4f}", flush=True)
    raise SystemExit

w = np.where(split == "review", a.review_weight, 1.0)
model = fit(tr | (split == "review") if a.reviews else tr, w)
res_ev = evaluate(model, ev)
thr = threshold(res_ev)
M = replay_metrics(res_ev, thr, a.eval)
print(f"\n{a.eval}:")
for k, v in M.items():
    print(f"  {k:56s} {v}")
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
(out / "model.pkl").write_bytes(pickle.dumps({"model": model, "features": FEATS, "cats": CATS, "code": code,
                                              "rates": RATES.to_dict(), "wait_id": WAIT_ID, "play_threshold": thr}))
if out.resolve().is_relative_to(Path("data/models").resolve()):
    from bot77.models import register
    register(f"card:{out.name}", "card", out.name, str(out / "model.pkl"), "card",
             what=a.what or ("A v4: A v3 + Cole's reviews as labels (weight %g)" % a.review_weight if a.reviews else
                             "A v3: card + timing (wait as a fifth candidate), board features, human replay moments"),
             data={"table": "moments + plays + labels", "moments_table_version": db.open_table("moments").version,
                   "train_moments": int(tr.sum() // 5), "sources": HUMAN_REPLAYS,
                   "review_moments": int((split == "review").sum() // 5) if a.reviews else 0,
                   "review_weight": a.review_weight if a.reviews else 0},
             features=FEATS, params=model.get_params() | {"play_threshold": thr}, metrics=M,
             notes="registered by train_timing.py; the play threshold is tuned on held-back (play as often as players)")
    print(f"registered card:{out.name} (not adopted)")
