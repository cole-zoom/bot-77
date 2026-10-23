"""bot-77 data browser, back end (phase 4 of docs/training-architecture.md, 2026-10-22): a FastAPI app over LanceDB.
Local only: it serves frames and videos from data/raw (laptop or the LaCie) and never leaves the machine.

  uv run python -m bot77.browser            -> http://127.0.0.1:8077  (serves ui/dist when built)
The front end (ui/, React + TypeScript + Tailwind) gets its types from /openapi.json (npm run gen:api).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import cv2
import lancedb
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from bot77.features import FEATURES, live_board, summary_features
from bot77.features.board import candidate_features
from bot77.features.datasets import play_history
from bot77.features.hand import candidates

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)  # data paths in the tables are repo-relative
app = FastAPI(title="bot-77 data browser", version="1")


@lru_cache(maxsize=1)
def db():
    return lancedb.connect("data/lancedb")


@lru_cache(maxsize=1)
def cards() -> dict[int, dict]:
    return {c["id"]: c for c in db().open_table("cards").search().where("kind = 'deck_card'").limit(1000)
            .select(["id", "name", "elixir", "type"]).to_arrow().to_pylist()}


def q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


# ---------------------------------------------------------------- overview
class TableInfo(BaseModel):
    name: str
    rows: int


class Count(BaseModel):
    key: str
    n: int


class Overview(BaseModel):
    tables: list[TableInfo]
    matches_by_source: list[Count]
    matches_by_split: list[Count]
    moments_by_kind: list[Count]
    labels_by_source: list[Count]
    runs: list[dict]
    models: list[str]


def counts(table: str, col: str, where: str | None = None) -> list[Count]:
    t = db().open_table(table)
    s = t.search()
    if where:
        s = s.where(where)
    vals = s.select([col]).limit(10**8).to_arrow().column(col).to_pylist()
    from collections import Counter
    return [Count(key=str(k), n=v) for k, v in Counter(vals).most_common()]


@app.get("/api/overview", response_model=Overview)
def overview():
    names = db().table_names(limit=10_000)
    return Overview(
        tables=[TableInfo(name=n, rows=db().open_table(n).count_rows()) for n in sorted(names)],
        matches_by_source=counts("matches", "source"), matches_by_split=counts("matches", "split"),
        moments_by_kind=counts("moments", "kind"), labels_by_source=counts("labels", "source"),
        runs=db().open_table("runs").to_arrow().to_pylist(),
        models=sorted(p.name for p in Path("data/models").iterdir()))


# ---------------------------------------------------------------- matches
class MatchRow(BaseModel):
    match_id: str
    video_id: str | None
    source: str | None
    split: str | None
    pov_side: str | None
    league: int | None
    result: str | None
    opponent_name: str | None
    deck_names: list[str] | None
    truth_set: str | None


class MatchList(BaseModel):
    total: int
    rows: list[MatchRow]


@app.get("/api/matches", response_model=MatchList)
def matches(source: str | None = None, split: str | None = None, text: str | None = None,
            bottom_only: bool = True, limit: int = 100, offset: int = 0):
    conds = []
    if source:
        conds.append(f"source = {q(source)}")
    if split:
        conds.append(f"split = {q(split)}")
    if bottom_only:
        conds.append("pov_side = 'bottom'")
    s = db().open_table("matches").search()
    if conds:
        s = s.where(" AND ".join(conds))
    rows = s.select(list(MatchRow.model_fields)).limit(10**6).to_arrow().to_pylist()
    if text:
        rows = [r for r in rows if text.lower() in r["match_id"].lower() or text.lower() in (r["opponent_name"] or "").lower()
                or any(text.lower() in d.lower() for d in r["deck_names"] or [])]
    rows.sort(key=lambda r: r["match_id"], reverse=True)
    return MatchList(total=len(rows), rows=[MatchRow(**r) for r in rows[offset:offset + limit]])


class Play(BaseModel):
    source: str
    run_id: str | None
    side: str
    t: float
    t_match: float | None
    card: str | None
    kind: str | None
    x: float | None
    y: float | None
    reason: str | None


class MomentStub(BaseModel):
    moment_id: str
    side: str
    t: float
    kind: str
    action: str | None


class MatchDetail(BaseModel):
    match: dict
    video: dict | None
    media: dict | None
    pair: dict | None
    plays: list[Play]
    moments: list[MomentStub]
    runs: list[str]


@app.get("/api/matches/{match_id}", response_model=MatchDetail)
def match(match_id: str):
    m = db().open_table("matches").search().where(f"match_id = {q(match_id)}").limit(1).to_arrow().to_pylist()
    if not m:
        raise HTTPException(404, "no such match")
    m = m[0]
    m.pop("card_snapshot", None)
    game = match_id.replace("_top_m", "_m", 1)
    v = db().open_table("videos").search().where(f"video_id = {q(m['video_id'])}").limit(1).to_arrow().to_pylist()
    v = v[0] if v else None
    md = None
    if v and v.get("path"):
        mm = db().open_table("media").search().where(f"path = {q(v['path'])}").limit(1).to_arrow().to_pylist()
        md = mm[0] if mm else {"path": v["path"], "available": Path(v["path"]).exists()}
        md["available"] = Path(os.path.realpath(v["path"])).exists()
    pair = None
    if m.get("pair_match_id"):
        pp = db().open_table("matches").search().where(f"match_id = {q(m['pair_match_id'])}").limit(1) \
            .select(["match_id", "deck_names", "opponent_name"]).to_arrow().to_pylist()
        pair = pp[0] if pp else None
    plays = db().open_table("plays").search().where(f"game_id = {q(game)}").limit(10**5).select(list(Play.model_fields)) \
        .to_arrow().to_pylist()
    moms = db().open_table("moments").search().where(f"game_id = {q(game)}").select(["moment_id", "side", "t", "kind"]) \
        .limit(10**5).to_arrow().to_pylist()
    acts = {l["moment_id"]: l["action"] for l in db().open_table("labels").search()
            .where(f"moment_id LIKE {q(game + ':%')} AND source IN ('replay', 'bot')").select(["moment_id", "action"])
            .limit(10**6).to_arrow().to_pylist()}
    runs = sorted(set(db().open_table("detections").search().where(f"match_id = {q(game)}").select(["run_id"])
                      .limit(10**8).to_arrow().column("run_id").unique().to_pylist()))
    return MatchDetail(match=m, video=v, media=md, pair=pair,
                       plays=sorted((Play(**p) for p in plays), key=lambda p: p.t),
                       moments=sorted((MomentStub(**x, action=acts.get(x["moment_id"])) for x in moms), key=lambda x: x.t),
                       runs=runs)


# ---------------------------------------------------------------- media
def playable(path: str) -> Path:
    """The file to read: the bot's .mkv recordings (scrcpy, stopped without an index) can't be seeked by the browser or
    OpenCV, so they're remuxed once (stream copy, ~1 s) to data/cache/remux/<name>.mp4."""
    real = Path(os.path.realpath(path))
    if real.suffix.lower() != ".mkv":
        return real
    out = Path("data/cache/remux") / f"{real.parent.name}.mp4"
    if not out.exists() or out.stat().st_mtime < real.stat().st_mtime:
        import subprocess
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(real), "-c", "copy", "-movflags", "+faststart", str(out)],
                       check=True, timeout=120)
    return out


@app.get("/api/video/{video_id}")
def video(video_id: str):
    v = db().open_table("videos").search().where(f"video_id = {q(video_id)}").limit(1).to_arrow().to_pylist()
    if not v or not v[0]["path"]:
        raise HTTPException(404, "no video file")
    real = Path(os.path.realpath(v[0]["path"]))
    if not real.exists():
        raise HTTPException(404, "file not available (on the LaCie? plug it in)")
    return FileResponse(playable(str(real)))


@app.get("/api/frame/{video_id}")
def frame(video_id: str, t: float = Query(...), width: int = 540):
    """One frame as JPEG (for videos the browser can't play or seek, e.g. the bot's unfinished .mkv)."""
    v = db().open_table("videos").search().where(f"video_id = {q(video_id)}").limit(1).to_arrow().to_pylist()
    if not v or not v[0]["path"]:
        raise HTTPException(404, "no video file")
    real = os.path.realpath(v[0]["path"])
    if not Path(real).exists():
        raise HTTPException(404, "file not available (on the LaCie? plug it in)")
    cap = cv2.VideoCapture(str(playable(real)))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0) * 1000)
    ok, im = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(404, "no frame at that time")
    h = round(im.shape[0] * width / im.shape[1])
    ok, buf = cv2.imencode(".jpg", cv2.resize(im, (width, h)), [cv2.IMWRITE_JPEG_QUALITY, 82])
    return Response(buf.tobytes(), media_type="image/jpeg")


@app.get("/api/botframe/{game_id}/{k}")
def botframe(game_id: str, k: int, which: str = "at"):
    """The screenshot the bot saved at its k-th play (which=at) or 2 s later (after): exact, no video seeking."""
    if which not in ("at", "after") or "/" in game_id or ".." in game_id:
        raise HTTPException(400, "bad request")
    f = Path("data/raw/live_games") / game_id / f"{k:03d}_{which}.jpg"
    if not f.exists():
        raise HTTPException(404, "no saved frame")
    return FileResponse(f)


class Box(BaseModel):
    cls: str
    card: str | None
    team: str
    conf: float
    x0: float  # fractions of the video frame
    y0: float
    x1: float
    y1: float


class Boxes(BaseModel):
    t: float | None
    boxes: list[Box]


@app.get("/api/detections/{match_id}", response_model=Boxes)
def detections(match_id: str, t: float, run_id: str = "v5l_e15", min_conf: float = 0.35):
    """The detector's boxes in the frame nearest t (within 0.5 s), as fractions of the frame."""
    game = match_id.replace("_top_m", "_m", 1)
    m = db().open_table("matches").search().where(f"match_id = {q(game)}").limit(1).select(["video_id"]).to_arrow().to_pylist()
    v = db().open_table("videos").search().where(f"video_id = {q(m[0]['video_id'])}").limit(1).to_arrow().to_pylist()[0] if m else None
    if not v or not v["width"]:
        return Boxes(t=None, boxes=[])
    rows = db().open_table("detections").search().where(
        f"run_id = {q(run_id)} AND match_id = {q(game)} AND t >= {t - 0.5} AND t <= {t + 0.5} AND conf >= {min_conf}") \
        .select(["t", "cls", "card", "team", "conf", "x0", "y0", "x1", "y1"]).limit(10**5).to_arrow().to_pylist()
    if not rows:
        return Boxes(t=None, boxes=[])
    tt = min({r["t"] for r in rows}, key=lambda x: abs(x - t))
    W, H = v["width"], v["height"]
    return Boxes(t=tt, boxes=[Box(cls=r["cls"], card=r["card"], team=r["team"], conf=r["conf"], x0=r["x0"] / W, y0=r["y0"] / H,
                                  x1=r["x1"] / W, y1=r["y1"] / H) for r in rows if r["t"] == tt])


# ---------------------------------------------------------------- moments
class CardScore(BaseModel):
    slot: int
    card: str | None
    score: float
    features: dict


class SpotScore(BaseModel):
    kind: str
    x: float
    y: float
    score: float


class MomentDetail(BaseModel):
    moment: dict
    labels: list[dict]
    board: dict
    cards: list[CardScore]
    spots: list[SpotScore]
    spot_card: str | None


@lru_cache(maxsize=1)
def card_model():
    from bot77.moves import MovePredictor
    return MovePredictor("data/models/moves_v1/model.pkl")


@lru_cache(maxsize=1)
def placer():
    from bot77.place import Placer
    return Placer("data/models/place_v2/model.pkl")


@app.get("/api/moments/{moment_id}", response_model=MomentDetail)
def moment(moment_id: str):
    m = db().open_table("moments").search().where(f"moment_id = {q(moment_id)}").limit(1).to_arrow().to_pylist()
    if not m:
        raise HTTPException(404, "no such moment")
    m = m[0]
    labels = db().open_table("labels").search().where(f"moment_id = {q(moment_id)}").limit(100).to_arrow().to_pylist()
    C = cards()
    names = {c["name"]: i for i, c in C.items()}
    el = {c["name"]: c.get("elixir") or 0 for c in C.values()}
    units = [{"card": u["card"], "team": u["team"], "x": u["x"], "y": u["y"]} for u in m["units"]]
    towers = {k: bool(v) for k, v in (m["towers"] or {}).items()}
    summ = summary_features(units, towers, el, names)
    # A v1's scores for the 4 cards in hand, from the same feature code the bot uses
    out_cards = []
    if m["hand"] and len(m["hand"]) == 4 and all(c is not None and c >= 0 for c in m["hand"]):
        src = "bot" if m["source"] == "bot" else "hand"
        hist = [p for p in db().open_table("plays").search().where(
            f"game_id = {q(m['game_id'])} AND side = {q(m['side'])} AND source = {q(src)} AND t < {m['t']}")
            .select(["t", "card", "kind"]).limit(10**4).to_arrow().to_pylist()]
        own = sorted(({"t": p["t"], "card": "__ability" if p["kind"] == "ability" else p["card"]} for p in hist), key=lambda p: p["t"])
        opp_side = "top" if m["side"] == "bottom" else "bottom"
        opp = sorted(({"t": p["t"], "card": p["card"]} for p in db().open_table("plays").search().where(
            f"game_id = {q(m['game_id'])} AND side = {q(opp_side)} AND source = 'hand'").select(["t", "card"])
            .limit(10**4).to_arrow().to_pylist()), key=lambda p: p["t"])
        last = {}
        for p in own:
            if p["card"] and names.get(p["card"]) is not None:
                last[names[p["card"]]] = p["t"]
        h = {"hand": m["hand"], "next": m["next"], "elixir": m["elixir"], "t_match": m["t_match"], "multiplier": m["multiplier"] or 1}
        rows = candidates(h, m["t"], own, opp, last, C, names)
        scores = card_model().scores([dict(r) for r in rows])
        for k, (r, s) in enumerate(zip(rows, scores)):
            cname = C.get(r["cand_card"], {}).get("name")
            board_f = candidate_features(cname, C.get(r["cand_card"], {}).get("type"), units, el)
            out_cards.append(CardScore(slot=k, card=cname, score=float(s), features={**r, **board_f}))
    # the placement model's spots for the card played (or the label's card)
    act = next((l["action"] for l in labels if l["source"] in ("replay", "bot") and l["action"] not in (None, "wait")), None)
    out_spots = []
    if act in names:
        import numpy as np
        from bot77.features.spots import spot_features, spots
        P = placer()
        cand = spots(act, C[names[act]]["type"], units, P.fixed, el)
        rows = [{**summ, **spot_features(s, act, C[names[act]]["type"], units, el, towers), "p_card": P.code.get(names[act], np.nan)}
                for s in cand]
        X = np.array([[r.get(f, np.nan) for f in P.feats] for r in rows], dtype=np.float64)
        sc = P.model.predict_proba(X)[:, 1]
        out_spots = [SpotScore(kind=s["kind"], x=s["x"], y=s["y"], score=float(v)) for s, v in zip(cand, sc)]
    m.pop("units_raw", None)
    return MomentDetail(moment=m, labels=labels, board=summ, cards=out_cards, spots=out_spots, spot_card=act)


@app.get("/api/moments")
def moments_list(game_id: str, kind: str | None = None, limit: int = 500):
    w = f"game_id = {q(game_id)}" + (f" AND kind = {q(kind)}" if kind else "")
    return db().open_table("moments").search().where(w).select(["moment_id", "side", "t", "kind", "elixir"]) \
        .limit(limit).to_arrow().to_pylist()


# ---------------------------------------------------------------- models registry
class ModelInfo(BaseModel):
    model_id: str
    family: str
    version: str
    path: str | None
    role: str
    adopted: bool
    status: str
    trained_at: str | None
    what: str | None
    data: dict
    features: list
    params: dict
    metrics: dict
    notes: str | None
    on_disk: bool


@app.get("/api/models", response_model=list[ModelInfo])
def models():
    from bot77.models import list_models
    out = []
    for r in list_models():
        out.append(ModelInfo(**{**r, "data": json.loads(r["data"] or "{}"), "features": json.loads(r["features"] or "[]"),
                                "params": json.loads(r["params"] or "{}"), "metrics": json.loads(r["metrics"] or "{}"),
                                "on_disk": bool(r["path"]) and Path(r["path"]).exists()}))
    return out


# ---------------------------------------------------------------- features + training data
class FeatureInfo(BaseModel):
    name: str
    group: str
    level: str
    doc: str
    since: str


@app.get("/api/features", response_model=list[FeatureInfo])
def features():
    return [FeatureInfo(**f.__dict__) for f in FEATURES]


class FeatureStat(BaseModel):
    name: str
    split: str
    n: int
    missing: float
    mean: float | None
    p5: float | None
    p50: float | None
    p95: float | None
    categorical: bool = False
    distinct: int | None = None  # card-valued features: how many different cards, and the commonest
    top: str | None = None


class TrainingView(BaseModel):
    table: str
    decisions_by_split: list[Count]
    stats: list[FeatureStat]
    sample: list[dict]


@app.get("/api/training/card", response_model=TrainingView)
def training_card(sample_split: str = "heldback", sample_n: int = 40):
    """A's training rows (move_samples_m: rebuilt from moments) as the model receives them, and each feature's
    distribution per split (catches leaks and broken features)."""
    import numpy as np
    from bot77.features import CAND_FEATURES, CTX_FEATURES
    t = db().open_table("move_samples_m").to_arrow()
    rows = t.to_pylist()
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[r["split"]].append(r)
    stats = []
    from collections import Counter
    cats = {"cand_card", "next_card"} | {f for f in CTX_FEATURES if f.startswith(("own_prev", "opp_prev")) and not f.endswith("_dt")}
    C = cards()
    for f in CAND_FEATURES + CTX_FEATURES:
        for sp, rs in by.items():
            if f in cats:
                vals = [r[f] for r in rs if r[f] is not None and r[f] >= 0]
                top = Counter(vals).most_common(1)
                stats.append(FeatureStat(name=f, split=sp, n=len(rs), missing=1 - len(vals) / max(len(rs), 1), mean=None,
                                         p5=None, p50=None, p95=None, categorical=True, distinct=len(set(vals)),
                                         top=C.get(top[0][0], {}).get("name") if top else None))
                continue
            vals = np.array([r[f] for r in rs if r[f] is not None and r[f] == r[f]], dtype=float)
            stats.append(FeatureStat(name=f, split=sp, n=len(rs), missing=1 - len(vals) / max(len(rs), 1),
                                     mean=float(vals.mean()) if len(vals) else None,
                                     p5=float(np.percentile(vals, 5)) if len(vals) else None,
                                     p50=float(np.percentile(vals, 50)) if len(vals) else None,
                                     p95=float(np.percentile(vals, 95)) if len(vals) else None))
    dec = {sp: len({(r["game"], r["decision"]) for r in rs}) for sp, rs in by.items()}
    smp = [r for r in by.get(sample_split, [])][: sample_n * 4]
    return TrainingView(table="move_samples_m", decisions_by_split=[Count(key=k, n=v) for k, v in dec.items()], stats=stats,
                        sample=smp)


# the built front end (npm run build in ui/): its files, and index.html for the app's own routes (/matches/..., ...)
DIST = ROOT / "ui" / "dist"


@app.get("/{path:path}", include_in_schema=False)
def ui(path: str):
    if path.startswith("api/"):
        raise HTTPException(404, "no such endpoint")
    f = (DIST / path).resolve()
    if path and f.is_file() and DIST.resolve() in f.parents:
        return FileResponse(f)
    if not (DIST / "index.html").exists():
        raise HTTPException(404, "the front end isn't built: cd ui && npm install && npm run build")
    return FileResponse(DIST / "index.html")
