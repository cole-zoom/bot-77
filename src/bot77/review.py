"""Write a self-contained HTML page for reviewing a processed video's play events.

Verdicts autosave in the browser (localStorage) and export as JSON; put the export in
data/reviews/ so it can be scored against the detector.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import lancedb


def _b64(img: bytes | None) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(img).decode() if img else ""


def _clock(t: float) -> str:
    t = max(t, 0)
    return f"{int(t // 60)}:{int(t % 60):02d}"


def write_review(db_dir: Path, video_id: str, out: Path, match_indices: list[int] | None = None) -> int:
    db = lancedb.connect(db_dir)
    video = db.open_table("videos").search().where(f"video_id = '{video_id}'").to_arrow().to_pylist()[0]
    matches = sorted(db.open_table("matches").search().where(f"video_id = '{video_id}'").limit(1000)
                     .to_arrow().to_pylist(), key=lambda m: m["index"])
    events = sorted(db.open_table("events").search().where(f"video_id = '{video_id}'").limit(100_000)
                    .to_arrow().to_pylist(), key=lambda e: e["t_video"])
    if match_indices:
        matches = [m for m in matches if m["index"] in match_indices]
        keep = {m["match_id"] for m in matches}
        events = [e for e in events if e["match_id"] in keep]

    import hashlib

    run = hashlib.sha1("|".join(f"{e['event_id']}:{e['t_video']:.2f}:{e['card']}" for e in events).encode()).hexdigest()[:10]
    data = {
        "video_id": video_id,
        "run": run,
        "url": video["url"],
        "matches": [{
            "match_id": m["match_id"], "index": m["index"], "opponent": m["opponent_name"],
            "rating": m["opponent_rating"], "deck": m["deck_names"], "t_start": m["t_start"],
            "length": _clock(m["t_end"] - max(m["t_start"], 0)),
        } for m in matches],
        "events": [{
            "id": e["event_id"], "match_id": e["match_id"], "t_video": round(e["t_video"], 1),
            "clock": _clock(e["t_match"]), "kind": e["kind"], "card": e["card"], "form": e["form"],
            "cost": round(e["measured_cost"], 2), "before": round(e["elixir_before"], 2),
            "after": round(e["elixir_after"], 2),
            "confidence": e["confidence"], "notes": e["notes"],
            "hand": _b64(e["thumb_hand"]), "arena": _b64(e["thumb_arena"]),
        } for e in events],
    }
    title = f"Play review · {video.get('creator') or video_id}"
    page = TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return len(events)


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #f6f7f9; --panel: #ffffff; --ink: #1b1f24; --muted: #5d6673; --line: #e2e5ea;
  --accent: #6b3fd4; --good: #1f8a4c; --bad: #c43d3d; --warn: #b7791f;
  --good-bg: #e6f4ec; --bad-bg: #fbeaea; --warn-bg: #fdf3e1;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #111418; --panel: #1a1e24; --ink: #e8eaed; --muted: #9aa3ad; --line: #2c323a;
    --accent: #a98bff; --good: #52c585; --bad: #f07575; --warn: #e8b04b;
    --good-bg: #173324; --bad-bg: #3a1c1c; --warn-bg: #3a2e16; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
header { position: sticky; top: 0; z-index: 5; background: var(--panel); border-bottom: 1px solid var(--line); padding: 12px 16px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
header h1 { font-size: 16px; margin: 0 12px 0 0; }
.progress { color: var(--muted); }
.bar { flex: 1 1 160px; height: 6px; background: var(--line); border-radius: 3px; overflow: hidden; max-width: 280px; }
.bar > div { height: 100%; background: var(--accent); width: 0; }
button, select, input { font: inherit; color: inherit; }
button { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 5px 10px; cursor: pointer; }
button:hover { border-color: var(--accent); }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
select, input { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 4px 6px; }
main { padding: 16px; max-width: 1180px; margin: 0 auto; }
.match { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; margin-bottom: 20px; overflow: hidden; }
.match > h2 { font-size: 15px; margin: 0; padding: 12px 16px; border-bottom: 1px solid var(--line); display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: baseline; }
.deck { color: var(--muted); font-weight: normal; font-size: 13px; }
.event { display: grid; grid-template-columns: 70px minmax(150px, 1fr) 380px 220px; gap: 12px; padding: 10px 16px; border-bottom: 1px solid var(--line); align-items: start; }
.event:last-of-type { border-bottom: 0; }
.event.v-correct { background: var(--good-bg); }
.event.v-wrong { background: var(--bad-bg); }
.event.v-unsure { background: var(--warn-bg); }
.time a { color: var(--accent); font-weight: 600; text-decoration: none; font-variant-numeric: tabular-nums; }
.card { font-weight: 600; }
.form { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; padding: 1px 6px; border-radius: 4px; background: var(--line); margin-left: 6px; }
.meta { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.pill { display: inline-block; font-size: 11px; padding: 1px 7px; border-radius: 10px; border: 1px solid currentColor; }
.c-high { color: var(--good); } .c-medium { color: var(--warn); } .c-low { color: var(--bad); }
.notes { color: var(--muted); font-size: 12px; margin-top: 4px; }
.thumbs { display: flex; gap: 6px; }
.thumbs img { display: block; border-radius: 4px; border: 1px solid var(--line); cursor: zoom-in; }
.thumbs .hand { width: 260px; height: auto; align-self: flex-start; }
.thumbs .arena { width: 110px; height: auto; }
.verdict { display: flex; flex-direction: column; gap: 6px; }
.verdict .btns { display: flex; gap: 6px; }
.verdict .btns button[aria-pressed="true"].ok { background: var(--good); border-color: var(--good); color: #fff; }
.verdict .btns button[aria-pressed="true"].no { background: var(--bad); border-color: var(--bad); color: #fff; }
.verdict .btns button[aria-pressed="true"].meh { background: var(--warn); border-color: var(--warn); color: #fff; }
.fix { display: none; gap: 6px; flex-direction: column; }
.v-wrong .fix { display: flex; }
.missing { padding: 12px 16px; background: var(--bg); display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.missing ul { flex-basis: 100%; margin: 4px 0 0; padding-left: 18px; }
.hidden { display: none !important; }
dialog { border: 0; padding: 0; background: transparent; }
dialog img { max-width: 92vw; max-height: 92vh; border-radius: 8px; }
dialog::backdrop { background: rgba(0,0,0,.7); }
@media (max-width: 760px) {
  .event { grid-template-columns: 60px 1fr; }
  .thumbs, .verdict { grid-column: 1 / -1; }
}
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <span class="progress" id="progress"></span>
  <div class="bar"><div id="bar"></div></div>
  <label><input type="checkbox" id="onlyUnsure"> only medium/low confidence</label>
  <button id="import">Import…</button>
  <button class="primary" id="export">Export verdicts</button>
  <input type="file" id="importFile" accept="application/json" class="hidden">
</header>
<main id="main"></main>
<dialog id="zoom"><img alt=""></dialog>
<script>
const DATA = __DATA__;
// Verdicts are tied to this exact detection run: event ids are reused when detection is re-run.
const KEY = "bot77-review-" + DATA.video_id + "-" + DATA.run;
let state = { verdicts: {}, missing: [] };
try { state = Object.assign(state, JSON.parse(localStorage.getItem(KEY) || "{}")); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} render(); };

const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v; else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined) n.append(k.nodeType ? k : document.createTextNode(k));
  return n;
};
const yt = t => `${DATA.url}&t=${Math.max(0, Math.floor(t - 2))}s`;
const zoom = src => { const d = document.getElementById("zoom"); d.querySelector("img").src = src; d.showModal(); };
document.getElementById("zoom").addEventListener("click", e => e.currentTarget.close());

function eventRow(ev, deck) {
  const v = state.verdicts[ev.id] || {};
  const set = verdict => { state.verdicts[ev.id] = Object.assign({}, v, { verdict: v.verdict === verdict ? undefined : verdict }); save(); };
  const label = ev.kind === "ability" ? `${ev.card || "?"} ability` : (ev.card || "Unknown card");
  const fixSel = el("select", { onchange: e => { state.verdicts[ev.id] = Object.assign({}, v, { verdict: "wrong", correct: e.target.value }); save(); } },
    el("option", { value: "" }, "What was it really?"),
    deck.map(c => el("option", { value: c, selected: v.correct === c ? "" : null }, c)),
    el("option", { value: "__ability", selected: v.correct === "__ability" ? "" : null }, "Champion/hero ability"),
    el("option", { value: "__not_a_play", selected: v.correct === "__not_a_play" ? "" : null }, "Not a play at all"));
  const note = el("input", { placeholder: "note (optional)", value: v.note || "",
    onchange: e => { state.verdicts[ev.id] = Object.assign({}, state.verdicts[ev.id] || {}, { note: e.target.value }); save(); } });
  return el("div", { class: "event" + (v.verdict ? " v-" + v.verdict : ""), "data-conf": ev.confidence },
    el("div", { class: "time" }, el("a", { href: yt(ev.t_video), target: "_blank", rel: "noopener", title: "Open on YouTube, 2s before" }, ev.clock),
       el("div", { class: "meta" }, `${ev.t_video}s`)),
    el("div", {},
      el("div", {}, el("span", { class: "card" }, label), ev.form && ev.form !== "normal" ? el("span", { class: "form" }, ev.form) : null),
      el("div", { class: "meta" }, `elixir ${ev.before} → ${ev.after} · spent ${ev.cost}`),
      el("span", { class: `pill c-${ev.confidence}` }, ev.confidence),
      ev.notes.length ? el("div", { class: "notes" }, ev.notes.join("; ")) : null),
    el("div", { class: "thumbs" },
      ev.hand ? el("img", { class: "hand", src: ev.hand, alt: "hand before", onclick: () => zoom(ev.hand) }) : null,
      ev.arena ? el("img", { class: "arena", src: ev.arena, alt: "arena after", onclick: () => zoom(ev.arena) }) : null),
    el("div", { class: "verdict" },
      el("div", { class: "btns" },
        el("button", { class: "ok", "aria-pressed": String(v.verdict === "correct"), onclick: () => set("correct") }, "✓ Right"),
        el("button", { class: "no", "aria-pressed": String(v.verdict === "wrong"), onclick: () => set("wrong") }, "✗ Wrong"),
        el("button", { class: "meh", "aria-pressed": String(v.verdict === "unsure"), onclick: () => set("unsure") }, "?")),
      el("div", { class: "fix" }, fixSel), note));
}

function missingBox(m) {
  const t = el("input", { placeholder: "m:ss", size: 5 });
  const c = el("select", {}, m.deck.map(x => el("option", { value: x }, x)), el("option", { value: "__ability" }, "Champion/hero ability"));
  const n = el("input", { placeholder: "note" });
  const mine = state.missing.filter(x => x.match_id === m.match_id);
  return el("div", { class: "missing" },
    el("strong", {}, "Missed a play?"), t, c, n,
    el("button", { onclick: () => { if (!t.value) return; state.missing.push({ match_id: m.match_id, clock: t.value, card: c.value, note: n.value }); save(); } }, "Add"),
    mine.length ? el("ul", {}, mine.map(x => el("li", {}, `${x.clock} ${x.card}${x.note ? " — " + x.note : ""} `,
      el("button", { onclick: () => { state.missing.splice(state.missing.indexOf(x), 1); save(); } }, "remove")))) : null);
}

function render() {
  const only = document.getElementById("onlyUnsure").checked;
  const main = document.getElementById("main"); main.replaceChildren();
  for (const m of DATA.matches) {
    const evs = DATA.events.filter(e => e.match_id === m.match_id && (!only || e.confidence !== "high"));
    main.append(el("section", { class: "match" },
      el("h2", {}, `Match ${m.index} · vs ${m.opponent || "?"}${m.rating ? " (" + m.rating + ")" : ""} · ${m.length}`,
        el("span", { class: "deck" }, m.deck.join(" · "))),
      evs.map(e => eventRow(e, m.deck)), missingBox(m)));
  }
  const done = DATA.events.filter(e => (state.verdicts[e.id] || {}).verdict).length;
  document.getElementById("progress").textContent = `${done} / ${DATA.events.length} reviewed`;
  document.getElementById("bar").style.width = (100 * done / Math.max(DATA.events.length, 1)) + "%";
}

document.getElementById("onlyUnsure").addEventListener("change", render);
document.getElementById("export").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify({ video_id: DATA.video_id, run: DATA.run, exported_at: new Date().toISOString(), ...state }, null, 1)], { type: "application/json" });
  const a = el("a", { href: URL.createObjectURL(blob), download: `review_${DATA.video_id}.json` }); a.click();
});
document.getElementById("import").addEventListener("click", () => document.getElementById("importFile").click());
document.getElementById("importFile").addEventListener("change", async e => {
  const f = e.target.files[0]; if (!f) return;
  const got = JSON.parse(await f.text()); state = { verdicts: got.verdicts || {}, missing: got.missing || [] }; save();
});
render();
</script>
</body>
</html>
"""


def score_review(db_dir: Path, review_json: Path) -> dict:
    """Precision / recall of detected plays against a reviewer's exported verdicts.

    precision = right / (right + wrong); recall = right / (right + missed plays).
    Unreviewed and "unsure" events are left out."""
    got = json.loads(review_json.read_text())
    db = lancedb.connect(db_dir)
    events = db.open_table("events").search().where(f"video_id = '{got['video_id']}'").limit(100_000) \
        .select(["event_id", "confidence", "kind"]).to_arrow().to_pylist()
    conf = {e["event_id"]: e["confidence"] for e in events}
    verdicts = {k: v.get("verdict") for k, v in got.get("verdicts", {}).items() if v.get("verdict")}

    def rates(ids):
        right = sum(verdicts.get(i) == "correct" for i in ids)
        wrong = sum(verdicts.get(i) == "wrong" for i in ids)
        return {"right": right, "wrong": wrong, "precision": round(right / (right + wrong), 3) if right + wrong else None}

    out = {"reviewed": len(verdicts), "total": len(events), "missed": len(got.get("missing", [])),
           "overall": rates(list(conf)), "by_confidence": {c: rates([i for i, x in conf.items() if x == c])
                                                            for c in ("high", "medium", "low")}}
    right = out["overall"]["right"]
    out["recall"] = round(right / (right + out["missed"]), 3) if right + out["missed"] else None
    return out


ABILITY = "__ability"
NOT_A_PLAY = "__not_a_play"
MATCH_WINDOW_S = 1.5  # a detected event matches a true one within this much video time


def truth_from_review(db_dir: Path, review_json: Path) -> dict:
    """Ground truth for fully reviewed matches, independent of event ids, so later detector
    runs can be scored against it. Each item: {match_id, t_video, label}, where label is a
    card name or "__ability". Only matches where every event got a verdict are included."""
    got = json.loads(review_json.read_text())
    db = lancedb.connect(db_dir)
    events = db.open_table("events").search().where(f"video_id = '{got['video_id']}'").limit(100_000) \
        .select(["event_id", "match_id", "t_video", "kind", "card"]).to_arrow().to_pylist()
    starts = {m["match_id"]: m["t_start"] for m in db.open_table("matches").search()
              .where(f"video_id = '{got['video_id']}'").to_arrow().to_pylist()}
    verdicts = got.get("verdicts", {})

    complete = sorted({e["match_id"] for e in events}
                      - {e["match_id"] for e in events if not verdicts.get(e["event_id"], {}).get("verdict")})
    items, unsure = [], 0
    for e in events:
        if e["match_id"] not in complete:
            continue
        v = verdicts[e["event_id"]]
        detected = ABILITY if e["kind"] == "ability" else e["card"]
        if v["verdict"] == "correct":
            label = detected
        elif v["verdict"] == "wrong":
            label = v.get("correct") or None
        else:
            unsure += 1
            continue
        if label and label != NOT_A_PLAY:
            items.append({"match_id": e["match_id"], "t_video": round(e["t_video"], 2), "label": label,
                          "note": v.get("note", "")})
    for mis in got.get("missing", []):
        if mis["match_id"] in complete:
            mm, ss = mis["clock"].split(":")
            items.append({"match_id": mis["match_id"], "t_video": round(starts[mis["match_id"]] + int(mm) * 60 + int(ss), 2),
                          "label": mis["card"], "note": mis.get("note", ""), "approximate_time": True})
    items.sort(key=lambda x: (x["match_id"], x["t_video"]))
    return {"video_id": got["video_id"], "matches": complete, "items": items, "skipped_unsure": unsure}


def score_against_truth(db_dir: Path, truth: dict) -> dict:
    """Greedy one-to-one matching of detected events to true ones (same label, within
    MATCH_WINDOW_S; approximate-time truth items allow 3x that)."""
    db = lancedb.connect(db_dir)
    events = db.open_table("events").search().where(f"video_id = '{truth['video_id']}'").limit(100_000) \
        .select(["match_id", "t_video", "kind", "card", "confidence"]).to_arrow().to_pylist()
    events = [e for e in events if e["match_id"] in truth["matches"]]
    free = list(truth["items"])
    tp, fp = [], []
    for e in sorted(events, key=lambda e: e["t_video"]):
        label = ABILITY if e["kind"] == "ability" else e["card"]
        best = None
        for k, t in enumerate(free):
            window = MATCH_WINDOW_S * (3 if t.get("approximate_time") else 1)
            d = abs(t["t_video"] - e["t_video"])
            if t["match_id"] == e["match_id"] and t["label"] == label and d <= window and (best is None or d < best[0]):
                best = (d, k)
        if best:
            tp.append(e)
            free.pop(best[1])
        else:
            fp.append(e)
    by_conf = {}
    for c in ("high", "medium", "low"):
        a, b = sum(e["confidence"] == c for e in tp), sum(e["confidence"] == c for e in fp)
        by_conf[c] = {"right": a, "wrong": b, "precision": round(a / (a + b), 3) if a + b else None}
    n_tp = len(tp)
    return {
        "matches": truth["matches"], "true_events": len(truth["items"]), "detected": len(events),
        "right": n_tp, "wrong": len(fp), "missed": len(free),
        "precision": round(n_tp / len(events), 3) if events else None,
        "recall": round(n_tp / len(truth["items"]), 3) if truth["items"] else None,
        "by_confidence": by_conf,
        "wrong_events": [(e["match_id"][-3:], round(e["t_video"], 1), e["kind"], e["card"], e["confidence"]) for e in fp],
        "missed_events": [(t["match_id"][-3:], t["t_video"], t["label"]) for t in free],
    }
