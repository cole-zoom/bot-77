"""Spot-check page for auto-labelled frames: a stratified random sample, each frame drawn with its new
(auto) box highlighted and the provisional boxes for other units thin. Cole judges the new box, and flags
frames whose other boxes are clearly wrong. Exports JSON like the other review pages."""

from __future__ import annotations

import base64
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2


def draw(img_path: Path, lbl_path: Path, classes: list[str]) -> bytes:
    im = cv2.imread(str(img_path))
    H, W = im.shape[:2]
    lines = lbl_path.read_text().splitlines()
    n_main = sum(1 for l in lines if l.split()[0] == lines[0].split()[0])  # the play's boxes come first, same class
    for k, l in list(enumerate(lines))[::-1]:  # main boxes last so they're on top
        c, cx, cy, w, h = l.split()
        cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        p0, p1 = (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2))
        if k < n_main:
            cv2.rectangle(im, p0, p1, (0, 0, 0), 5)
            cv2.rectangle(im, p0, p1, (0, 240, 255), 3)
            cv2.putText(im, classes[int(c)], (p0[0], max(p0[1] - 6, 14)), 0, 0.6, (0, 0, 0), 4)
            cv2.putText(im, classes[int(c)], (p0[0], max(p0[1] - 6, 14)), 0, 0.6, (0, 240, 255), 2)
        else:
            cv2.rectangle(im, p0, p1, (200, 200, 200), 1)
            cv2.putText(im, classes[int(c)][:14], (p0[0], p0[1] - 2), 0, 0.32, (230, 230, 230), 1)
    return cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def write_spotcheck(root: Path, out: Path, n: int = 100, seed: int = 0, manifests: list[Path] | None = None,
                    title: str = "Auto-label spot check") -> int:
    classes = json.loads(Path("config/classes_v2.json").read_text())
    files = manifests if manifests is not None else sorted(root.glob("manifest_*.json"))
    manifest = [x for f in files for x in json.loads(Path(f).read_text())]
    by = defaultdict(list)
    for x in manifest:
        by[x["entity"]].append(x)
    rng = random.Random(seed)
    pick, ents = [], sorted(by)
    while len(pick) < min(n, len(manifest)):  # round-robin over entities so rare ones show up
        for e in ents:
            if by[e] and len(pick) < n:
                pick.append(by[e].pop(rng.randrange(len(by[e]))))
    rng.shuffle(pick)
    data = [{"id": x["name"], "entity": x["entity"], "source": x["source"], "how": x["how"],
             "img": "data:image/jpeg;base64," + base64.b64encode(
                 draw(root / "images" / f"{x['name']}.jpg", root / "labels" / f"{x['name']}.txt", classes)).decode()}
            for x in pick]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/")).replace("Auto-label spot check", title))
    return len(data)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Auto-label spot check</title>
<style>
:root { --bg:#f6f7f9; --panel:#fff; --ink:#1b1f24; --muted:#5d6673; --line:#e2e5ea; --good:#1f8a4c; --bad:#c43d3d; --warn:#b7791f; --accent:#6b3fd4; }
@media (prefers-color-scheme: dark) { :root { --bg:#111418; --panel:#1a1e24; --ink:#e8eaed; --muted:#9aa3ad; --line:#2c323a; --good:#52c585; --bad:#f07575; --warn:#e8b04b; --accent:#a98bff; } }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
header { position:sticky; top:0; z-index:2; background:var(--panel); border-bottom:1px solid var(--line); padding:12px 16px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
h1 { font-size:16px; margin:0; } .muted { color:var(--muted); }
button { font:inherit; color:inherit; background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:4px 10px; cursor:pointer; white-space:nowrap; }
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
main { padding:16px; display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:14px; }
.card { background:var(--panel); border:2px solid var(--line); border-radius:10px; padding:8px; display:flex; flex-direction:column; gap:6px; }
.card.v-right { border-color:var(--good); } .card.v-wrong { border-color:var(--bad); } .card.v-off { border-color:var(--warn); }
.card img { width:100%; border-radius:6px; }
.row { display:flex; gap:6px; flex-wrap:wrap; }
button[aria-pressed="true"].r { background:var(--good); border-color:var(--good); color:#fff; }
button[aria-pressed="true"].w { background:var(--bad); border-color:var(--bad); color:#fff; }
button[aria-pressed="true"].o { background:var(--warn); border-color:var(--warn); color:#fff; }
label { font-size:12px; color:var(--muted); }
</style></head><body>
<header><h1>Auto-label spot check</h1><span class="muted" id="progress"></span>
<span class="muted">Judge the <b>yellow</b> box: is it on the unit named above it? Grey boxes are the old detector's — only flag them if clearly wrong.</span>
<button class="primary" id="export">Export</button></header>
<main id="main"></main>
<script>
const DATA = __DATA__;
const KEY = "bot77-spotcheck-" + DATA.map(d => d.id).join("|").length;
let state = {}; try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} render(); };
function render() {
  const main = document.getElementById("main"); main.replaceChildren();
  for (const d of DATA) {
    const s = state[d.id] || {};
    const c = document.createElement("div"); c.className = "card" + (s.verdict ? " v-" + s.verdict : "");
    c.innerHTML = `<img alt="${d.entity}" src="${d.img}"><div><strong>${d.entity}</strong> <span class="muted">${d.source} · ${d.how}</span></div>`;
    const row = document.createElement("div"); row.className = "row";
    for (const [cls, label, v, tip] of [["r","✓ Right","right","right unit, box fits"],["o","~ Box off","off","right unit, box too big/small/shifted"],["w","✗ Wrong","wrong","wrong unit or nothing there"]]) {
      const b = document.createElement("button"); b.className = cls; b.textContent = label; b.title = tip;
      b.setAttribute("aria-pressed", String(s.verdict === v));
      b.onclick = () => { state[d.id] = { ...s, verdict: s.verdict === v ? undefined : v }; save(); }; row.append(b);
    }
    const lab = document.createElement("label"); const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = !!s.others_bad;
    cb.onchange = () => { state[d.id] = { ...s, others_bad: cb.checked }; save(); }; lab.append(cb, " grey boxes clearly wrong");
    c.append(row, lab); main.append(c);
  }
  document.getElementById("progress").textContent = `${DATA.filter(d => (state[d.id] || {}).verdict).length} / ${DATA.length} judged`;
}
document.getElementById("export").onclick = () => {
  const out = DATA.map(d => ({ id: d.id, entity: d.entity, source: d.source, ...(state[d.id] || {}) }));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify({ exported_at: new Date().toISOString(), frames: out }, null, 1)], { type: "application/json" }));
  a.download = "spotcheck.json"; a.click();
};
render();
</script></body></html>
"""
