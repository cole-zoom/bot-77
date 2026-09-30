"""An approval page for cut sprite slices: keep / reject each, and name the entity for cards
with more than one (Goblinstein's Doctor and Monster). Exports JSON like the review pages."""

from __future__ import annotations

import base64
import json
from pathlib import Path

ENTITIES = {"Goblinstein": ["goblinstein-monster", "goblinstein-doctor"],
            "Spirit Empress": ["spirit-empress-ground", "spirit-empress-air"],
            "Tombstone": ["tombstone-hero", "tomb-queen"]}


KEEP_SLICE = "Keep = a clean cut-out of just that unit (no other units, text or background)."
KEEP_CANDIDATE = "Keep = this really is that card, and the unit is mostly visible. Background is fine: approved ones get cut out with SAM next."


def write_gallery(manifest_path: Path, out: Path, hint: str = KEEP_SLICE) -> int:
    items = json.loads(manifest_path.read_text())
    data = []
    for k, it in enumerate(items):
        png = Path(it["file"]).read_bytes()
        mime = "image/jpeg" if it["file"].endswith(".jpg") else "image/png"
        card = it["card"]
        data.append({"id": Path(it["file"]).name, "card": card, "t": it["t_video"], "match": it["match_id"],
                     "entities": ENTITIES.get(card, [card.lower().replace(" ", "-")]),
                     "default": it.get("entity") or ENTITIES.get(card, [card.lower().replace(" ", "-")])[0],
                     "caption": it.get("caption", ""),
                     "img": f"data:{mime};base64," + base64.b64encode(png).decode()})
    page = GALLERY.replace("__HINT__", hint).replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return len(data)


GALLERY = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Slice approval</title>
<style>
:root { --bg:#f6f7f9; --panel:#fff; --ink:#1b1f24; --muted:#5d6673; --line:#e2e5ea; --good:#1f8a4c; --bad:#c43d3d; --accent:#6b3fd4; }
@media (prefers-color-scheme: dark) { :root { --bg:#111418; --panel:#1a1e24; --ink:#e8eaed; --muted:#9aa3ad; --line:#2c323a; --good:#52c585; --bad:#f07575; --accent:#a98bff; } }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
header { position:sticky; top:0; background:var(--panel); border-bottom:1px solid var(--line); padding:12px 16px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; z-index:2; }
h1 { font-size:16px; margin:0; } .muted { color:var(--muted); }
button, select { font:inherit; color:inherit; background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:4px 10px; cursor:pointer; }
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
main { padding:16px; display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:12px; }
.card { background:var(--panel); border:2px solid var(--line); border-radius:10px; padding:8px; display:flex; flex-direction:column; gap:6px; }
.card.keep { border-color:var(--good); } .card.reject { border-color:var(--bad); opacity:.55; }
.img { height:150px; display:flex; align-items:center; justify-content:center; border-radius:6px;
  background: repeating-conic-gradient(#8883 0% 25%, transparent 0% 50%) 50% / 16px 16px; }
.img img { max-width:100%; max-height:150px; image-rendering:auto; }
.row { display:flex; gap:6px; } .row button { flex:1; }
.keep .k { background:var(--good); border-color:var(--good); color:#fff; }
.reject .r { background:var(--bad); border-color:var(--bad); color:#fff; }
</style></head><body>
<header><h1>Slice approval</h1><span class="muted" id="progress"></span>
<span class="muted">__HINT__</span>
<button class="primary" id="export">Export</button></header>
<main id="main"></main>
<script>
const DATA = __DATA__;
const KEY = "bot77-slices-" + DATA.map(d => d.id).join("|");
let state = {}; try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} render(); };
function render() {
  const main = document.getElementById("main"); main.replaceChildren();
  for (const d of DATA) {
    const s = state[d.id] || { entity: d.default };
    const card = document.createElement("div"); card.className = "card " + (s.verdict || "");
    card.innerHTML = `<div class="img"><img alt="${d.card}" src="${d.img}"></div>
      <div><strong>${d.card}</strong> <span class="muted">${d.t}s</span></div>
      <div class="muted" style="font-size:12px">${d.caption}</div>`;
    if (d.entities.length > 1) {
      const sel = document.createElement("select");
      for (const e of d.entities) { const o = document.createElement("option"); o.value = e; o.textContent = e; o.selected = s.entity === e; sel.append(o); }
      sel.onchange = () => { state[d.id] = { ...s, entity: sel.value }; save(); };
      card.append(sel);
    }
    const row = document.createElement("div"); row.className = "row";
    for (const [cls, label, v] of [["k", "✓ Keep", "keep"], ["r", "✗ Reject", "reject"]]) {
      const b = document.createElement("button"); b.className = cls; b.textContent = label;
      b.onclick = () => { state[d.id] = { ...s, verdict: s.verdict === v ? undefined : v }; save(); }; row.append(b);
    }
    card.append(row); main.append(card);
  }
  const done = DATA.filter(d => (state[d.id] || {}).verdict).length;
  document.getElementById("progress").textContent = `${done} / ${DATA.length} decided`;
}
document.getElementById("export").onclick = () => {
  const out = DATA.map(d => ({ id: d.id, card: d.card, ...(state[d.id] || { entity: d.default }) }));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify({ exported_at: new Date().toISOString(), slices: out }, null, 1)], { type: "application/json" }));
  a.download = "slice_review.json"; a.click();
};
render();
</script></body></html>
"""
