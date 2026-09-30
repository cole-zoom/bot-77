"""Click-to-cut: for cards the old detector can't box (e.g. Goblinstein), show frames after a
corrected play and let the reviewer click each unit; SAM cuts the sprite from the click point."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import lancedb

from bot77.calibrate import read_frame
from bot77.layout import Layout

OFFSETS_S = (1.5, 2.5, 3.5, 5.0, 7.0, 9.0)  # after the play (once the deploy clock is gone), while the units are alive
SHOW_W = 620


def write_click_page(truth_path: Path, card: str, entities: list[str], out: Path, db_dir: Path = Path("data/lancedb")) -> int:
    truth = json.loads(truth_path.read_text())
    db = lancedb.connect(db_dir)
    video = db.open_table("videos").search().where(f"video_id = '{truth['video_id']}'").to_arrow().to_pylist()[0]
    layout = Layout.load(video["layout_id"])
    x0, y0, x1, y1 = layout.katacr_arena
    frames = []
    for item in (i for i in truth["items"] if i["label"] == card):
        for dt in OFFSETS_S:
            t = round(item["t_video"] + dt, 2)
            f = read_frame(Path(video["path"]), t)
            s = layout.scale(f.shape[1])
            cx0, cy0 = max(int(x0 * s), 0), max(int(y0 * s), 0)
            crop = f[cy0:int(y1 * s), cx0:int(x1 * s)]
            k = SHOW_W / crop.shape[1]
            img = cv2.resize(crop, (SHOW_W, round(crop.shape[0] * k)), interpolation=cv2.INTER_AREA)
            jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            frames.append({"id": f"{item['match_id']}_{t:07.2f}", "video_id": truth["video_id"], "match_id": item["match_id"],
                           "t": t, "play_t": item["t_video"], "offset_x": cx0, "offset_y": cy0, "scale": k,
                           "crop_w": crop.shape[1], "crop_h": crop.shape[0],
                           "img": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()})
    page = CLICK_PAGE.replace("__DATA__", json.dumps({"card": card, "entities": entities, "frames": frames}).replace("</", "<\\/"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return len(frames)


CLICK_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Click to cut</title>
<style>
:root { --bg:#f6f7f9; --panel:#fff; --ink:#1b1f24; --muted:#5d6673; --line:#e2e5ea; --accent:#6b3fd4; }
@media (prefers-color-scheme: dark) { :root { --bg:#111418; --panel:#1a1e24; --ink:#e8eaed; --muted:#9aa3ad; --line:#2c323a; --accent:#a98bff; } }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
header { position:sticky; top:0; z-index:3; background:var(--panel); border-bottom:1px solid var(--line); padding:12px 16px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
h1 { font-size:16px; margin:0 8px 0 0; } .muted { color:var(--muted); }
button { font:inherit; color:inherit; background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:5px 12px; cursor:pointer; }
button[aria-pressed="true"] { background:var(--accent); border-color:var(--accent); color:#fff; }
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
main { padding:16px; display:grid; grid-template-columns:repeat(auto-fill,minmax(440px,1fr)); gap:14px; align-items:start; }
.frame { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px; max-width:100%; }
.frame .wrap { position:relative; display:inline-block; cursor:crosshair; }
.frame img { display:block; max-width:100%; border-radius:6px; }
.dot { position:absolute; width:18px; height:18px; margin:-9px 0 0 -9px; border-radius:50%; border:3px solid #fff; box-shadow:0 0 0 2px #000; pointer-events:none; }
.dot.e0 { background:#ff3b6b; } .dot.e1 { background:#2fd07a; } .dot.e2 { background:#ffb020; }
.bar { display:flex; justify-content:space-between; gap:8px; margin-bottom:6px; align-items:center; }
</style></head><body>
<header><h1 id="title"></h1><span class="muted">Click on:</span><span id="entities"></span>
<span class="muted" id="progress"></span><button class="primary" id="export">Export clicks</button></header>
<main id="main"></main>
<script>
const DATA = __DATA__;
const KEY = "bot77-clicks-" + DATA.card + "-" + DATA.frames.length;
let state = { clicks: {} }; try { state = JSON.parse(localStorage.getItem(KEY) || '{"clicks":{}}'); } catch (e) {}
let current = 0;
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} render(); };
document.getElementById("title").textContent = "Click to cut · " + DATA.card;
function renderEntities() {
  const box = document.getElementById("entities"); box.replaceChildren();
  DATA.entities.forEach((e, i) => { const b = document.createElement("button"); b.textContent = e; b.setAttribute("aria-pressed", String(i === current));
    b.onclick = () => { current = i; renderEntities(); }; box.append(b); });
}
function render() {
  const main = document.getElementById("main"); main.replaceChildren();
  for (const f of DATA.frames) {
    const clicks = state.clicks[f.id] || [];
    const div = document.createElement("div"); div.className = "frame";
    const bar = document.createElement("div"); bar.className = "bar";
    bar.innerHTML = f.card
      ? `<span><strong>${f.card}</strong> · ${Math.floor(f.t / 60)}:${String(Math.floor(f.t % 60)).padStart(2, "0")} <span class="muted">(skip if it isn't visible)</span></span>`
      : `<span><strong>${f.t}s</strong> <span class="muted">(${(f.t - f.play_t).toFixed(1)}s after the play)</span></span>`;
    const clear = document.createElement("button"); clear.textContent = "Clear"; clear.onclick = () => { delete state.clicks[f.id]; save(); }; bar.append(clear);
    const wrap = document.createElement("div"); wrap.className = "wrap";
    const img = document.createElement("img"); img.src = f.img; img.alt = "frame " + f.t;
    wrap.append(img);
    for (const c of clicks) { const d = document.createElement("div"); d.className = "dot e" + DATA.entities.indexOf(c.entity);
      d.style.left = (c.u * 100) + "%"; d.style.top = (c.v * 100) + "%"; wrap.append(d); }
    wrap.onclick = ev => { if (f.entities && !f.entities.includes(DATA.entities[current])) { current = DATA.entities.indexOf(f.entities[0]); renderEntities(); }
      const r = img.getBoundingClientRect(); const u = (ev.clientX - r.left) / r.width, v = (ev.clientY - r.top) / r.height;
      const list = (state.clicks[f.id] || []).filter(c => c.entity !== DATA.entities[current]);
      list.push({ entity: DATA.entities[current], u, v }); state.clicks[f.id] = list; save(); };
    div.append(bar, wrap); main.append(div);
  }
  const n = Object.values(state.clicks).reduce((a, l) => a + l.length, 0);
  document.getElementById("progress").textContent = `${n} clicks on ${Object.keys(state.clicks).length}/${DATA.frames.length} frames`;
}
document.getElementById("export").onclick = () => {
  const out = [];
  // u, v are fractions of the shown crop; frame px = offset + u * show_width / scale
  for (const f of DATA.frames) for (const c of state.clicks[f.id] || []) {
    out.push({ frame: f.id, video_id: f.video_id, match_id: f.match_id, card: f.card || DATA.card, t: f.t, entity: c.entity, u: c.u, v: c.v,
               offset_x: f.offset_x, offset_y: f.offset_y, scale: f.scale, crop_w: f.crop_w, crop_h: f.crop_h });
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify({ card: DATA.card, exported_at: new Date().toISOString(), clicks: out }, null, 1)], { type: "application/json" }));
  a.download = "clicks_" + DATA.card.toLowerCase().replace(/ /g, "_") + ".json"; a.click();
};
renderEntities(); render();
</script></body></html>
"""


def write_click_page_times(video: Path, layout_id: str, targets: list[dict], out: Path, video_id: str,
                           crop_src: tuple[int, int, int, int] | None = None, src_scale_to: tuple[int, int] | None = None,
                           show_w: int = SHOW_W) -> int:
    """Click page from explicit times. `targets`: {"card", "entities", "times": [...]}. For
    recordings where the game is a region of a bigger frame (iPhone Mirroring), `crop_src`
    (x, y, w, h) and `src_scale_to` (w, h) reproduce the converted video's frames exactly, so
    clicks line up with it."""
    frames = []
    for tg in targets:
        # a target may name its own video (pages that mix videos)
        t_video, t_layout_id, t_vid = tg.get("video", video), tg.get("layout", layout_id), tg.get("video_id", video_id)
        layout = Layout.load(t_layout_id)
        x0, y0, x1, y1 = layout.katacr_arena
        for t in tg["times"]:
            f = read_frame(Path(t_video), t)
            if crop_src:
                cx, cy, cw, ch = crop_src
                f = cv2.resize(f[cy:cy + ch, cx:cx + cw], src_scale_to, interpolation=cv2.INTER_LANCZOS4)
            s = layout.scale(f.shape[1])
            cx0, cy0 = max(int(x0 * s), 0), max(int(y0 * s), 0)
            crop = f[cy0:int(y1 * s), cx0:int(x1 * s)]
            k = show_w / crop.shape[1]
            img = cv2.resize(crop, (show_w, round(crop.shape[0] * k)), interpolation=cv2.INTER_AREA)
            jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            frames.append({"id": f"{t_vid}_{t:07.2f}", "video_id": t_vid, "match_id": tg.get("match_id"), "card": tg["card"],
                           "entities": tg["entities"], "t": t, "play_t": tg["times"][0], "offset_x": cx0, "offset_y": cy0,
                           "scale": k, "crop_w": crop.shape[1], "crop_h": crop.shape[0],
                           "img": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()})
    ents = list(dict.fromkeys(e for tg in targets for e in tg["entities"]))
    page = CLICK_PAGE.replace("__DATA__", json.dumps({"card": "Cole recording", "entities": ents, "frames": frames}).replace("</", "<\\/"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return len(frames)
