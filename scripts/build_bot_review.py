"""Review pages for the bot's games (playbook step 0, 2026-10-21): one card per bot play with the frame at the play
(the bot's tap marked) and 2 s later. Cole marks OK / Mistake; for a mistake he can click where the card should have
gone, pick the card it should have played (or "wait"), and write a note. Export -> JSON for config/playbook_rules.yaml.
Pages hold up to 150 plays (Cole's cap), newest games first.
--prefill seeds verdicts from exported reviews (e.g. Claude's, skill review-bot-plays, 2026-10-22): they show with a
"Claude" tag until Cole agrees or changes them; the export keeps who decided each play (by).

  uv run python scripts/build_bot_review.py [--since 2026-10-21] -> data/reviews/bot_<date>/review_<n>.html
"""
import argparse, datetime, json
from pathlib import Path

import lancedb

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bot play review</title><style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1b1f24;--muted:#5d6673;--line:#e2e5ea;--good:#1f8a4c;--bad:#c43d3d;--accent:#6b3fd4}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--panel:#1a1e24;--ink:#e8eaed;--muted:#9aa3ad;--line:#2c323a;--good:#52c585;--bad:#f07575;--accent:#a98bff}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--panel);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0}.muted{color:var(--muted)}button,select,input{font:inherit;color:inherit;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
main{padding:16px;display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:14px}
.card{background:var(--panel);border:2px solid var(--line);border-radius:10px;padding:10px;display:flex;flex-direction:column;gap:8px}
.card.ok{border-color:var(--good)}.card.bad{border-color:var(--bad)}
.imgs{display:flex;gap:6px}.wrap{position:relative;flex:1}.wrap img{width:100%;display:block;border-radius:6px}
.dot{position:absolute;width:18px;height:18px;margin:-9px 0 0 -9px;border-radius:50%;pointer-events:none}
.bot{border:3px solid #ffd400}.fix{border:3px solid #18e06a;background:#18e06a55}
.tag{font-size:12px;padding:1px 6px;border-radius:9px;background:var(--accent);color:#fff}
textarea.note{width:100%;box-sizing:border-box;min-height:4.5em;resize:vertical;font-family:inherit;font-size:15px;line-height:1.45;color:inherit;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:6px 8px}
.row{display:flex;gap:6px;flex-wrap:wrap;align-items:center}.ok .k{background:var(--good);color:#fff}.bad .m{background:var(--bad);color:#fff}
</style></head><body>
<header><h1>Bot play review __N__</h1><span class="muted" id="progress"></span>
<span class="muted">OK if the play was fine. Mistake: click the left image where it should have gone, pick the card (or wait), add a note.</span>
<button class="primary" id="export">Export</button></header><main id="main"></main>
<script>
const DATA=__DATA__;const PREFILL=__PREFILL__;const KEY="bot77-botreview-"+DATA.map(d=>d.id).join("|").length+"-"+(DATA[0]||{}).id;
let S={};try{S=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
for(const [id,p] of Object.entries(PREFILL))if(!S[id])S[id]=p;
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){};render()};
function render(){const main=document.getElementById("main");main.replaceChildren();let done=0;
for(const d of DATA){const s=S[d.id]||{};if(s.verdict)done++;
const c=document.createElement("div");c.className="card "+(s.verdict||"");
const th=d.threats.map(t=>t[0]).join(", ")||"none";
c.innerHTML=`<div>${s.by==="claude"?`<span class="tag">Claude · ${s.confidence||"?"}</span> `:""}<strong>${d.card}</strong> <span class="muted">${d.reason} · elixir ${d.elixir} · ${d.t}s · threats: ${th}</span></div>
<div class="imgs"><div class="wrap at"><img src="${d.at}" alt="at the play"><div class="dot bot" style="left:${d.tap[0]*100}%;top:${d.tap[1]*100}%"></div>
${s.spot?`<div class="dot fix" style="left:${s.spot[0]*100}%;top:${s.spot[1]*100}%"></div>`:""}</div>
<div class="wrap"><img src="${d.after}" alt="2 s later" onerror="this.parentNode.textContent='(game ended)'"></div></div>`;
c.querySelector(".at").onclick=e=>{const r=e.currentTarget.getBoundingClientRect();S[d.id]={...s,by:"cole",verdict:"bad",spot:[(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]};save()};
const row=document.createElement("div");row.className="row";
const ok=document.createElement("button");ok.className="k";ok.textContent="OK";ok.onclick=()=>{S[d.id]={verdict:"ok",by:"cole"};save()};
const bad=document.createElement("button");bad.className="m";bad.textContent="Mistake";bad.onclick=()=>{S[d.id]={...s,by:"cole",verdict:"bad"};save()};
const sel=document.createElement("select");for(const o of ["(card it should play)",...d.hand.filter(Boolean),"wait (no play)"]){const op=document.createElement("option");op.textContent=o;op.selected=s.card===o;sel.append(op)}
sel.onchange=()=>{S[d.id]={...s,by:"cole",verdict:"bad",card:sel.value};save()};
const note=document.createElement("textarea");note.className="note";note.rows=3;note.placeholder="note";note.value=s.note||"";note.onchange=()=>{S[d.id]={...s,by:"cole",verdict:s.verdict||"bad",note:note.value};save()};
row.append(ok,bad,sel);if(s.by==="claude"){const ag=document.createElement("button");ag.textContent="Agree";ag.onclick=()=>{S[d.id]={...s,by:"cole"};save()};row.append(ag)}c.append(row,note);main.append(c)}
const cl=DATA.filter(d=>(S[d.id]||{}).by==="claude").length;
document.getElementById("progress").textContent=`${done}/${DATA.length} reviewed`+(cl?` · ${cl} by Claude to check`:"")}
document.getElementById("export").onclick=()=>{const out=DATA.filter(d=>S[d.id]).map(d=>({id:d.id,game:d.game,k:d.k,card:d.card,reason:d.reason,
...S[d.id],spot_px:S[d.id].spot?[Math.round(S[d.id].spot[0]*d.frame[0]),Math.round(S[d.id].spot[1]*d.frame[1])]:null}));
const b=new Blob([JSON.stringify({exported_at:new Date().toISOString(),plays:out},null,1)],{type:"application/json"});
const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="bot_review.json";a.click()};
render();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="", help="only games whose id date is on/after this (YYYY-MM-DD)")
    ap.add_argument("--per-page", type=int, default=150)
    ap.add_argument("--losses", action="store_true", help="only lost games (data/live/game_results.json or bot_games)")
    ap.add_argument("--prefill", nargs="*", default=[], help="exported review JSONs to seed verdicts from (later files win)")
    ap.add_argument("--max-plays", type=int, default=0, help="whole games, newest first, up to this many plays")
    a = ap.parse_args()

    db = lancedb.connect("data/lancedb")
    cards = {c["id"]: c["name"] for c in db.open_table("cards").search().where("kind = 'deck_card'").limit(1000)
             .select(["id", "name"]).to_arrow().to_pylist()}
    rows = db.open_table("bot_actions").to_arrow().to_pylist()
    rows = [r for r in rows if r.get("shot") and (not a.since or r["game_id"][4:14] >= a.since)]
    if a.losses:
        res = json.loads(Path("data/live/game_results.json").read_text()) if Path("data/live/game_results.json").exists() else {}
        for g in db.open_table("bot_games").to_arrow().to_pylist():
            if g.get("result") in ("win", "loss"):
                res[g["game_id"]] = g["result"]
        rows = [r for r in rows if res.get(r["game_id"]) == "loss"]
    rows.sort(key=lambda r: (r["game_id"], r["t"]), reverse=True)
    if a.max_plays:  # whole games, newest first
        keep, n = set(), 0
        for gid in sorted({r["game_id"] for r in rows}, reverse=True):
            k = sum(1 for r in rows if r["game_id"] == gid)
            if n + k > a.max_plays:
                break
            keep.add(gid); n += k
        rows = [r for r in rows if r["game_id"] in keep]
    rows.sort(key=lambda r: (r["game_id"], r["t"]))  # each game in play order
    out = Path(f"data/reviews/bot_{datetime.date.today()}"); out.mkdir(parents=True, exist_ok=True)
    games_dir = Path("data/raw/live_games").resolve()
    prefill = {}
    for f in a.prefill:
        ex = json.loads(Path(f).read_text())
        for q in ex["plays"]:
            prefill[q["id"]] = {"verdict": q["verdict"], "card": q.get("card"), "note": q.get("note", ""),
                                "by": ex.get("reviewer", "cole"), "confidence": q.get("confidence")}
    pages = [rows[i:i + a.per_page] for i in range(0, len(rows), a.per_page)]
    for n, page in enumerate(pages, 1):
        items = []
        for r in page:
            d = games_dir / r["game_id"]
            taps = json.loads(r["taps"])
            hand = [cards.get(c) for c in json.loads(r["hand"])]
            items.append({"id": f"{r['game_id']}:{r['k']}", "game": r["game_id"], "k": r["k"], "t": round(r["t"], 1),
                          "card": r["card"], "reason": r["reason"], "elixir": round(r["elixir"] or 0, 1),
                          "threats": json.loads(r.get("threats") or "[]"), "hand": hand,
                          "at": (d / r["shot"]).as_uri(), "after": (d / f"{r['k']:03d}_after.jpg").as_uri(),
                          "tap": [taps[2] / r["frame_w"], taps[3] / r["frame_h"]], "frame": [r["frame_w"], r["frame_h"]]})
        (out / f"review_{n}.html").write_text(PAGE.replace("__DATA__", json.dumps(items)).replace("__PREFILL__", json.dumps({i["id"]: prefill[i["id"]] for i in items if i["id"] in prefill})).replace("__N__", f"{n}/{len(pages)}"))
    print(len(rows), "plays ->", [str(out / f"review_{n}.html") for n in range(1, len(pages) + 1)])


if __name__ == "__main__":
    main()
