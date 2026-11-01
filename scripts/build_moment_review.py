"""The yes / no page for mined moments (scripts/mine_moments.py; 2026-10-31, Cole: "set it up in a way that I can just mark
yes or no so we can train on all of the moments where our model is failing").

  uv run --no-sync python scripts/build_moment_review.py data/reviews/mined_2026-10-31 [--port 8792]
  uv run --no-sync python scripts/serve_review.py data/reviews/mined_2026-10-31 --port 8792

One moment at a time: a clip of the bot's own recording from LEAD s before the moment to TAIL s after, looping (the
Battle Log replay of the game one key away), what the miner saw, and Y (a real mistake) / N (fine). Optional: the card
it should have played or "wait", and a note. Keyboard only works: Y, N, S skip, ← → previous / next, R the replay
video, Space pause. Saved in the browser; Export writes moments_review.json for scripts/moments_to_examples.py.
Videos: as build_loss_review (remuxed beside the originals, offsets read off the game's own timer).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_loss_review as blr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEAD, TAIL = 6.0, 6.0
KIND = {"missed_spell": "Missed spell", "counter_spent": "Spent the counter", "push_through": "Push got through",
        "tower_lost": "Tower lost",
        "leaked": "Leaked elixir"}

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Moment Review</title><style>
:root{--bg:#f4f5f7;--panel:#fff;--ink:#1b1f24;--muted:#5d6673;--line:#e2e5ea;--good:#1f8a4c;--bad:#c43d3d;--accent:#6b3fd4}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--panel:#1a1e24;--ink:#e8eaed;--muted:#9aa3ad;--line:#2c323a;--good:#52c585;--bad:#f07575;--accent:#a98bff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0}.muted{color:var(--muted)}.small{font-size:12px}
button,select,textarea{font:inherit;color:inherit;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:5px 10px}
button{cursor:pointer}button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.bar{height:6px;background:var(--line);border-radius:3px;flex:1;min-width:120px;overflow:hidden}.bar div{height:100%;background:var(--accent)}
main{display:grid;grid-template-columns:minmax(280px,480px) minmax(280px,520px);gap:20px;padding:16px;justify-content:center}
@media (max-width:860px){main{grid-template-columns:1fr}}
video{width:100%;max-height:calc(100vh - 140px);background:#000;border-radius:8px}
.box{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;display:flex;flex-direction:column;gap:12px}
.kind{display:inline-block;font-size:12px;font-weight:700;padding:2px 8px;border-radius:9px;background:var(--accent);color:#fff}
.what{font-size:17px;font-weight:600}.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.yn button{flex:1;font-size:17px;font-weight:700;padding:12px}.yn .y.on{background:var(--bad);border-color:var(--bad);color:#fff}
.yn .n.on{background:var(--good);border-color:var(--good);color:#fff}textarea{width:100%;min-height:3.5em;resize:vertical}
kbd{font:700 12px ui-monospace,monospace;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px}
</style></head><body>
<header><h1>Moment review</h1><div class="bar"><div id="fill"></div></div><span class="muted small" id="prog"></span>
<select id="filter"><option value="">all kinds</option>__KINDS__</select><button class="primary" id="export">Export</button></header>
<main><div><video id="v" playsinline muted controls></video><div class="row small muted" style="margin-top:6px">
<span id="vsrc"></span><button id="swap">R: switch video</button><button id="slow">0.5×</button><button id="norm">1×</button></div></div>
<div class="box"><div><span class="kind" id="kind"></span> <span class="muted small" id="where"></span></div>
<div class="what" id="text"></div>
<div class="muted small">Is this a real mistake by the bot?</div>
<div class="row yn"><button class="y" id="yes">Y · yes, a mistake</button><button class="n" id="no">N · no, fine</button></div>
<div class="row"><select id="should"><option value="">should have… (optional)</option></select></div>
<textarea id="note" placeholder="Note (optional): what it should have done"></textarea>
<div class="row"><button id="prev">← prev</button><button id="skip">S skip</button><button id="next">next →</button></div>
<div class="muted small"><kbd>Y</kbd> yes · <kbd>N</kbd> no (both move on) · <kbd>S</kbd> skip · <kbd>←</kbd><kbd>→</kbd> move · <kbd>R</kbd> other video · <kbd>Space</kbd> pause</div></div></main>
<script>
const D=__DATA__,DECK=__DECK__,LEAD=__LEAD__,TAIL=__TAIL__,KEY="bot77-moments-"+D.moments.length+"-"+(D.moments[0]||{}).game+(D.moments[0]||{}).t;
let S={};try{S=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}};
const v=document.getElementById("v");let i=0,vid="bot";
const id=m=>m.game+"@"+m.t+"/"+m.kind;
const list=()=>{const f=document.getElementById("filter").value;return D.moments.map((m,k)=>k).filter(k=>!f||D.moments[k].kind===f)};
document.getElementById("should").innerHTML+= ["wait",...DECK].map(c=>`<option>${c}</option>`).join("");
function clip(){const m=D.moments[i],g=D.games[m.game]||{},src=vid==="bot"?g.bot_video:g.replay_video,off=vid==="bot"?g.bot_offset:g.replay_offset;
 document.getElementById("vsrc").textContent=src?(vid==="bot"?"bot's screen":"Battle Log replay"):"(no video)";
 if(!src){v.removeAttribute("src");return}
 const a=Math.max(0,m.t-(off==null?4:off)-LEAD),b=a+LEAD+TAIL;v.dataset.a=a;v.dataset.b=b;
 if(!v.src.endsWith(src))v.src=src;const go=()=>{v.currentTime=a;v.play().catch(()=>{})};if(v.readyState>=1)go();else v.onloadedmetadata=go}
v.ontimeupdate=()=>{if(v.currentTime>=+v.dataset.b)v.currentTime=+v.dataset.a};
function show(){const ks=list();if(!ks.includes(i))i=ks[0]??0;const m=D.moments[i],r=S[id(m)]||{};const g=D.games[m.game]||{};
 document.getElementById("kind").textContent=D.kinds[m.kind]||m.kind;
 document.getElementById("where").textContent=`${m.clock} · vs ${g.opp||m.opp} (${g.result||m.result}) · ${ks.indexOf(i)+1} of ${ks.length}`;
 document.getElementById("text").textContent=m.text;
 document.getElementById("yes").classList.toggle("on",r.v==="yes");document.getElementById("no").classList.toggle("on",r.v==="no");
 document.getElementById("should").value=r.should||"";document.getElementById("note").value=r.note||"";
 const done=D.moments.filter(m=>(S[id(m)]||{}).v).length;document.getElementById("prog").textContent=`${done} / ${D.moments.length} marked`;
 document.getElementById("fill").style.width=(100*done/D.moments.length)+"%";clip()}
function move(d){const ks=list(),k=ks.indexOf(i);i=ks[Math.max(0,Math.min(ks.length-1,k+d))];show()}
function mark(x){const m=D.moments[i];S[id(m)]={...(S[id(m)]||{}),v:x};save();move(1)}
document.getElementById("yes").onclick=()=>mark("yes");document.getElementById("no").onclick=()=>mark("no");
document.getElementById("skip").onclick=()=>move(1);document.getElementById("next").onclick=()=>move(1);document.getElementById("prev").onclick=()=>move(-1);
document.getElementById("swap").onclick=()=>{vid=vid==="bot"?"rep":"bot";clip()};
document.getElementById("slow").onclick=()=>v.playbackRate=0.5;document.getElementById("norm").onclick=()=>v.playbackRate=1;
document.getElementById("filter").onchange=show;
document.getElementById("should").onchange=e=>{const m=D.moments[i];S[id(m)]={...(S[id(m)]||{}),should:e.target.value};save()};
document.getElementById("note").oninput=e=>{const m=D.moments[i];S[id(m)]={...(S[id(m)]||{}),note:e.target.value};save()};
document.addEventListener("keydown",e=>{if(["TEXTAREA","INPUT","SELECT"].includes(e.target.tagName))return;const k=e.key.toLowerCase();
 if(k==="y")mark("yes");else if(k==="n")mark("no");else if(k==="s"||e.key==="ArrowRight")move(1);else if(e.key==="ArrowLeft")move(-1);
 else if(k==="r")document.getElementById("swap").click();else if(e.key===" "){e.preventDefault();v.paused?v.play():v.pause()}else return;e.preventDefault()});
document.getElementById("export").onclick=()=>{const out=D.moments.map(m=>({...m,...(S[id(m)]||{})}));
 const b=new Blob([JSON.stringify({by:"cole",exported:new Date().toISOString(),moments:out},null,1)],{type:"application/json"});
 const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="moments_review_"+new Date().toISOString().slice(0,10)+".json";a.click()};
show();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="the miner's output folder (moments.json)")
    a = ap.parse_args()
    out = Path(a.folder)
    data = json.loads((out / "moments.json").read_text())
    vid = out / "videos"
    vid.mkdir(exist_ok=True)
    for gid, g in data["games"].items():
        mkv = blr.GAMES / gid / "game.mkv"
        if mkv.exists() and "bot_video" not in g:
            link = vid / f"{gid}.mp4"
            if not link.exists():
                link.symlink_to(blr.remux(mkv))
            g["bot_video"], g["bot_offset"] = f"videos/{link.name}", blr.offset_of(link)
        rid = next((m["replay"] for m in data["moments"] if m["game"] == gid), None)
        if rid and "replay_video" not in g:
            date, n = rid.split("_")[1], rid.split("_r")[1].split("_")[0]
            mp4 = blr.REPLAYS / date / f"replay_{n}.mp4"
            if mp4.exists():
                link = vid / f"{gid}_replay.mp4"
                if not link.exists():
                    link.symlink_to(mp4.resolve())
                g["replay_video"], g["replay_offset"] = f"videos/{link.name}", blr.offset_of(link)
    (out / "moments.json").write_text(json.dumps(data, indent=1))  # offsets cached for the next build
    data["kinds"] = KIND
    kinds = "".join(f'<option value="{k}">{v}</option>' for k, v in KIND.items())
    html = (PAGE.replace("__DATA__", json.dumps(data)).replace("__DECK__", json.dumps(blr.DECK))
            .replace("__LEAD__", str(LEAD)).replace("__TAIL__", str(TAIL)).replace("__KINDS__", kinds))
    (out / "index.html").write_text(html)
    print(f"{len(data['moments'])} moments, {len(data['games'])} games -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
