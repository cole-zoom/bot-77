"""A video review page for a live run's games (2026-10-31, Cole: "I'd love to review the 14 losses, but u should rebuild
teh entire game or somth in the sims replay mode or link to video. I cant review still frames").

  uv run --no-sync python scripts/build_loss_review.py --runs run40v4 run40v4b --date 2026-10-31 [--all] [--port 8790]

Per game (losses by default): the bot's own screen recording (data/raw/live_games/<gid>/game.mkv, remuxed to .mp4, no
re-encode) and the Battle Log replay of the same game (both sides; matched by the opponent's name read off its entry),
the bot's plays from the run log (card, where, game time; a click seeks both videos to 2 s before it), and per play
OK / Mistake, the card it should have played or "wait", a note. "Mark this moment" adds a note at the video's current
time for what wasn't a play. Saved in the browser as you go; Export writes the JSON for config/playbook_rules.yaml and
the next play-page situations. The page and the video links go to data/reviews/<name>/ (gitignored); the remuxed
videos sit beside the originals on the LaCie, linked in. Served locally (videos are GBs and private).

Game time: the run log's countdown, in order (regulation 180 - countdown; once it jumps back up, overtime 180 + 120 -
countdown). Each video's offset from game time is read off the game's own "Time left" timer in a few of its frames
(offset_of): the bot's recording starts ~4 s into the game (measured: 4 s at 20 / 31 / 60 s in bot_2026-10-31_012724),
and a replay has its intro first. Neither the bot's decision clock nor the recording's length gives it.
"""
import argparse
import difflib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data/live"
GAMES = ROOT / "data/raw/live_games"
REPLAYS = ROOT / "data/raw/replays"
TIMER_AT = (20, 30, 40, 50)  # s into a video where its "Time left" timer is read (offset_of)
DECK = ["Giant", "Battle Ram", "Mini P.E.K.K.A", "Mega Minion", "Bomber", "Goblin Hut", "Fireball", "Arrows"]


def parse_runs(tags: list[str]) -> list[dict]:
    games = []
    for tag in tags:
        for log in sorted(LIVE.glob(f"{tag}_chunk*.log")):
            cur, gid = None, None
            for line in log.read_text(errors="replace").splitlines():
                m = re.search(r"game (bot_\d{4}-\d\d-\d\d_\d{6}) starting", line)
                if m:
                    gid = m.group(1)
                    continue
                m = re.match(r"▶ Game (\d+)\s+vs (.*)$", line.strip())
                if m:
                    cur = {"run": tag, "chunk": log.stem.split("_")[-1], "n": int(m.group(1)), "opp": m.group(2).strip(),
                           "gid": gid, "plays": [], "hands": [], "_ot": False, "_last": 999}
                    continue
                if cur is None:
                    continue
                m = re.match(r"\s*(·\s*)?(\d):(\d\d)\s+(.*)$", line)
                if m:
                    cd = int(m.group(2)) * 60 + int(m.group(3))
                    if cd > cur["_last"] + 30:  # the countdown jumped back up: overtime
                        cur["_ot"] = True
                    cur["_last"] = cd
                    t = (180 + 120 - cd) if cur["_ot"] else (180 - cd)
                    rest = m.group(4)
                    if m.group(1):  # a "hand:" line
                        h = re.match(r"hand:\s*(.*?)\s+·\s+(.*?)\s+[▮▯]", rest)
                        if h:
                            cur["hands"].append({"t": t, "hand": h.group(1), "threats": h.group(2)})
                    else:
                        p = re.match(r"(.+?)\s+→\s+(.+?)\s{2,}.*?([\d.]+)\s*$", rest)
                        if p:
                            cur["plays"].append({"t": t, "card": p.group(1).strip(), "where": p.group(2).strip(),
                                                 "elixir": float(p.group(3)), "clock": f"{m.group(2)}:{m.group(3)}"
                                                 + (" OT" if cur["_ot"] else "")})
                    continue
                m = re.match(r"■ Game (\d+): (\w+)\s+·\s+(\d+) plays\s+·\s+([\d.]+) min", line.strip())
                if m:
                    cur.update(result=m.group(2), minutes=float(m.group(4)))
                    for k in ("_ot", "_last"):
                        cur.pop(k)
                    games.append(cur)
                    cur = None
    return games


def replay_texts(date: str) -> dict[Path, list[str]]:
    """Battle Log replay mp4 -> every text on its entry's header (Apple Vision OCR): names, clans, trophies."""
    from ocrmac import ocrmac
    from PIL import Image
    out = {}
    for png in sorted((REPLAYS / date).glob("replay_*_entry.png")):
        im = Image.open(png)
        w, h = im.size
        crop = im.crop((0, 0, w, int(h * 0.36)))
        mp4 = png.with_name(png.name.replace("_entry.png", ".mp4"))
        if mp4.exists():
            out[mp4] = [t for t, _c, _b in ocrmac.OCR(crop, recognition_level="accurate").recognize()]
    return out


def match_replays(games: list[dict], texts: dict[Path, list[str]]) -> None:
    """Each game's replay: the entry with a text most like its opponent's name (the run log's, from the bot's own
    OCR), best pairs first, each entry used once."""
    pairs = []
    for i, g in enumerate(games):
        for mp4, ts in texts.items():
            score = max((difflib.SequenceMatcher(None, norm(g["opp"]), norm(t)).ratio() for t in ts), default=0.0)
            pairs.append((score, i, mp4))
    used_g, used_r = set(), set()
    for score, i, mp4 in sorted(pairs, key=lambda x: -x[0]):
        if score < 0.6 or i in used_g or mp4 in used_r:
            continue
        games[i]["_replay"], games[i]["_replay_score"] = mp4, round(score, 2)
        used_g.add(i)
        used_r.add(mp4)


def offset_of(video: Path) -> float | None:
    """Game time minus video time, from the "Time left m:ss" timer (top right: the top 10% in the bot's recording, 13-18%
    in a Battle Log replay, under the opponent's name) in frames at TIMER_AT seconds (the
    median; whole seconds, so +-0.5 s). None if the timer never reads."""
    import io
    import statistics
    from ocrmac import ocrmac
    from PIL import Image
    got = []
    for vt in TIMER_AT:
        png = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(vt), "-i", str(video), "-frames:v", "1",
                              "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
        if not png:
            continue
        im = Image.open(io.BytesIO(png))
        w, h = im.size
        for t, _c, _b in ocrmac.OCR(im.crop((int(w * 0.70), 0, w, int(h * 0.20))), recognition_level="accurate").recognize():
            m = re.fullmatch(r"([0-3]):([0-5]\d)", t.strip())
            if m:
                got.append(180 - (int(m.group(1)) * 60 + int(m.group(2))) - vt)
                break
    return float(statistics.median(got)) if got else None


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def remux(src: Path) -> Path:
    """game.mkv -> game.mp4 beside it (on the LaCie when the .mkv is a link there), stream copy."""
    real = src.resolve()
    dst = real.with_suffix(".mp4")
    if not dst.exists():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(real), "-c", "copy", "-movflags", "+faststart",
                        str(dst)], check=True)
    return dst


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Loss Review</title><style>
:root{--bg:#f4f5f7;--panel:#fff;--ink:#1b1f24;--muted:#5d6673;--line:#e2e5ea;--good:#1f8a4c;--bad:#c43d3d;--accent:#6b3fd4;--pick:#fff4c2}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--panel:#1a1e24;--ink:#e8eaed;--muted:#9aa3ad;--line:#2c323a;--good:#52c585;--bad:#f07575;--accent:#a98bff;--pick:#3a3320}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0}.muted{color:var(--muted)}.small{font-size:12px}
button,select,input,textarea{font:inherit;color:inherit;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
button{cursor:pointer}button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
#games{display:flex;gap:6px;flex-wrap:wrap}#games button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
#games button.done{border-color:var(--good)}
main{display:grid;grid-template-columns:minmax(300px,520px) 1fr;gap:16px;padding:16px;align-items:start}
@media (max-width:900px){main{grid-template-columns:1fr}}
.vids{position:sticky;top:64px;display:flex;flex-direction:column;gap:8px}
.tabs{display:flex;gap:6px}.tabs button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
video{width:100%;max-height:calc(100vh - 170px);background:#000;border-radius:8px}
.side{display:flex;flex-direction:column;gap:10px}
.box{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.play{display:grid;grid-template-columns:62px 1fr;gap:4px 10px;padding:8px;border-radius:8px;border:2px solid transparent}
.play.ok{border-color:var(--good)}.play.bad{border-color:var(--bad)}.play.now{background:var(--pick)}
.play .t{font:600 13px ui-monospace,Menlo,monospace;cursor:pointer;color:var(--accent)}
.play .what{font-weight:600}.row{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.ok .k{background:var(--good);color:#fff;border-color:var(--good)}.bad .m{background:var(--bad);color:#fff;border-color:var(--bad)}
textarea{width:100%;min-height:3.2em;resize:vertical}
.moment{display:grid;grid-template-columns:62px 1fr auto;gap:8px;align-items:start;padding:6px 0;border-top:1px solid var(--line)}
.hand{font-size:12px;color:var(--muted);padding-left:72px}
</style></head><body>
<header><h1>Loss review</h1><div id="games"></div><span class="muted small" id="progress"></span>
<button class="primary" id="export">Export</button></header>
<main><div class="vids"><div class="tabs"><button id="tab-bot" class="on">Bot's screen</button><button id="tab-rep">Battle Log replay (both sides)</button>
<span class="muted small" id="vinfo"></span></div>
<video id="v" controls playsinline preload="metadata"></video>
<div class="row"><button id="back">⟲ 5 s</button><button id="slow">0.5×</button><button id="norm">1×</button><button id="fast">2×</button>
<button class="primary" id="mark">Mark this moment</button></div>
<div class="muted small">Space play / pause · ← → 2 s · click a play's time to jump to 2 s before it (either video).</div></div>
<div class="side"><div class="box" id="head"></div><div class="box"><b>Moments</b> <span class="muted small">(anything that wasn't one of the bot's plays: a card it should have played, a missed defence…)</span><div id="moments"></div></div>
<div class="box"><b>The bot's plays</b><div id="plays"></div></div>
<div class="box"><b>Game note</b><textarea id="gnote" placeholder="What lost this game?"></textarea></div></div></main>
<script>
const G=__DATA__,DECK=__DECK__,KEY="bot77-lossreview-"+G.map(g=>g.gid).join("|").length+"-"+G[0].gid;
let S={};try{S=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const st=g=>(S[g.gid]=S[g.gid]||{plays:{},moments:[],note:""});
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}};
let gi=0,tab="bot";const v=document.getElementById("v");
const fmt=t=>{t=Math.max(0,t);return Math.floor(t/60)+":"+String(Math.floor(t%60)).padStart(2,"0")};
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function gameButtons(){document.getElementById("games").innerHTML=G.map((g,i)=>{const s=st(g),n=Object.keys(s.plays).length;
 return `<button class="${i===gi?"on":""} ${n||s.moments.length||s.note?"done":""}" data-i="${i}">${i+1}. ${esc(g.opp)}</button>`}).join("");
 document.querySelectorAll("#games button").forEach(b=>b.onclick=()=>{gi=+b.dataset.i;show()});
 const done=G.filter(g=>{const s=st(g);return Object.keys(s.plays).length||s.moments.length||s.note}).length;
 document.getElementById("progress").textContent=`${done} / ${G.length} games touched`}
function setVideo(){const g=G[gi],src=tab==="bot"?g.bot_video:g.replay_video;
 document.getElementById("tab-bot").classList.toggle("on",tab==="bot");document.getElementById("tab-rep").classList.toggle("on",tab==="rep");
 document.getElementById("tab-rep").disabled=!g.replay_video;
 document.getElementById("vinfo").textContent=src?"":"(no video)";if(src&&!v.src.endsWith(src)){const t=v.currentTime;v.src=src;v.currentTime=t}}
// game time -> this video's time: the offset read off the game's timer (build_loss_review.offset_of); 4 s if unread
const off=()=>{const g=G[gi],o=tab==="bot"?g.bot_offset:g.replay_offset;return o==null?4:o};
function seek(t){v.currentTime=Math.max(0,t-off()-2);v.play()}
function show(){const g=G[gi],s=st(g);gameButtons();setVideo();
 document.getElementById("head").innerHTML=`<b>${esc(g.opp)}</b> · ${g.result} · ${g.minutes} min · ${g.plays.length} plays<div class="muted small">${g.run} ${g.chunk} game ${g.n} · ${g.gid}${g.replay_video?"":" · no Battle Log replay matched"}</div>`;
 document.getElementById("gnote").value=s.note||"";
 const hands=g.hands;let hi=0;
 document.getElementById("plays").innerHTML=g.plays.map((p,i)=>{const r=s.plays[i]||{};let h="";while(hi<hands.length&&hands[hi].t<=p.t){h=hands[hi];hi++}
  return (h?`<div class="hand">hand ${esc(h.hand)} · ${esc(h.threats)}</div>`:"")+`<div class="play ${r.v==="ok"?"ok":r.v==="bad"?"bad":""}" data-i="${i}">
  <div class="t" data-t="${p.t}">${p.clock}</div><div><span class="what">${esc(p.card)}</span> → ${esc(p.where)} <span class="muted small">at ${p.elixir} elixir</span>
  <div class="row" style="margin-top:4px"><button class="k">OK</button><button class="m">Mistake</button>
  ${r.v==="bad"?`<select class="sb"><option value="">should have…</option>${["wait",...DECK].map(c=>`<option ${r.should===c?"selected":""}>${c}</option>`).join("")}</select>`:""}</div>
  ${r.v==="bad"?`<textarea class="n" placeholder="Why, and where it should have gone">${esc(r.note||"")}</textarea>`:""}</div></div>`}).join("");
 document.querySelectorAll(".play").forEach(el=>{const i=el.dataset.i,r=()=>(s.plays[i]=s.plays[i]||{});
  el.querySelector(".t").onclick=()=>seek(+el.querySelector(".t").dataset.t);
  el.querySelector(".k").onclick=()=>{s.plays[i]={v:"ok"};save();show()};el.querySelector(".m").onclick=()=>{r().v="bad";save();show()};
  const sb=el.querySelector(".sb");if(sb)sb.onchange=()=>{r().should=sb.value;save()};const n=el.querySelector(".n");if(n)n.oninput=()=>{r().note=n.value;save()}});
 document.getElementById("moments").innerHTML=s.moments.length?s.moments.map((m,j)=>`<div class="moment"><div class="t" style="cursor:pointer;color:var(--accent);font:600 13px ui-monospace,monospace" data-t="${m.t}">${fmt(m.t)}</div>
  <div><div class="row"><select class="ms"><option value="">should have…</option>${["wait",...DECK].map(c=>`<option ${m.should===c?"selected":""}>${c}</option>`).join("")}</select>
  <span class="muted small">game time</span></div><textarea class="mn" placeholder="What should have happened here">${esc(m.note||"")}</textarea></div>
  <button class="del" title="Remove">✕</button></div>`).join(""):'<div class="muted small">None yet: pause the video where something went wrong and press "Mark this moment".</div>';
 document.querySelectorAll(".moment").forEach((el,j)=>{const m=s.moments[j];el.querySelector(".t").onclick=()=>{v.currentTime=Math.max(0,m.t-off()-2);v.play()};
  el.querySelector(".ms").onchange=e=>{m.should=e.target.value;save()};el.querySelector(".mn").oninput=e=>{m.note=e.target.value;save()};
  el.querySelector(".del").onclick=()=>{s.moments.splice(j,1);save();show()}})}
document.getElementById("gnote").oninput=e=>{st(G[gi]).note=e.target.value;save();gameButtons()};
document.getElementById("tab-bot").onclick=()=>{tab="bot";setVideo()};document.getElementById("tab-rep").onclick=()=>{tab="rep";setVideo()};
document.getElementById("back").onclick=()=>v.currentTime=Math.max(0,v.currentTime-5);
document.getElementById("slow").onclick=()=>v.playbackRate=0.5;document.getElementById("norm").onclick=()=>v.playbackRate=1;document.getElementById("fast").onclick=()=>v.playbackRate=2;
document.getElementById("mark").onclick=()=>{v.pause();const s=st(G[gi]),t=+(v.currentTime+off()).toFixed(1);s.moments.push({t,should:"",note:""});
 s.moments.sort((a,b)=>a.t-b.t);save();show();const ts=document.querySelectorAll(".mn");const k=s.moments.findIndex(m=>m.t===t);if(ts[k])ts[k].focus()};
document.addEventListener("keydown",e=>{if(["TEXTAREA","INPUT","SELECT"].includes(e.target.tagName))return;
 if(e.key===" "){e.preventDefault();v.paused?v.play():v.pause()}else if(e.key==="ArrowLeft")v.currentTime-=2;else if(e.key==="ArrowRight")v.currentTime+=2});
document.getElementById("export").onclick=()=>{const out=G.map(g=>({gid:g.gid,opp:g.opp,run:g.run,n:g.n,result:g.result,plays:g.plays.map((p,i)=>({...p,...(st(g).plays[i]||{})})),
 moments:st(g).moments,note:st(g).note}));const b=new Blob([JSON.stringify({by:"cole",exported:new Date().toISOString(),games:out},null,1)],{type:"application/json"});
 const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="loss_review_"+new Date().toISOString().slice(0,10)+".json";a.click()};
show();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--date", required=True, help="the replays' folder under data/raw/replays")
    ap.add_argument("--all", action="store_true", help="every game, not only the losses")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    games = [g for g in parse_runs(a.runs) if a.all or g["result"] == "LOSS"]
    out = ROOT / "data/reviews" / (a.name or f"losses_{a.date}")
    vid = out / "videos"
    vid.mkdir(parents=True, exist_ok=True)
    match_replays(games, replay_texts(a.date))
    for g in games:
        mkv = GAMES / g["gid"] / "game.mkv"
        g["bot_video"] = None
        if mkv.exists():
            link = vid / f"{g['gid']}.mp4"
            if not link.exists():
                link.symlink_to(remux(mkv))
            g["bot_video"] = f"videos/{link.name}"
            g["bot_offset"] = offset_of(link)
        g["replay_video"] = None
        mp4 = g.pop("_replay", None)
        if mp4 is not None:
            link = vid / f"{g['gid']}_replay.mp4"
            if link.is_symlink():
                link.unlink()
            link.symlink_to(mp4.resolve())
            g["replay_video"] = f"videos/{link.name}"
            g["replay_file"] = mp4.name
            g["replay_offset"] = offset_of(mp4)
        print(f"{g['run']} {g['chunk']} game {g['n']:2d} vs {g['opp']:20s} {g['result']:4s} {len(g['plays']):2d} plays  "
              f"bot video {'yes' if g['bot_video'] else 'NO'} (offset {g.get('bot_offset')})  replay {g.get('replay_file', 'NO')} "
              f"{g.pop('_replay_score', '')} (offset {g.get('replay_offset')})")
    html = PAGE.replace("__DATA__", json.dumps(games)).replace("__DECK__", json.dumps(DECK))
    (out / "index.html").write_text(html)
    print(f"{len(games)} games -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
