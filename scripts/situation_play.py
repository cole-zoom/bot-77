"""Play the defence situations yourself (2026-10-28, Cole: "can u let me play live? to defend").

  uv run --no-sync --with gymnasium==1.3.0 python scripts/situation_play.py [--ckpt data/models/rl_clones_v2/final.pth]
  then open http://localhost:8777

A local page (the simulator runs on this Mac, so it can't be a published page): the same top-down board as the replays,
your hand along the bottom, the game in real time (you act on any 0.1 s tick; the opponent clone decides once a second
as trained; the bot it is compared with still decides once a second). Click a card, then a tile, to play
it; otherwise you wait. Each situation is drawn exactly as in training (bot77.rl.situations, the opponent clone
playing on), scored the same way, and then replayed from the identical start by the scripted defence and the model,
so the three scores sit side by side. Every game is appended to data/live/situation_play.jsonl (gitignored) as a
reference for later; Save bookmarks a start (situation type + seed + a note) in data/live/saved_situations.json, and
the Saved list reloads one exactly.
"""
import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data/live/situation_play.jsonl"
SAVED = ROOT / "data/live/saved_situations.json"  # bookmarked starts (Save button): situation type + seed + note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/models/rl_clones_v2/final.pth")
    ap.add_argument("--port", type=int, default=8777)
    a = ap.parse_args()
    os.environ.update({"CLASH_DECK": "2,81,5,43,9,95,7,3", "RL_CLONE_CKPT": str(ROOT / "data/models/bc_v2/frozen.pth"),
                       "RL_PRO_DECKS": str(ROOT / "data/cache/rl/pro_decks.json"), "RL_SITUATION_SHARE": "1"})
    ckpt = str((ROOT / a.ckpt).resolve())
    sys.path.insert(0, str(ROOT / "third_party" / "clash-royale-env"))
    os.chdir(ROOT / "third_party" / "clash-royale-env")
    import python_ai  # noqa: F401
    import torch
    from torch.distributions import Categorical
    from python_ai.models.net import MicroRoyaleNet
    from python_ai.models.policy_io import LSTM_HIDDEN

    from bot77.rl import situations as sit
    from bot77.rl.clone_env import CloneEnv
    from bot77.sim import cards as sc
    torch.set_num_threads(2)
    names = {i: c["name"] for i, c in sc.engine_cards().items()}
    net = MicroRoyaleNet()
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"], strict=False)
    net.eval()
    templates = {"hog_rider": sit.hog_rider, "battle_ram": sit.battle_ram, "melee": sit.melee,
                 "supported_push": sit.supported_push}
    game = {}

    def start(env, tpl, seed):
        env.forced_template = tpl
        env.rng = np.random.default_rng(seed)
        env.scenario_rng = env.rng
        torch.manual_seed(seed)
        obs, _ = env.reset()
        return obs

    def state(env, done=False):
        g = env.game
        units = [{"id": u.id, "name": u.name, "team": u.team, "x": round(u.x, 2), "y": round(u.y, 2), "hp": round(u.hp),
                  "flying": bool(u.flying), "building": bool(u.building), "tower": bool(u.tower)} for u in g.get_units()]
        hand = [{"id": c, "name": names.get(c, str(c)), "cost": sit.COST.get(c, 0),
                 "spell": c in (sit.FIREBALL, sit.ARROWS)} for c in g.get_hand()]
        return {"units": units, "hand": hand, "elixir": round(g.get_elixir(), 2),
                "opp_elixir": round(g.get_elixir_for_team(1), 2), "towers": [[max(0, g.get_tower_hp(t, s)) for s in range(3)] for t in (0, 1)],
                "t": game.get("t", 0), "done": done}

    def run_other(tpl, seed, who):
        env = CloneEnv({"scenario_seed": 0, "decision_ticks": 10})  # the bot (and the yardstick) at its trained 1 s
        obs = start(env, tpl, seed)
        h = (torch.zeros(1, LSTM_HIDDEN), torch.zeros(1, LSTM_HIDDEN))
        while True:
            if who == "scripted":
                k, x, y = sit.scripted_defence(env)
            else:
                o = torch.tensor(obs).unsqueeze(0)
                with torch.no_grad():
                    m = net.affordability_mask(o)
                    f, e, sp, hi = net.extract_features_hires(o)
                    lg, _, _, _, h = net.step_lstm_and_card(f, h, m)
                    kk = Categorical(logits=lg).sample()
                    xx, yy = net.cell_to_xy(Categorical(logits=net.placement_given_card(h[0], e, kk, o, sp, hires_map=hi)).sample())
                k, x, y = int(kk.item()), float(xx.item()), float(yy.item())
            obs, r, term, trunc, info = env.step({"card_index": k, "target_x": np.array([x], np.float32),
                                                  "target_y": np.array([y], np.float32),
                                                  "activate_ability_slot1": 0, "activate_ability_slot2": 0})
            if info.get("rl_situation_end") or term or trunc:
                return env.last_situation

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def send(self, code, body, ctype="application/json"):
            b = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self.send(200, PAGE, "text/html; charset=utf-8")
            if self.path == "/saved":
                return self.send(200, SAVED.read_text() if SAVED.exists() else "[]")
            self.send(404, "{}")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/new":
                name = body.get("template", "hog_rider")
                seed = int(body.get("seed") or int(time.time()) % 100_000)
                env = CloneEnv({"scenario_seed": 0, "decision_ticks": 1})  # you act every tick (0.1 s): real time
                obs = start(env, templates[name], seed)
                spec = env.situation.spec
                game.clear()
                game.update({"env": env, "tpl": name, "seed": seed, "t": 0, "plays": [], "spec_name": spec["name"],
                             "own_elixir": spec["own_elixir"], "opp_elixir": spec["opp_elixir"],
                             "threat": [names.get(c, str(c)) for c, _, _ in spec["units"]]})
                return self.send(200, json.dumps({"state": state(env), "situation": spec["name"], "seed": seed,
                                                  "threat": game["threat"], "start_elixir": [round(spec["own_elixir"], 1), round(spec["opp_elixir"], 1)]}))
            if self.path == "/save":
                saved = json.loads(SAVED.read_text()) if SAVED.exists() else []
                saved.append({"template": body["template"], "seed": int(body["seed"]), "note": body.get("note", ""),
                              "threat": body.get("threat", ""), "saved": time.strftime("%Y-%m-%d %H:%M")})
                SAVED.parent.mkdir(parents=True, exist_ok=True)
                SAVED.write_text(json.dumps(saved, indent=1))
                return self.send(200, json.dumps(saved))
            if self.path == "/step":
                env = game.get("env")
                if env is None:
                    return self.send(400, json.dumps({"error": "start a situation first"}))
                slot = body.get("slot")
                k, x, y = (4, 0.0, 0.0) if slot is None else (int(slot), float(body["x"]), float(body["y"]))
                hand = list(env.game.get_hand())
                el0 = env.game.get_elixir()
                obs, r, term, trunc, info = env.step({"card_index": k, "target_x": np.array([x], np.float32),
                                                      "target_y": np.array([y], np.float32),
                                                      "activate_ability_slot1": 0, "activate_ability_slot2": 0})
                game["t"] = round(game["t"] + 0.1, 1)
                if k < 4:
                    landed = env.game.get_elixir() < el0 - 0.5
                    game["plays"].append({"t": game["t"], "card": names.get(hand[k], str(hand[k])), "x": x, "y": y,
                                          "landed": bool(landed)})
                done = bool(info.get("rl_situation_end") or term or trunc)
                out = {"state": state(env, done), "plays": game["plays"]}
                if done:
                    res = env.last_situation
                    tpl = templates[game["tpl"]]
                    others = {"scripted": run_other(tpl, game["seed"], "scripted"), "model": run_other(tpl, game["seed"], "model")}
                    out["result"] = {"you": res, **others}
                    LOG.parent.mkdir(parents=True, exist_ok=True)
                    with LOG.open("a") as f:
                        f.write(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "situation": game["spec_name"],
                                            "seed": game["seed"], "plays": game["plays"], "result": out["result"]}) + "\n")
                    game["env"] = None
                return self.send(200, json.dumps(out, default=float))
            self.send(404, "{}")

    print(f"open http://localhost:{a.port}  (model: {Path(ckpt).parent.name}/{Path(ckpt).name})", flush=True)
    HTTPServer(("127.0.0.1", a.port), H).serve_forever()


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Defend It Yourself</title>
<style>
:root { --bg:#eef1ec; --panel:#fff; --ink:#1d2620; --muted:#5d6a61; --line:#d3dbd3; --grass:#a9c98a; --grass2:#9cbf7c;
  --river:#6fb6d9; --bridge:#b48a5a; --ours:#2f6fd6; --theirs:#d6453a; --accent:#c99a1b; --legal:#ffffff55; --elx:#b44ac7; }
@media (prefers-color-scheme: dark) { :root { --bg:#121815; --panel:#1b231e; --ink:#e6ece7; --muted:#9aa89e; --line:#2c3830;
  --grass:#4f6b3f; --grass2:#476238; --river:#2f6e8f; --bridge:#7d5f3e; --ours:#6aa0ff; --theirs:#ff7a6e; --accent:#e3b740; --legal:#ffffff22; color-scheme: dark } }
body { margin:0; background:var(--bg); color:var(--ink); font:15px system-ui, sans-serif; }
.wrap { max-width: 980px; margin: 0 auto; padding: 16px; display:grid; grid-template-columns: auto minmax(0,1fr); gap: 18px; }
@media (max-width: 760px) { .wrap { grid-template-columns: minmax(0,1fr); } }
h1 { margin:0 0 4px; font-size:1.5rem; } .muted { color: var(--muted); }
canvas { width: 342px; max-width: 100%; aspect-ratio: 18/34; border-radius: 8px; cursor: crosshair; display:block; }
.hand { display:grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-top: 8px; max-width: 342px; }
.card { border:2px solid var(--line); background:var(--panel); color:var(--ink); border-radius:8px; padding:8px 4px; font-weight:600; cursor:pointer; text-align:center; font-size:.85rem; }
.card[disabled] { opacity:.45; cursor:not-allowed; } .card.sel { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent); }
.card .c { display:block; color: var(--elx); font-family: ui-monospace, monospace; }
.elx { height:10px; background: var(--line); border-radius:5px; overflow:hidden; max-width:342px; margin-top:8px; } .elx > div { height:100%; background: var(--elx); }
.panel { background: var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px; display:grid; gap:10px; align-content:start; }
.row { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
button.act, select { font: inherit; padding:6px 10px; border-radius:6px; border:1px solid var(--line); background: var(--panel); color: var(--ink); }
button.go { background: var(--accent); border-color: var(--accent); color:#1d1606; font-weight:700; }
ol { margin:0; padding-left: 1.2em; font-family: ui-monospace, monospace; font-size:.85rem; }
table { border-collapse: collapse; font-size:.9rem; } td, th { padding: 4px 8px; text-align:right; border-bottom: 1px solid var(--line); } td:first-child, th:first-child { text-align:left; }
.big { font-size: 1.1rem; font-weight: 700; }
[hidden] { display: none !important; }
.boardwrap { position: relative; width: 342px; max-width: 100%; }
.count { position: absolute; inset: 0; display: grid; place-items: center; font: 800 6rem system-ui, sans-serif; color: #fff; text-shadow: 0 4px 18px #000a; pointer-events: none; }
.stepbar { height: 8px; background: var(--line); border-radius: 4px; overflow: hidden; max-width: 342px; margin-top: 8px; }
.stepbar > div { height: 100%; width: 0; background: var(--accent); }
.steplab { font: .8rem ui-monospace, monospace; color: var(--muted); margin-top: 2px; }
</style></head><body>
<div class="wrap">
  <div>
    <div class="boardwrap"><canvas id="board" width="540" height="1020"></canvas><div class="count" id="count" hidden></div></div>
    <div class="steplab" id="steplab">—</div>
    <div class="elx"><div id="elx"></div></div>
    <div class="muted" id="elxt" style="font-family:ui-monospace,monospace;font-size:.85rem;margin-top:4px">elixir —</div>
    <div class="hand" id="hand"></div>
  </div>
  <div class="panel">
    <div><h1>Defend it yourself</h1><div class="muted">Pick a situation and press New (Restart replays the same start). A 3-2-1 holds the game so you can get to the board, and you can already queue a card; then it runs in real time (a tick every 0.1 s) and a card lands on the next tick after you place it. Half and quarter speed slow everything down, and take effect immediately. Click a card (or press 1-4), then a tile on our half (spells anywhere); Esc cancels, Space pauses. Do nothing and you wait.</div></div>
    <div class="row">
      <select id="tpl"><option value="hog_rider">Hog Rider</option><option value="battle_ram">Battle Ram</option><option value="melee">Melee (Mini P.E.K.K.A / Knight / Prince)</option><option value="supported_push">Supported push (Giant / Golem)</option></select>
      <select id="speed"><option value="100">real time</option><option value="200">half speed</option><option value="400">quarter speed</option></select>
      <button class="act go" id="start">New</button><button class="act" id="restart">Restart</button><button class="act" id="pause">Pause</button>
    </div>
    <div class="row">
      <input id="note" placeholder="note for this start (optional)" style="flex:1 1 180px;min-width:0;font:inherit;padding:6px 8px;border-radius:6px;border:1px solid var(--line);background:var(--panel);color:var(--ink)">
      <button class="act" id="save">Save this start</button>
      <select id="savedlist"><option value="">Saved starts…</option></select>
    </div>
    <div id="setup" class="muted"></div>
    <div><b>Your plays</b><ol id="plays"></ol></div>
    <div id="result"></div>
  </div>
</div>
<script>
// One game loop per game (a generation number cancels the old one: New / Restart mid-game used to let a late
// response from the previous game overwrite the new one). The hand's buttons are built once and only updated
// (rebuilding them ten times a second ate clicks). The speed applies as soon as it changes. The 3-2-1 holds the game;
// a card queued during it lands on the first tick.
let S = null, sel = null, pending = null, paused = false, lastPlay = null, lastSeed = null, lastTpl = null, lastThreat = "";
let gen = 0, running = false;
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const cv = $("board"), ctx = cv.getContext("2d"), W = cv.width, H = cv.height, sx = W / 18, sy = H / 34;
const ABBR = {"Mini PEKKA":"MP","Mega Minion":"MM","Goblin Hut":"Hut","Battle Ram":"Ram","Hog Rider":"Hog","Spear Goblins":"gob","Barbarians":"barb","Musketeer":"Musk","Executioner":"Exec","Golemite":"gmt","Minions":"min"};
const maxHp = new Map();
const period = () => +$("speed").value;
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function post(path, body) { const r = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)}); return r.json(); }

function draw() {
  ctx.clearRect(0, 0, W, H);
  for (let r = 0; r < 34; r++) { ctx.fillStyle = r % 2 ? css("--grass") : css("--grass2"); ctx.fillRect(0, H - (r + 1) * sy, W, sy); }
  ctx.fillStyle = css("--river"); ctx.fillRect(0, H - 17.5 * sy, W, 2 * sy);
  ctx.fillStyle = css("--bridge"); for (const bx of [2, 14]) ctx.fillRect(bx * sx - sx / 2, H - 17.5 * sy, 2 * sx, 2 * sy);
  if (sel !== null && S && S.hand[sel] && !S.hand[sel].spell) { ctx.fillStyle = css("--legal"); ctx.fillRect(0, H - 15 * sy, W, 15 * sy); }
  if (!S) return;
  for (const u of S.units) {
    if (!maxHp.has(u.id) || u.hp > maxHp.get(u.id)) maxHp.set(u.id, u.hp);
    const px = (u.x + 0.5) * sx, py = H - (u.y + 0.5) * sy, r = u.tower ? 1.5 * sx : u.building ? 0.9 * sx : (u.flying ? 0.45 : 0.55) * sx;
    ctx.fillStyle = u.team === 0 ? css("--ours") : css("--theirs");
    if (u.tower || u.building) ctx.fillRect(px - r, py - r, 2 * r, 2 * r);
    else { ctx.beginPath(); ctx.arc(px, py, r, 0, 7); ctx.fill(); if (u.flying) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke(); } }
    const f = Math.max(0, Math.min(1, u.hp / (maxHp.get(u.id) || 1)));
    ctx.fillStyle = "#000a"; ctx.fillRect(px - r, py - r - 10, 2 * r, 6);
    ctx.fillStyle = f > .5 ? "#3fbf5a" : f > .25 ? "#e3b740" : "#e5533f"; ctx.fillRect(px - r, py - r - 10, 2 * r * f, 6);
    const lab = u.tower ? (u.hp > 0 ? Math.round(u.hp) : "") : (ABBR[u.name] ?? (u.name || "").slice(0, 5));
    if (lab !== "") { ctx.font = `600 ${u.tower ? 20 : 16}px ui-monospace,monospace`; ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(lab, px, py); }
  }
  if (pending) { ctx.strokeStyle = css("--accent"); ctx.lineWidth = 5; ctx.beginPath(); ctx.arc((pending.x + 0.5) * sx, H - (pending.y + 0.5) * sy, 1.2 * sx, 0, 7); ctx.stroke(); }
  if (lastPlay) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.setLineDash([6, 5]); ctx.beginPath(); ctx.arc((lastPlay.x + 0.5) * sx, H - (lastPlay.y + 0.5) * sy, 1.2 * sx, 0, 7); ctx.stroke(); ctx.setLineDash([]); }
}

// the four card buttons, made once
const cardBtns = [0, 1, 2, 3].map(i => {
  const b = document.createElement("button"); b.className = "card"; b.type = "button";
  b.addEventListener("pointerdown", ev => { ev.preventDefault(); pick(i); });   // on press: a click straddling an update can't be lost
  $("hand").appendChild(b); return b;
});
function pick(i) {
  if (!S || S.done || !S.hand[i] || S.hand[i].cost > S.elixir + 1e-6) return;
  sel = sel === i ? null : i; updateHand(); draw();
}
function updateHand() {
  cardBtns.forEach((b, i) => {
    const c = S && S.hand[i];
    const label = c ? `<span style="color:var(--muted);font-size:.75rem">${i + 1}</span> ${c.name}<span class="c">${c.cost}</span>` : "";
    if (b.dataset.label !== label) { b.innerHTML = label; b.dataset.label = label; }
    b.disabled = !c || S.done || c.cost > S.elixir + 1e-6;
    b.classList.toggle("sel", sel === i);
  });
  if (S) {
    $("elx").style.width = (S.elixir * 10) + "%";
    $("elxt").textContent = `elixir ${S.elixir.toFixed(1)}  ·  theirs ${S.opp_elixir.toFixed(1)}  ·  ${(+S.t).toFixed(1)} s`;
  }
}

cv.addEventListener("pointerdown", ev => {
  if (!S || sel === null || S.done) return;
  const rect = cv.getBoundingClientRect();
  const x = Math.min(17, Math.max(0, Math.floor((ev.clientX - rect.left) / rect.width * 18)));
  const y = Math.min(33, Math.max(0, Math.floor((rect.bottom - ev.clientY) / rect.height * 34)));
  if (!S.hand[sel].spell && y > 14) { flash("troops and buildings go on our half"); return; }
  pending = {slot: sel, x, y}; sel = null; updateHand(); draw();
});
function flash(msg) { $("steplab").textContent = msg; }

async function loop(g) {
  running = true;
  while (g === gen && S && !S.done) {
    const t0 = performance.now();
    if (!paused) {
      const act = pending; pending = null;
      const out = await post("/step", act || {});
      if (g !== gen) return;                     // a New / Restart happened while this step was in flight
      if (act) lastPlay = act;
      S = out.state; updateHand(); draw();
      $("plays").innerHTML = (out.plays || []).map(p => `<li>${(+p.t).toFixed(1)}s ${p.card} → (${p.x}, ${p.y})${p.landed ? "" : " — refused (illegal spot?)"}</li>`).join("");
      $("steplab").textContent = S.done ? "finished" : `${(+S.t).toFixed(1)} s`;
      if (out.result) { showResult(out.result); break; }
    } else { $("steplab").textContent = "paused"; }
    await sleep(Math.max(0, period() - (performance.now() - t0)));
  }
  if (g === gen) running = false;
}

async function begin(seed) {
  const g = ++gen;                                 // cancels any running loop
  sel = null; pending = null; lastPlay = null; paused = false; maxHp.clear();
  $("pause").textContent = "Pause"; $("result").innerHTML = ""; $("plays").innerHTML = "";
  const out = await post("/new", {template: $("tpl").value, seed});
  if (g !== gen) return;
  lastSeed = out.seed; lastTpl = $("tpl").value; lastThreat = out.threat.join(" + ");
  S = out.state; updateHand(); draw();
  $("setup").innerHTML = `Threat <b>${lastThreat}</b> · your elixir <b>${out.start_elixir[0]}</b> · theirs <b>${out.start_elixir[1]}</b> · seed ${out.seed}`;
  const cnt = $("count");
  for (const n of [3, 2, 1]) {                     // the game waits; you can already queue a card
    if (g !== gen) return;
    cnt.textContent = n; cnt.hidden = false; $("steplab").textContent = `starting in ${n}`;
    await sleep(1000);
  }
  if (g !== gen) return;
  cnt.hidden = true;
  loop(g);
}

function fmt(r) { return `<td class="big">${(r.score >= 0 ? "+" : "") + r.score.toFixed(2)}</td><td>${Math.round(r.tower_hp_lost)}</td><td>${r.elixir_spent}</td><td>${r.unresolved ? "alive" : "handled"}</td><td>${r.seconds}s</td>`; }
function showResult(R) {
  $("result").innerHTML = `<b>Result</b> (higher is better)<table><tr><th></th><th>score</th><th>tower HP lost</th><th>elixir</th><th>threat</th><th>time</th></tr>
  <tr><td>You</td>${fmt(R.you)}</tr><tr><td>Scripted defence</td>${fmt(R.scripted)}</tr><tr><td>Bot (round 2)</td>${fmt(R.model)}</tr></table>
  <div class="muted" style="margin-top:6px">Same start for all three; the opponent clone plays on, so its plays can differ.</div>`;
}

$("start").onclick = () => begin(null);
$("restart").onclick = () => { if (lastSeed !== null) { $("tpl").value = lastTpl; begin(lastSeed); } };
$("pause").onclick = () => { paused = !paused; $("pause").textContent = paused ? "Resume" : "Pause"; };
document.addEventListener("keydown", ev => {   // 1-4 pick a card, Esc cancels, Space pauses
  if (ev.target.tagName === "SELECT" || ev.target.tagName === "INPUT") return;
  if (["1", "2", "3", "4"].includes(ev.key)) { pick(+ev.key - 1); ev.preventDefault(); }
  else if (ev.key === "Escape") { sel = null; pending = null; updateHand(); draw(); }
  else if (ev.key === " ") { $("pause").click(); ev.preventDefault(); }
});

async function loadSaved() {
  const list = await (await fetch("/saved")).json(), s = $("savedlist");
  s.innerHTML = `<option value="">Saved starts (${list.length})…</option>` + list.map((x, i) =>
    `<option value="${i}">${x.template.replace("_", " ")} · ${x.threat} · seed ${x.seed}${x.note ? " · " + x.note : ""}</option>`).join("");
  s.dataset.list = JSON.stringify(list);
}
$("save").onclick = async () => {
  if (lastSeed === null) return;
  await post("/save", {template: lastTpl, seed: lastSeed, note: $("note").value, threat: lastThreat});
  $("note").value = ""; $("save").textContent = "Saved"; setTimeout(() => $("save").textContent = "Save this start", 1200);
  loadSaved();
};
$("savedlist").onchange = ev => {
  const list = JSON.parse(ev.target.dataset.list || "[]"), x = list[+ev.target.value];
  if (!x) return;
  $("tpl").value = x.template; ev.target.value = ""; begin(x.seed);
};
loadSaved(); updateHand(); draw();
</script></body></html>"""


if __name__ == "__main__":
    main()
