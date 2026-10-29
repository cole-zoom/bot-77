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
    ap.add_argument("--ckpt", default="data/models/rl_clones_v3/final.pth")
    ap.add_argument("--model-ticks", type=int, default=1, help="the model's decision interval in 0.1 s ticks (v3: 1)")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--log", default=str(LOG), help="where finished games are appended (a test server writes elsewhere)")
    a = ap.parse_args()
    log = Path(a.log)
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
    from bot77.sim import placement as pl
    torch.set_num_threads(2)
    names = {i: c["name"] for i, c in sc.engine_cards().items()}
    net = MicroRoyaleNet()
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"], strict=False)
    net.eval()
    # every template, Cole's 1 / 8 / 9 and the tempo ones included, plus the held-out pressure pair (P3, 2026-10-29)
    templates = {t.__name__: t for t in sit.TEMPLATES + sit.REVIEW_TEMPLATES + sit.HELD_OUT_TEMPLATES}
    game = {}

    def start(env, tpl, seed):
        env.forced_template = tpl
        env.rng = np.random.default_rng(seed)
        env.scenario_rng = env.rng
        torch.manual_seed(seed)
        obs, _ = env.reset()
        return obs

    specs = {}  # card id -> its ghost on the page (bot77.sim.placement.card_spec), read from the engine once

    def ghost(c):
        if c not in specs:
            specs[c] = pl.card_spec(c)
        return specs[c]

    def state(env, done=False):
        g = env.game
        units = [{"id": u.id, "name": u.name, "team": u.team, "x": round(u.x, 2), "y": round(u.y, 2), "hp": round(u.hp),
                  "flying": bool(u.flying), "building": bool(u.building), "tower": bool(u.tower),
                  "card_id": int(u.card_id), "max_hp": pl.max_hp(g, u)} for u in g.get_units()]
        # per card: the engine's own legality at every cell centre, this tick (buildings on the board change it)
        hand = [{"id": c, "name": names.get(c, str(c)), "cost": sit.COST.get(c, 0), **{k: v for k, v in ghost(c).items()
                 if k in ("spell", "building", "placement_radius", "spell_radius", "offsets")},
                 "legal": pl.legal_mask(g, c)} for c in g.get_hand()]
        return {"units": units, "hand": hand, "elixir": round(g.get_elixir(), 2),
                "opp_elixir": round(g.get_elixir_for_team(1), 2), "towers": [[max(0, g.get_tower_hp(t, s)) for s in range(3)] for t in (0, 1)],
                "t": game.get("t", 0), "done": done, "clock": round(g.get_current_tick() / 10.0, 1)}

    def run_other(tpl, seed, who):
        # the yardstick once a second as in the exam's replays; the bot at its trained interval
        env = CloneEnv({"scenario_seed": 0, "decision_ticks": 10 if who == "scripted" else a.model_ticks})
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
            if self.path == "/board":  # the terrain before any game (a throwaway engine with our deck)
                import clash_royale_env as cre
                deck = [int(c) for c in os.environ["CLASH_DECK"].split(",")]
                g = cre.ClashRoyaleEnv(deck, deck, 3600)
                g.reset()
                return self.send(200, json.dumps({**pl.board_geometry(g), **pl.CLOCK}))
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
                             "threat": [names.get(u[0], str(u[0])) for u in spec["units"]]})
                return self.send(200, json.dumps({"state": state(env), "situation": spec["name"], "seed": seed,
                                                  "threat": game["threat"], "plan": spec.get("opp"),
                                                  "board": {**pl.board_geometry(env.game), **pl.CLOCK}, "start_elixir": [round(spec["own_elixir"], 1), round(spec["opp_elixir"], 1)]}))
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
                legal = k < 4 and k < len(hand) and env.game.is_valid_placement(hand[k], x, y, 0)
                obs, r, term, trunc, info = env.step({"card_index": k, "target_x": np.array([x], np.float32),
                                                      "target_y": np.array([y], np.float32),
                                                      "activate_ability_slot1": 0, "activate_ability_slot2": 0})
                game["t"] = round(game["t"] + 0.1, 1)
                if k < 4:
                    landed = env.game.get_elixir() < el0 - 0.5
                    play = {"t": game["t"], "card": names.get(hand[k], str(hand[k])), "x": x, "y": y,
                            "landed": bool(landed)}
                    if not landed:  # additive (2026-10-29): why the engine or the guard refused it
                        play["why"] = ("spell guard: nothing in reach" if info.get("spell_veto") else
                                       "not a legal tile" if not legal else "not enough elixir")
                    game["plays"].append(play)
                done = bool(info.get("rl_situation_end") or term or trunc)
                out = {"state": state(env, done), "plays": game["plays"]}
                if done:
                    res = env.last_situation
                    tpl = templates[game["tpl"]]
                    others = {"scripted": run_other(tpl, game["seed"], "scripted"), "model": run_other(tpl, game["seed"], "model")}
                    out["result"] = {"you": res, **others}
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open("a") as f:
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
:root { --bg:#eef1ec; --panel:#fff; --ink:#1d2620; --muted:#5d6a61; --line:#d3dbd3; --grass:#a9c98a; --grass2:#9dbf7d;
  --river:#6fb6d9; --bridge:#b48a5a; --dead:#6d7a70; --grid:rgba(20,40,20,.42); --gridlab:rgba(20,40,20,.62);
  --ours:#2f6fd6; --theirs:#d6453a; --accent:#c99a1b; --legal:rgba(255,255,255,.34); --illegal:rgba(0,0,0,.26);
  --hover:#ffffff; --elx:#b44ac7; --oelx:#d6453a; }
@media (prefers-color-scheme: dark) { :root { --bg:#121815; --panel:#1b231e; --ink:#e6ece7; --muted:#9aa89e; --line:#2c3830;
  --grass:#4f6b3f; --grass2:#46613a; --river:#2f6e8f; --bridge:#7d5f3e; --dead:#26302a; --grid:rgba(0,0,0,.55); --gridlab:rgba(255,255,255,.55);
  --ours:#6aa0ff; --theirs:#ff7a6e; --accent:#e3b740; --legal:rgba(255,255,255,.2); --illegal:rgba(0,0,0,.38);
  --hover:#ffffff; --oelx:#ff7a6e; color-scheme: dark } }
body { margin:0; background:var(--bg); color:var(--ink); font:15px system-ui, sans-serif; }
.wrap { max-width: 1240px; margin: 0 auto; padding: 16px; display:grid; grid-template-columns: auto minmax(0,1fr); gap: 18px; align-items:start; }
@media (max-width: 760px) { .wrap { grid-template-columns: minmax(0,1fr); } }
h1 { margin:0 0 4px; font-size:1.5rem; } .muted { color: var(--muted); }
.leftcol { width: fit-content; max-width: 100%; }
/* the board fills the laptop's height: ~22-26 px a tile on a 13-15" screen */
canvas { height: clamp(520px, calc(100vh - 215px), 1100px); width: auto; aspect-ratio: 18/34; border-radius: 6px; cursor: crosshair; display:block; touch-action: none; }
@media (max-width: 760px) { canvas { height: auto; width: 100%; } .leftcol { width: 100%; } }
.topbar { contain: inline-size; overflow: hidden; display:flex; align-items:baseline; gap:10px; margin-bottom:6px; font-family: ui-monospace, monospace; }
.clock { font: 800 1.45rem ui-monospace, monospace; letter-spacing: .02em; }
.phase { font-size:.8rem; font-weight:700; padding:1px 6px; border-radius:4px; background: var(--elx); color:#fff; }
.tile { margin-left:auto; font-size:.9rem; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0; }
.hand { display:grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-top: 8px; }
.card { border:2px solid var(--line); background:var(--panel); color:var(--ink); border-radius:8px; padding:8px 4px; font-weight:600; cursor:pointer; text-align:center; font-size:.85rem; }
.card[disabled] { opacity:.45; cursor:not-allowed; } .card.sel { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent); }
.card .c { display:block; color: var(--elx); font-family: ui-monospace, monospace; }
.ebar { display:grid; grid-template-columns: 3.2em 1fr 2.6em; align-items:center; gap:6px; margin-top:6px; font: 600 .85rem ui-monospace, monospace; }
.ebar .en { text-align:right; }
.elx { position:relative; height:14px; background: var(--line); border-radius:4px; overflow:hidden; }
.elx > div { height:100%; background: var(--elx); }
.ebar.them .elx > div { background: var(--oelx); }
.elx::after { content:""; position:absolute; inset:0; background: repeating-linear-gradient(to right, transparent 0 calc(10% - 2px), var(--panel) calc(10% - 2px) 10%); }
.panel { background: var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px; display:grid; gap:10px; align-content:start; }
.row { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
button.act, select { font: inherit; padding:6px 10px; border-radius:6px; border:1px solid var(--line); background: var(--panel); color: var(--ink); }
button.go { background: var(--accent); border-color: var(--accent); color:#1d1606; font-weight:700; }
ol { margin:0; padding-left: 1.2em; font-family: ui-monospace, monospace; font-size:.85rem; }
table { border-collapse: collapse; font-size:.9rem; } td, th { padding: 4px 8px; text-align:right; border-bottom: 1px solid var(--line); } td:first-child, th:first-child { text-align:left; }
.big { font-size: 1.1rem; font-weight: 700; }
[hidden] { display: none !important; }
.boardwrap { position: relative; }
.count { position: absolute; inset: 0; display: grid; place-items: center; font: 800 6rem system-ui, sans-serif; color: #fff; text-shadow: 0 4px 18px #000a; pointer-events: none; }
.steplab { font: .8rem ui-monospace, monospace; color: var(--muted); margin-top: 4px; min-height: 1.2em; }
</style></head><body>
<div class="wrap">
  <div class="leftcol">
    <div class="topbar"><span class="clock" id="clock">3:00</span><span class="phase" id="phase" hidden></span><span class="muted" id="sitt"></span><span class="tile" id="tile">hover a tile</span></div>
    <div class="boardwrap"><canvas id="board"></canvas><div class="count" id="count" hidden></div></div>
    <div class="steplab" id="steplab">—</div>
    <div class="ebar"><span>You</span><div class="elx"><div id="elx"></div></div><span class="en" id="elxn">—</span></div>
    <div class="ebar them"><span>Them</span><div class="elx"><div id="oelx"></div></div><span class="en" id="oelxn">—</span></div>
    <div class="hand" id="hand"></div>
  </div>
  <div class="panel">
    <div><h1>Defend it yourself</h1><div class="muted">Pick a situation and press New (Restart replays the same start). A 3-2-1 holds the game so you can get to the board, and you can already queue a card; then it runs in real time (a tick every 0.1 s) and a card lands on the next tick after you place it. Half and quarter speed slow everything down, and take effect immediately. Click a card (or press 1-4), then a tile: the tiles you can use light up, the others dim, and the card's footprint (a spell's radius) follows the cursor; the click plays at that tile's centre, the coordinates shown top right; Esc cancels, Space pauses. Do nothing and you wait.</div></div>
    <div class="row">
      <select id="tpl"><option value="hog_rider">Hog Rider</option><option value="battle_ram">Battle Ram</option><option value="melee">Melee (Mini P.E.K.K.A / Knight / Prince)</option><option value="supported_push">Supported push (Giant / Golem)</option><option value="arrows_value">5 · Arrows value</option><option value="fireball_value">6 · Fireball value</option><option value="emergency">10 · Emergency</option><option value="safe_opening">1 · Safe opening (they react)</option><option value="defend_counter">8 · Defend, then counter (they react)</option><option value="dont_overcommit">9 · Don't overcommit (they react)</option><option value="chill">Tempo · chill (calm, they're 8-10)</option><option value="press_calm">Tempo · press (calm, they're 0-3)</option><option value="desperate_even">Tempo · even game, lead 0-1 (chill)</option><option value="desperate">Tempo · desperate: late, down a tower, lead 0-1</option><option value="spell_finish">Tempo · spell finish (their tower nearly down)</option><option value="hold_after_defence">Held out · hold after defence (they're 7-10)</option><option value="punish">Held out · punish after defence (they're 0-3)</option></select>
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
//
// The board is the engine's (P0.1, 2026-10-29): 18 x 34 cells, cell i covers [i - 0.5, i + 0.5], so an engine point
// (x, y) is at canvas ((x + 0.5) * sx, H - (y + 0.5) * sy) and a click in a tile plays at that tile's centre, the
// integer (x, y) the engine places at. River, bridges and the dead back-row corners come from the server
// (bot77.sim.placement.board_geometry); a card's legal tiles are the engine's is_valid_placement at every cell centre,
// sent with every tick; its ghost is its engine footprint (building 1.0, troop 0.4) or spell radius.
let S = null, B = null, sel = null, pending = null, paused = false, lastPlay = null, lastSeed = null, lastTpl = null, lastThreat = "";
let gen = 0, running = false, hov = null;
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const cv = $("board"), ctx = cv.getContext("2d");
let W = 0, H = 0, sx = 1, sy = 1, dpr = 1, C = {};
const X = x => (x + 0.5) * sx, Y = y => H - (y + 0.5) * sy;   // engine point -> canvas pixel
const ABBR = {"Mini PEKKA":"MP","Mega Minion":"MM","Goblin Hut":"Hut","Battle Ram":"Ram","Hog Rider":"Hog","Spear Goblins":"gob","Barbarians":"barb","Musketeer":"Musk","Executioner":"Exec","Golemite":"gmt","Minions":"min"};
const maxHp = new Map();
const period = () => +$("speed").value;
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function post(path, body) { const r = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)}); return r.json(); }

function readColors() { for (const n of ["grass","grass2","river","bridge","dead","grid","gridlab","ours","theirs","accent","legal","illegal","hover"]) C[n] = css("--" + n); }
function resize() {   // backing store at device pixels, so 1 px lines stay crisp
  dpr = window.devicePixelRatio || 1;
  const r = cv.getBoundingClientRect();
  if (!r.width) return;
  cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
  W = cv.width; H = cv.height; sx = W / 18; sy = H / 34; draw();
}
const legalAt = (c, x, y) => !!c && !!c.legal && c.legal[y * 18 + x] === "1";
const deadAt = (x, y) => !!B && B.dead[y * 18 + x] === "1";
function unitR(u) {   // engine body radii: towers Tower::getCollisionRadius, buildings Building::COLLISION_RADIUS
  if (u.tower) return B && Math.abs(u.x - B.center_x) < 1 ? B.king_radius : (B ? B.princess_radius : 1.5);
  return u.building ? 1.0 : u.flying ? 0.45 : 0.55;
}
function unitUnder(h) {
  if (!S || !h) return null;
  let best = null, bd = 1e9;
  for (const u of S.units) { const d = Math.hypot(u.x - h.ex, u.y - h.ey); if (d <= unitR(u) + 0.15 && d < bd) { bd = d; best = u; } }
  return best;
}
function cellRect(x, y) { return [x * sx, H - (y + 1) * sy, sx, sy]; }
function ell(x, y, r) { ctx.beginPath(); ctx.ellipse(X(x), Y(y), r * sx, r * sy, 0, 0, 7); }
function ghost(c, x, y, ok, alpha) {
  const col = ok ? C.accent : C.theirs;
  ctx.save(); ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2 * dpr;
  if (c.spell) {
    ell(x, y, c.spell_radius); ctx.globalAlpha = .2 * alpha; ctx.fill(); ctx.globalAlpha = alpha; ctx.stroke();
    ell(x, y, .15); ctx.fill();
  } else {
    for (const [ox, oy] of (c.offsets && c.offsets.length ? c.offsets : [[0, 0]])) {
      ell(x + ox, y + oy, c.placement_radius); ctx.globalAlpha = .45 * alpha; ctx.fill(); ctx.globalAlpha = alpha; ctx.stroke();
    }
  }
  ctx.restore();
}
function label(text, px, py, fg, bg) {   // a small boxed label, kept on the canvas
  ctx.font = `600 ${12 * dpr}px ui-monospace,monospace`; ctx.textBaseline = "middle"; ctx.textAlign = "left";
  const w = ctx.measureText(text).width + 10 * dpr, h = 18 * dpr;
  px = Math.max(2, Math.min(W - w - 2, px)); py = Math.max(h / 2 + 2, Math.min(H - h / 2 - 2, py));
  ctx.fillStyle = bg; ctx.fillRect(px, py - h / 2, w, h); ctx.fillStyle = fg; ctx.fillText(text, px + 5 * dpr, py);
}

function draw() {
  if (!W) return;
  ctx.clearRect(0, 0, W, H);
  const c = sel !== null && S && S.hand[sel] ? S.hand[sel] : null;
  // tiles: a checkerboard so neighbours differ, the dead back-row corners dark
  for (let y = 0; y < 34; y++) for (let x = 0; x < 18; x++) {
    ctx.fillStyle = deadAt(x, y) ? C.dead : (x + y) % 2 ? C.grass : C.grass2; ctx.fillRect(...cellRect(x, y));
  }
  if (B) {   // the river band and the bridge cells, from the engine
    const [r0, r1] = B.river;
    ctx.fillStyle = C.river; ctx.fillRect(0, H - (r1 + 0.5) * sy, W, (r1 - r0) * sy);
    ctx.fillStyle = C.bridge;
    for (const bx of B.bridge_cols) ctx.fillRect(bx * sx, H - (r1 + 0.5) * sy, sx, (r1 - r0) * sy);
  }
  if (c) {   // the selected card's legal tiles (the engine's rule, this tick)
    for (let y = 0; y < 34; y++) for (let x = 0; x < 18; x++) {
      if (deadAt(x, y)) continue;
      ctx.fillStyle = legalAt(c, x, y) ? C.legal : C.illegal; ctx.fillRect(...cellRect(x, y));
    }
  }
  // every tile outlined; the lane split (between columns 8 and 9) and the river edges a little stronger
  const off = Math.round(dpr) % 2 ? 0.5 : 0;
  ctx.strokeStyle = C.grid; ctx.lineWidth = Math.max(1, Math.round(dpr));
  ctx.beginPath();
  for (let i = 0; i <= 18; i++) { const px = Math.round(i * sx) + off; ctx.moveTo(px, 0); ctx.lineTo(px, H); }
  for (let j = 0; j <= 34; j++) { const py = Math.round(H - j * sy) + off; ctx.moveTo(0, py); ctx.lineTo(W, py); }
  ctx.stroke();
  ctx.lineWidth = 2 * Math.max(1, Math.round(dpr)); ctx.beginPath();
  ctx.moveTo(Math.round(9 * sx), 0); ctx.lineTo(Math.round(9 * sx), H);
  if (B) for (const yy of B.river) { const py = Math.round(H - (yy + 0.5) * sy); ctx.moveTo(0, py); ctx.lineTo(W, py); }
  ctx.stroke();
  // coordinates on the edges: x along the bottom row, y up the left column
  ctx.fillStyle = C.gridlab; ctx.font = `600 ${Math.max(8, Math.min(11, sx / dpr * .42)) * dpr}px ui-monospace,monospace`;
  ctx.textAlign = "center"; ctx.textBaseline = "bottom";
  for (let x = 0; x < 18; x++) ctx.fillText(x, X(x), H - 1 * dpr);
  ctx.textAlign = "left"; ctx.textBaseline = "top";
  for (let y = 1; y < 34; y++) ctx.fillText(y, 2 * dpr, H - (y + 1) * sy + 1 * dpr);
  // the hovered tile
  const u0 = unitUnder(hov);
  if (hov) {
    const ok = !c || legalAt(c, hov.x, hov.y);
    ctx.fillStyle = "rgba(255,255,255,.22)"; ctx.fillRect(...cellRect(hov.x, hov.y));
    ctx.strokeStyle = ok ? C.hover : C.theirs; ctx.lineWidth = 2.5 * dpr; ctx.strokeRect(...cellRect(hov.x, hov.y));
  }
  if (!S) { if (hov) label(`(${hov.x}, ${hov.y})`, (hov.x + 1) * sx + 4 * dpr, H - (hov.y + 1) * sy, "#fff", "#000b"); return; }
  for (const u of S.units) {
    if (!maxHp.has(u.id) || u.hp > maxHp.get(u.id)) maxHp.set(u.id, u.hp);
    const full = u.max_hp || maxHp.get(u.id) || 1;
    const px = X(u.x), py = Y(u.y), r = unitR(u) * sx;
    ctx.fillStyle = u.team === 0 ? C.ours : C.theirs;
    if (u.tower || u.building) { ctx.globalAlpha = .9; ctx.fillRect(px - r, py - r, 2 * r, 2 * r); ctx.globalAlpha = 1; }
    else { ctx.beginPath(); ctx.arc(px, py, r, 0, 7); ctx.fill(); if (u.flying) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 * dpr; ctx.stroke(); } }
    if (u === u0) { ctx.strokeStyle = C.hover; ctx.lineWidth = 2.5 * dpr; u.tower || u.building ? ctx.strokeRect(px - r, py - r, 2 * r, 2 * r) : (ctx.beginPath(), ctx.arc(px, py, r + 2 * dpr, 0, 7), ctx.stroke()); }
    const f = Math.max(0, Math.min(1, u.hp / full));
    ctx.fillStyle = "#000a"; ctx.fillRect(px - r, py - r - 8 * dpr, 2 * r, 5 * dpr);
    ctx.fillStyle = f > .5 ? "#3fbf5a" : f > .25 ? "#e3b740" : "#e5533f"; ctx.fillRect(px - r, py - r - 8 * dpr, 2 * r * f, 5 * dpr);
    const lab = u.tower ? (u.hp > 0 ? Math.round(u.hp) : "") : (ABBR[u.name] ?? (u.name || "").slice(0, 5));
    if (lab !== "") { ctx.font = `600 ${(u.tower ? 13 : 10) * dpr}px ui-monospace,monospace`; ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(lab, px, py); }
  }
  if (pending) ghost(pending.card, pending.x, pending.y, true, .9);
  if (lastPlay) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 * dpr; ctx.setLineDash([6 * dpr, 5 * dpr]); ctx.strokeRect(...cellRect(lastPlay.x, lastPlay.y)); ctx.setLineDash([]); }
  if (hov && c) ghost(c, hov.x, hov.y, legalAt(c, hov.x, hov.y), 1);
  if (hov) label(`(${hov.x}, ${hov.y})`, (hov.x + 1) * sx + 4 * dpr, H - (hov.y + 1) * sy, "#fff", "#000b");
  if (u0) label(`${u0.name} ${Math.round(u0.hp).toLocaleString()} / ${(u0.max_hp || maxHp.get(u0.id) || 0).toLocaleString()} HP`,
                X(u0.x) - 60 * dpr, Y(u0.y) - unitR(u0) * sy - 20 * dpr, "#fff", u0.team === 0 ? "#1d3f80e6" : "#7a1f18e6");
}
function tileText() {
  if (!hov) { $("tile").textContent = "hover a tile"; return; }
  const c = sel !== null && S && S.hand[sel] ? S.hand[sel] : null, u = unitUnder(hov);
  let t = `tile (${hov.x}, ${hov.y})`;
  if (c) t += legalAt(c, hov.x, hov.y) ? ` · ${c.name} ✓` : ` · ${c.name}: not here`;
  else if (u) t += ` · ${u.name} ${Math.round(u.hp)} HP`;
  $("tile").textContent = t;
}

// the four card buttons, made once
const cardBtns = [0, 1, 2, 3].map(i => {
  const b = document.createElement("button"); b.className = "card"; b.type = "button";
  b.addEventListener("pointerdown", ev => { ev.preventDefault(); pick(i); });   // on press: a click straddling an update can't be lost
  $("hand").appendChild(b); return b;
});
function pick(i) {
  if (!S || S.done || !S.hand[i] || S.hand[i].cost > S.elixir + 1e-6) return;
  sel = sel === i ? null : i; updateHand(); draw(); tileText();
}
const mmss = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
function updateHand() {
  cardBtns.forEach((b, i) => {
    const c = S && S.hand[i];
    const label = c ? `<span style="color:var(--muted);font-size:.75rem">${i + 1}</span> ${c.name}<span class="c">${c.cost}</span>` : "";
    if (b.dataset.label !== label) { b.innerHTML = label; b.dataset.label = label; }
    b.disabled = !c || S.done || c.cost > S.elixir + 1e-6;
    b.classList.toggle("sel", sel === i);
  });
  if (!S) return;
  $("elx").style.width = (S.elixir * 10) + "%"; $("elxn").textContent = S.elixir.toFixed(1);
  $("oelx").style.width = (S.opp_elixir * 10) + "%"; $("oelxn").textContent = S.opp_elixir.toFixed(1);
  if (S.clock != null && B) {   // the in-game clock: counts down to 3:00 of regulation, then overtime
    const left = B.regulation_end - S.clock;
    $("clock").textContent = left > 0 ? mmss(Math.ceil(left)) : "OT +" + mmss(-left);
    const ph = S.clock >= B.triple_at ? "3x elixir" : S.clock >= B.double_at ? "2x elixir" : "";
    $("phase").textContent = ph; $("phase").hidden = !ph;
  }
  $("sitt").textContent = `situation ${(+S.t).toFixed(1)} s`;
}

function cellAt(ev) {   // the tile under the pointer, and the continuous engine point
  const r = cv.getBoundingClientRect();
  const fx = (ev.clientX - r.left) / r.width * 18, fy = (r.bottom - ev.clientY) / r.height * 34;
  return {x: Math.min(17, Math.max(0, Math.floor(fx))), y: Math.min(33, Math.max(0, Math.floor(fy))), ex: fx - 0.5, ey: fy - 0.5};
}
cv.addEventListener("pointermove", ev => { hov = cellAt(ev); draw(); tileText(); });
cv.addEventListener("pointerleave", () => { hov = null; draw(); tileText(); });
cv.addEventListener("pointerdown", ev => {
  if (!S || sel === null || S.done) return;
  const {x, y} = cellAt(ev), c = S.hand[sel];
  if (!legalAt(c, x, y)) { flash(c.spell ? "spells can't go on that tile" : `${c.name} can't go on tile (${x}, ${y})`); return; }
  pending = {slot: sel, x, y, card: c}; sel = null; updateHand(); draw(); tileText();
});
let flashUntil = 0;
function flash(msg) { $("steplab").textContent = msg; flashUntil = performance.now() + 2000; }

async function loop(g) {
  running = true;
  while (g === gen && S && !S.done) {
    const t0 = performance.now();
    if (!paused) {
      const act = pending; pending = null;
      const out = await post("/step", act ? {slot: act.slot, x: act.x, y: act.y} : {});
      if (g !== gen) return;                     // a New / Restart happened while this step was in flight
      if (act) lastPlay = act;
      S = out.state; updateHand(); draw(); tileText();
      $("plays").innerHTML = (out.plays || []).map(p => `<li>${(+p.t).toFixed(1)}s ${p.card} → (${p.x}, ${p.y})${p.landed ? "" : " — refused" + (p.why ? ": " + p.why : "")}</li>`).join("");
      if (S.done || performance.now() > flashUntil) $("steplab").textContent = S.done ? "finished" : `${(+S.t).toFixed(1)} s`;
      if (out.result) { showResult(out.result); break; }
    } else { $("steplab").textContent = "paused"; }
    await sleep(Math.max(0, period() - (performance.now() - t0)));
  }
  if (g === gen) running = false;
}

async function begin(seed) {
  const g = ++gen;                                 // cancels any running loop
  sel = null; pending = null; lastPlay = null; paused = false; maxHp.clear(); hov = null;
  $("pause").textContent = "Pause"; $("result").innerHTML = ""; $("plays").innerHTML = "";
  const out = await post("/new", {template: $("tpl").value, seed});
  if (g !== gen) return;
  lastSeed = out.seed; lastTpl = $("tpl").value; lastThreat = out.threat.join(" + ") || "nothing yet";
  B = out.board || B; S = out.state; updateHand(); draw();
  $("setup").innerHTML = `Threat <b>${lastThreat}</b>${out.plan ? ` · their plan <b>${PLAN[out.plan] || out.plan}</b>` : ""} · your elixir <b>${out.start_elixir[0]}</b> · theirs <b>${out.start_elixir[1]}</b> · seed ${out.seed}`;
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
  <tr><td>You</td>${fmt(R.you)}</tr><tr><td>Scripted defence</td>${fmt(R.scripted)}</tr><tr><td>Bot (v3)</td>${fmt(R.model)}</tr></table>
  <div class="muted" style="margin-top:6px">Same start for all three; the opponent clone plays on, so its plays can differ.</div>`;
}

const PLAN = {tank_back: "tank in the back", bridge_rush: "bridge rush", support_back: "support in the back",
  wait_defend: "wait and defend", defend: "defend what crosses", punish: "punish an overcommit",
  tempo: "defend what crosses, punish a commitment, push at a full bar"};
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
readColors();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readColors(); draw(); });
new ResizeObserver(resize).observe(cv);
fetch("/board").then(r => r.json()).then(b => { B = b; draw(); });
loadSaved(); updateHand(); resize();
</script></body></html>"""


if __name__ == "__main__":
    main()
