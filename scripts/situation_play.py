"""Play the defence situations yourself (2026-10-28, Cole: "can u let me play live? to defend").

  uv run --no-sync --with gymnasium==1.3.0 python scripts/situation_play.py [--ckpt data/models/rl_clones_v2/final.pth]
  then open http://localhost:8777

A local page (the simulator runs on this Mac, so it can't be a published page): the same top-down board as the replays,
your hand beside it, the game in real time (you act on any engine tick, 0.1 s today; the opponent clone decides once a
second as trained; the bot it is compared with still decides once a second). Click a card, then a tile, to play
it; otherwise you wait. The page itself (sprites, sound, hotkeys) is static files in scripts/play_page/, served from
here; docs/simulator.md has the hotkeys. Each situation is drawn exactly as in training (bot77.rl.situations, the opponent clone
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
STATIC = Path(__file__).resolve().parent / "play_page"  # index.html, style.css and the scripts; served as-is
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                ".js": "text/javascript; charset=utf-8"}
LOG = ROOT / "data/live/situation_play.jsonl"
SAVED = ROOT / "data/live/saved_situations.json"  # bookmarked starts (Save button): situation type + seed + note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/models/rl_clones_v3/final.pth")
    ap.add_argument("--model-ticks", type=int, default=1, help="the model's decision interval in 0.1 s ticks (v3: 1)")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--log", default=str(LOG), help="where finished games are appended (a test server writes elsewhere)")
    ap.add_argument("--engine-collision", action="store_true",
                    help="the engine's old pairwise pushing instead of the client's (Fidelity::clientCollision, notes §5)")
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
    import clash_royale_env as cre
    cre.set_client_collision(not a.engine_collision)  # Cole 10-29: a Mini P.E.K.K.A shouldn't slow his Hog
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

    catalog = pl.catalog()  # every card in the engine's registry: the page's sprites, ghosts and hover circles
    nxt = {}  # (hand, elixir spent) -> the next card (bot77.sim.placement.next_card), asked again only after a play
    specs = {}  # card id -> its ghost on the page (bot77.sim.placement.card_spec), read from the engine once

    def ghost(c):
        if c not in specs:
            specs[c] = pl.card_spec(c)
        return specs[c]

    def state(env, done=False):
        g = env.game
        # target / targetable / deploy are for the page only (hover, ghosted deploys, which attack hit what)
        units = [{"id": u.id, "name": u.name, "team": u.team, "x": round(u.x, 2), "y": round(u.y, 2), "hp": round(u.hp),
                  "flying": bool(u.flying), "building": bool(u.building), "tower": bool(u.tower),
                  "card_id": int(u.card_id), "max_hp": pl.max_hp(g, u), "target": int(u.target_id),
                  "targetable": bool(u.targetable), "deploy": int(u.deploy_ticks)} for u in g.get_units()]
        key = (tuple(g.get_hand()), round(float(g.get_elixir_spent(0)), 3))
        if key not in nxt:
            nxt.clear()
            nxt[key] = pl.next_card(g)
        n = nxt[key]
        # per card: the engine's own legality at every cell centre, this tick (buildings on the board change it)
        hand = [{"id": c, "name": names.get(c, str(c)), "cost": sit.COST.get(c, 0), **{k: v for k, v in ghost(c).items()
                 if k in ("spell", "building", "placement_radius", "spell_radius", "offsets")},
                 "legal": pl.legal_mask(g, c)} for c in g.get_hand()]
        return {"units": units, "hand": hand, "elixir": round(g.get_elixir(), 2),
                "opp_elixir": round(g.get_elixir_for_team(1), 2), "towers": [[max(0, g.get_tower_hp(t, s)) for s in range(3)] for t in (0, 1)],
                "t": game.get("t", 0), "done": done, "clock": round(g.get_current_tick() * pl.TICK_S, 3),
                "tick": g.get_current_tick(),
                # a full-game tempo start: the tempo the game is in now and the reward so far (bot77.rl.situations)
                "tempo": ({"mode": env.situation.mode, "total": round(env.situation.total, 3)}
                          if env.situation is not None and env.situation.spec.get("full_game") else None),
                "next": None if n is None else {"id": n, "name": names.get(n, str(n)), "cost": sit.COST.get(n, 0)}}

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
                return self.send(200, (STATIC / "index.html").read_bytes(), STATIC_TYPES[".html"])
            if self.path.startswith("/static/"):  # the page's own files, by bare name only
                f = STATIC / self.path[len("/static/"):].split("?")[0]
                if f.parent == STATIC and f.suffix in STATIC_TYPES and f.is_file():
                    return self.send(200, f.read_bytes(), STATIC_TYPES[f.suffix])
                return self.send(404, "{}")
            if self.path == "/catalog":
                return self.send(200, json.dumps({"cards": catalog, "tower_range": pl.TOWER_RANGE, "tick_s": pl.TICK_S}))
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
                game.update({"env": env, "tpl": name, "seed": seed, "t": 0, "n": 0, "plays": [], "spec_name": spec["name"],
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
                game["n"] += 1  # ticks played; t in engine seconds (exactly the old round(t + 0.1, 1) at 0.1 s a tick)
                game["t"] = round(game["n"] * pl.TICK_S, 3)
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
                                            "seed": game["seed"], "plays": game["plays"], "result": out["result"],
                                            "engine": {k: getattr(cre, "get_" + k)() for k in
                                                       ("client_collision", "tower_reach_footprint", "obstacle_slide",
                                                        "bridge_own_line")}}) + "\n")
                    game["env"] = None
                return self.send(200, json.dumps(out, default=float))
            self.send(404, "{}")

    print(f"open http://localhost:{a.port}  (model: {Path(ckpt).parent.name}/{Path(ckpt).name})", flush=True)
    HTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
