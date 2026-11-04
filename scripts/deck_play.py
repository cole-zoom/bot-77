"""Play the scripted deck opponents yourself (2026-11-04, Cole: "I'll play each one and 'act' like the bot ... once I get
these bots to my liking, likely they will be ready for training").

  uv run --no-sync python scripts/deck_play.py [--port 8778]
  then open http://localhost:8778

The play page (scripts/play_page/, the same board, hand, hotkeys and sounds as scripts/situation_play.py) running full
games: you play the bot's deck (Giant, Battle Ram, Mini P.E.K.K.A, Mega Minion, Bomber, Goblin Hut, Fireball, Arrows)
as team 0 against one of the scripted decks (bot77.rl.decks: docs/decks/*.md as code), from 0:00 to the end of the
game. The Events list shows every rule the opponent fires (its id in the deck's doc, what it does, its reaction
delay) next to its plays, so a bad rule can be pointed at. Save bookmarks the deck + seed + a note, with the clock and
the last rules fired, in data/live/deck_play_notes.json; every finished game (both sides' plays, every rule event,
the result) is appended to data/live/deck_play.jsonl (both gitignored).
"""
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from situation_play import STATIC, STATIC_TYPES  # noqa: E402  (the page's files)

LOG = ROOT / "data/live/deck_play.jsonl"
NOTES = ROOT / "data/live/deck_play_notes.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8778)
    ap.add_argument("--log", default=str(LOG))
    a = ap.parse_args()
    log = Path(a.log)

    import clash_royale_env as cre
    import bot77.rl.decks  # noqa: F401  (registers every deck)
    from bot77.rl.deck_bots import DECKS, OUR_DECK, card_id
    from bot77.sim import cards as sc
    from bot77.sim import placement as pl
    cre.set_client_collision(True)  # as the situation page (Cole 10-29)
    names = {i: c["name"] for i, c in sc.engine_cards().items()}
    cost = {i: float(c["cost"]) for i, c in sc.engine_cards().items()}
    ours = [card_id(n) for n in OUR_DECK]
    catalog = pl.catalog()
    specs, nxt = {}, {}
    game = {}

    def ghost(c):
        if c not in specs:
            specs[c] = pl.card_spec(c)
        return specs[c]

    def state(g, done=False):
        units = [{"id": u.id, "name": u.name, "team": u.team, "x": round(u.x, 2), "y": round(u.y, 2), "hp": round(u.hp),
                  "flying": bool(u.flying), "building": bool(u.building), "tower": bool(u.tower),
                  "card_id": int(u.card_id), "max_hp": pl.max_hp(g, u), "target": int(u.target_id),
                  "targetable": bool(u.targetable), "deploy": int(u.deploy_ticks)} for u in g.get_units()]
        key = (tuple(g.get_hand()), round(float(g.get_elixir_spent(0)), 3))
        if key not in nxt:
            nxt.clear()
            nxt[key] = pl.next_card(g)
        n = nxt[key]
        hand = [{"id": c, "name": names.get(c, str(c)), "cost": cost.get(c, 0), **{k: v for k, v in ghost(c).items()
                 if k in ("spell", "building", "placement_radius", "spell_radius", "offsets")},
                 "legal": pl.legal_mask(g, c)} for c in g.get_hand()]
        return {"units": units, "hand": hand, "elixir": round(g.get_elixir(), 2),
                "opp_elixir": round(g.get_elixir_for_team(1), 2),
                "towers": [[max(0, g.get_tower_hp(t, s)) for s in range(3)] for t in (0, 1)],
                "t": game.get("t", 0), "done": done, "clock": round(g.get_current_tick() * pl.TICK_S, 3),
                "tick": g.get_current_tick(), "tempo": None,
                "next": None if n is None else {"id": n, "name": names.get(n, str(n)), "cost": cost.get(n, 0)}}

    def result(g, bot):
        tw = [[max(0, g.get_tower_hp(t, s)) for s in range(3)] for t in (0, 1)]
        crowns = [3 if tw[1][0] <= 0 else sum(h <= 0 for h in tw[1][1:]), 3 if tw[0][0] <= 0 else sum(h <= 0 for h in tw[0][1:])]
        if crowns[0] != crowns[1]:
            win = 1 if crowns[0] > crowns[1] else -1
        else:
            w = int(g.resolve_timeout_outcome()) if g.is_game_over() or g.get_current_tick() >= 3600 else 0
            win = 1 if w == 0 else -1 if w == 1 else 0  # the engine's winning team (or -1 a draw)
        return {"deck": bot.NAME, "result": win, "crowns": crowns, "towers": tw, "clock": round(g.get_current_tick() / 10, 1),
                "fired": bot.fired, "opp_plays": len(bot.mine)}

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
            if self.path.startswith("/static/"):
                f = STATIC / self.path[len("/static/"):].split("?")[0]
                if f.parent == STATIC and f.suffix in STATIC_TYPES and f.is_file():
                    return self.send(200, f.read_bytes(), STATIC_TYPES[f.suffix])
                return self.send(404, "{}")
            if self.path == "/templates":  # the page's picker: the decks instead of the situations
                return self.send(200, json.dumps({"title": "Play the deck bots", "label": "Opponent deck",
                                                  "templates": [{"value": k, "name": c.NAME} for k, c in DECKS.items()]}))
            if self.path == "/catalog":
                return self.send(200, json.dumps({"cards": catalog, "tower_range": pl.TOWER_RANGE, "tick_s": pl.TICK_S}))
            if self.path == "/board":
                g = cre.ClashRoyaleEnv(ours, ours, 3600)
                g.reset()
                return self.send(200, json.dumps({**pl.board_geometry(g), **pl.CLOCK}))
            if self.path == "/saved":
                return self.send(200, NOTES.read_text() if NOTES.exists() else "[]")
            self.send(404, "{}")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/new":
                key = body.get("template") if body.get("template") in DECKS else next(iter(DECKS))
                seed = int(body.get("seed") or int(time.time()) % 100_000)
                bot = DECKS[key](seed)
                none = cre.TowerTroopType.NONE
                g = cre.ClashRoyaleEnv(ours, bot.deck_ids, 3600, none, none)
                g.seed(seed)
                g.reset()
                bot.bind(g)
                game.clear()
                game.update({"g": g, "bot": bot, "key": key, "seed": seed, "t": 0, "n": 0, "plays": [], "events": []})
                return self.send(200, json.dumps({"state": state(g), "situation": bot.NAME, "seed": seed,
                                                  "threat": [bot.NAME], "plan": None,
                                                  "board": {**pl.board_geometry(g), **pl.CLOCK},
                                                  "start_elixir": [round(g.get_elixir(), 1),
                                                                   round(g.get_elixir_for_team(1), 1)]}))
            if self.path == "/client_error":
                print(f"page error at clock {body.get('clock')}: {body.get('error', '')[:2000]}", flush=True)
                return self.send(200, "{}")
            if self.path == "/save":
                notes = json.loads(NOTES.read_text()) if NOTES.exists() else []
                g = game.get("g")
                notes.append({"template": body["template"], "seed": int(body["seed"]), "note": body.get("note", ""),
                              "threat": body.get("threat", ""), "saved": time.strftime("%Y-%m-%d %H:%M"),
                              "clock": round(g.get_current_tick() / 10, 1) if g is not None else None,
                              "last_rules": game.get("events", [])[-5:]})
                NOTES.parent.mkdir(parents=True, exist_ok=True)
                NOTES.write_text(json.dumps(notes, indent=1))
                return self.send(200, json.dumps(notes))
            if self.path == "/step":
                g, bot = game.get("g"), game.get("bot")
                if g is None:
                    return self.send(400, json.dumps({"error": "start a game first"}))
                slot = body.get("slot")
                k, x, y = (4, 0.0, 0.0) if slot is None else (int(slot), float(body["x"]), float(body["y"]))
                hand = list(g.get_hand())
                legal = k < 4 and k < len(hand) and g.is_valid_placement(hand[k], x, y, 0)
                spent0 = float(g.get_elixir_spent(0))
                mv = bot.tick()
                k1, x1, y1 = mv if mv else (4, 0.0, 0.0)
                g.step_self_play(k, x, y, k1, x1, y1, 1)
                game["n"] += 1
                game["t"] = round(game["n"] * pl.TICK_S, 3)
                if k < 4:
                    landed = float(g.get_elixir_spent(0)) > spent0 + 0.5
                    play = {"t": game["t"], "card": names.get(hand[k], str(hand[k])), "card_id": int(hand[k]),
                            "x": x, "y": y, "landed": bool(landed)}
                    if landed:
                        bot.note_play(hand[k], x, y)
                    else:
                        play["why"] = "not a legal tile" if not legal else "not enough elixir (or the card just came)"
                    game["plays"].append(play)
                events = list(bot.events)
                bot.events.clear()
                game["events"] += events
                done = bool(g.is_game_over() or g.get_current_tick() >= 3600)
                out = {"state": state(g, done), "plays": game["plays"], "events": events}
                if done:
                    out["result"] = {"you": result(g, bot)}
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open("a") as f:
                        f.write(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "deck": game["key"],
                                            "seed": game["seed"], "plays": game["plays"], "opp_plays": bot.mine,
                                            "events": game["events"], "result": out["result"]["you"]}, default=float) + "\n")
                    game["g"] = None
                return self.send(200, json.dumps(out, default=float))
            self.send(404, "{}")

    print(f"open http://localhost:{a.port}  (decks: {', '.join(c.NAME for c in DECKS.values())})", flush=True)
    HTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
