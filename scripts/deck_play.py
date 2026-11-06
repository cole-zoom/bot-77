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

"You play <deck> vs the model" (2026-11-04, Cole: play as the decks first, to teach them): you pilot the deck bot's
deck as team 1, shown from your side (the board turned round, y -> 33 - y), against the live model on our deck
(bot77.rl.model_player: v5 e3, no nudge, sampling, 0.1 s). The deck's rules run alongside without playing: the Events
list shows what they would play ("Rules"), and each of your plays says whether a rule had it (same card within 3 s) or
not (a gap, or a rule that's wrong). The end of the game sums it up per rule; the log keeps every suggestion.
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
REPEAT_S = 5.0  # "you play the deck": a rule's repeated suggestion shows once this often
MATCH_S = 3.0  # a play of yours "matches" a rule's suggestion of the same card this close in time
NOTES = ROOT / "data/live/deck_play_notes.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8778)
    ap.add_argument("--log", default=str(LOG))
    ap.add_argument("--ckpt", default=None, help="the model in 'you play the deck' mode (default: the live one)")
    a = ap.parse_args()
    log = Path(a.log)

    import clash_royale_env as cre
    import bot77.rl.decks  # noqa: F401  (registers every deck)
    from bot77.rl.deck_bots import DECKS, OUR_DECK, card_id
    from bot77.sim import cards as sc
    from bot77.sim import placement as pl
    from bot77.rl.model_player import LIVE_CKPT, ModelPlayer
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

    def flip_mask(m):  # board rows from y = 0 -> the page's rows from your back row
        W = len(m) // 34
        return "".join(m[r * W:(r + 1) * W] for r in range(33, -1, -1))

    def state_as(g, done=False):
        """The game from team 1's side, as the page draws team 0: y -> 33 - y, the teams swapped."""
        units = [{"id": u.id, "name": u.name, "team": 1 - u.team, "x": round(u.x, 2), "y": round(33.0 - u.y, 2),
                  "hp": round(u.hp), "flying": bool(u.flying), "building": bool(u.building), "tower": bool(u.tower),
                  "card_id": int(u.card_id), "max_hp": pl.max_hp(g, u), "target": int(u.target_id),
                  "targetable": bool(u.targetable), "deploy": int(u.deploy_ticks)} for u in g.get_units()]
        hand = [{"id": c, "name": names.get(c, str(c)), "cost": cost.get(c, 0), **{k: v for k, v in ghost(c).items()
                 if k in ("spell", "building", "placement_radius", "spell_radius", "offsets")},
                 "legal": flip_mask(pl.legal_mask(g, c, 1))} for c in g.get_hand_for_team(1)]
        return {"units": units, "hand": hand, "elixir": round(g.get_elixir_for_team(1), 2),
                "opp_elixir": round(g.get_elixir_for_team(0), 2),
                "towers": [[max(0, g.get_tower_hp(t, s)) for s in range(3)] for t in (1, 0)],
                "t": game.get("t", 0), "done": done, "clock": round(g.get_current_tick() * pl.TICK_S, 3),
                "tick": g.get_current_tick(), "tempo": None, "next": None}

    def summary(sugg, plays):
        """Per rule: how often it would have played and how often you did the same; your plays no rule had."""
        per = {}
        for s in sugg:
            d = per.setdefault(s["rule"], [0, 0])
            d[0] += 1
            d[1] += s["matched"]
        return {"rules": {k: {"suggested": v[0], "you_agreed": v[1]} for k, v in per.items()},
                "your_plays": sum(p["landed"] for p in plays),
                "no_rule_had": sum(p["landed"] and not p.get("rule") for p in plays)}

    def result_as(g, bot, sugg, plays):
        tw = [[max(0, g.get_tower_hp(t, s)) for s in range(3)] for t in (1, 0)]
        crowns = [3 if tw[1][0] <= 0 else sum(h <= 0 for h in tw[1][1:]), 3 if tw[0][0] <= 0 else sum(h <= 0 for h in tw[0][1:])]
        if crowns[0] != crowns[1]:
            win = 1 if crowns[0] > crowns[1] else -1
        else:
            w = int(g.resolve_timeout_outcome()) if g.is_game_over() or g.get_current_tick() >= 3600 else 0
            win = 1 if w == 1 else -1 if w == 0 else 0
        sm = summary(sugg, plays)
        rules = " · ".join(f"{k} {v['you_agreed']}/{v['suggested']}" for k, v in sorted(sm["rules"].items()))
        return {"deck": f"the model (you on {bot.NAME})", "result": win, "crowns": crowns, "towers": tw,
                "clock": round(g.get_current_tick() / 10, 1), "fired": bot.fired, "opp_plays": len(bot.mine),
                "as": True, "summary": sm,
                "summary_text": f"You agreed with the rules on: {rules or 'nothing (no rule suggested a play)'}. "
                                f"{sm['no_rule_had']} of your {sm['your_plays']} plays no rule had."}

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

    def refusal(legal, el, cid):
        """Why a play didn't land: the tile, the elixir, or the engine's 1 s lock on a card that just came in."""
        if not legal:
            return "not a legal tile"
        if cid is None or el < cost.get(cid, 0):
            return f"not enough elixir ({el:.1f})"
        return "the card just came in (1 s lock)"

    def step_as(body):
        """One tick of "you play the deck": your play (team 1, page frame = team 1's own frame), the model's, and the
        rules' suggestion alongside."""
        g, bot, model = game["g"], game["bot"], game["model"]
        slot = body.get("slot")
        k, x, y = (4, 0.0, 0.0) if slot is None else (int(slot), float(body["x"]), float(body["y"]))
        hand1, hand0 = list(g.get_hand_for_team(1)), list(g.get_hand())
        legal = k < 4 and k < len(hand1) and g.is_valid_placement(hand1[k], x, 33.0 - y, 1)
        el1 = float(g.get_elixir_for_team(1))
        sp0, sp1 = float(g.get_elixir_spent(0)), float(g.get_elixir_spent(1))
        events = list(bot.events)
        bot.events.clear()
        sug = bot.shadow_tick()
        # "dropped / skipped X" only means you didn't play the rule's card: left out
        events += [{**e, "who": "Rules"} for e in bot.events if not e["text"].startswith(("dropped ", "skipped "))]
        bot.events.clear()
        t = game["t"]
        if sug:
            rule, card, (sx, sy) = sug
            game["sugg"].append({"t": t, "rule": rule, "card": card, "x": round(sx, 1), "y": round(sy, 1), "matched": False})
            events.append({"who": "Rules", "rule": rule, "text": f"would play {card} at ({sx:.0f}, {33 - sy:.0f})"})
        mv = model.act(g)
        k0, x0, y0 = mv if mv else (4, 0.0, 0.0)
        g.step_self_play(k0, x0, y0, k, x, y, 1)
        game["n"] += 1
        game["t"] = round(game["n"] * pl.TICK_S, 3)
        if mv and float(g.get_elixir_spent(0)) > sp0 + 0.5:
            bot.note_play(hand0[k0], x0, y0)
        if k < 4:
            landed = float(g.get_elixir_spent(1)) > sp1 + 0.5
            card = names.get(hand1[k], str(hand1[k])) if k < len(hand1) else "?"
            play = {"t": game["t"], "card": card, "card_id": int(hand1[k]) if k < len(hand1) else None,
                    "x": x, "y": y, "landed": bool(landed)}
            if landed:
                bot.note_own_play(hand1[k], x, 33.0 - y)
                m = next((s for s in reversed(game["sugg"]) if not s["matched"] and s["card"] == bot.names.get(hand1[k])
                          and abs(s["t"] - game["t"]) <= MATCH_S), None)
                if m is not None:
                    m["matched"] = True
                    play["rule"] = m["rule"]
                    d = ((m["x"] - x) ** 2 + (m["y"] - (33.0 - y)) ** 2) ** 0.5
                    events.append({"who": "You", "rule": m["rule"], "text": f"{card}: same as the rule ({d:.1f} tiles from its spot)"})
                else:
                    events.append({"who": "You", "rule": "GAP?", "text": f"{card}: no rule had this play"})
            else:
                play["why"] = refusal(legal, el1, hand1[k] if k < len(hand1) else None)
            game["plays"].append(play)
        # a pending suggestion of the same card counts as yours a little later too
        for p in game["plays"][-3:]:
            if p["landed"] and not p.get("rule"):
                m = next((s for s in game["sugg"] if not s["matched"] and s["card"] == bot.names.get(p["card_id"])
                          and 0 <= s["t"] - p["t"] <= MATCH_S), None)
                if m is not None:
                    m["matched"], p["rule"] = True, m["rule"]
                    events.append({"who": "Rules", "rule": m["rule"], "text": f"agrees with your {p['card']}"})
        # the rules re-fire every tick you don't follow them: the same rule once every REPEAT_S on the page
        # (the log keeps them all)
        game["events"] += events
        shown, last = [], game.setdefault("last_shown", {})
        for e in events:
            k = (e.get("who"), e["rule"], e["text"].startswith("would play"))  # the lane can flip-flop: per rule
            if e.get("who") == "You" or game["t"] - last.get(k, -99) >= REPEAT_S:
                last[k] = game["t"]
                shown.append(e)
        events = shown
        done = bool(g.is_game_over() or g.get_current_tick() >= 3600)
        out = {"state": state_as(g, done), "plays": game["plays"], "events": events}
        if done:
            out["result"] = {"you": result_as(g, bot, game["sugg"], game["plays"])}
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as f:
                f.write(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "you_play_the_deck",
                                    "deck": game["key"], "seed": game["seed"], "model": model.ckpt,
                                    "plays": game["plays"], "suggestions": game["sugg"], "events": game["events"],
                                    "result": out["result"]["you"]}, default=float) + "\n")
            game["g"] = None
        return out

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
                                                  "templates": [{"value": f"as:{k}", "name": f"You play {c.NAME} vs the model"}
                                                                for k, c in DECKS.items()]
                                                  + [{"value": k, "name": f"vs {c.NAME} (you play the bot)"} for k, c in DECKS.items()]}))
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
                tpl = str(body.get("template") or "")
                as_deck = tpl.startswith("as:")
                key = tpl[3:] if as_deck else tpl
                key = key if key in DECKS else next(iter(DECKS))
                seed = int(body.get("seed") or int(time.time()) % 100_000)
                bot = DECKS[key](seed)
                none = cre.TowerTroopType.NONE
                g = cre.ClashRoyaleEnv(ours, bot.deck_ids, 3600, none, none)
                g.seed(seed)
                g.reset()
                bot.bind(g)
                game.clear()
                game.update({"g": g, "bot": bot, "key": key, "seed": seed, "t": 0, "n": 0, "plays": [], "events": [],
                             "as": as_deck, "sugg": []})
                if as_deck:
                    game["model"] = ModelPlayer(a.ckpt or LIVE_CKPT, seed)
                st = state_as(g) if as_deck else state(g)
                return self.send(200, json.dumps({"state": st, "situation": bot.NAME, "seed": seed,
                                                  "threat": [f"the model ({game['model'].ckpt})" if as_deck else bot.NAME],
                                                  "plan": None,
                                                  "board": {**pl.board_geometry(g), **pl.CLOCK},
                                                  "start_elixir": [round(g.get_elixir_for_team(1 if as_deck else 0), 1),
                                                                   round(g.get_elixir_for_team(0 if as_deck else 1), 1)]}))
            if self.path == "/client_error":
                print(f"page error at clock {body.get('clock')}: {body.get('error', '')[:2000]}", flush=True)
                return self.send(200, "{}")
            if self.path == "/save":
                notes = json.loads(NOTES.read_text()) if NOTES.exists() else []
                g = game.get("g")
                notes.append({"template": body["template"], "seed": int(body["seed"]), "note": body.get("note", ""),
                              "you_played": "the deck" if game.get("as") else "our deck",
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
                if game.get("as"):
                    return self.send(200, json.dumps(step_as(body), default=float))
                slot = body.get("slot")
                k, x, y = (4, 0.0, 0.0) if slot is None else (int(slot), float(body["x"]), float(body["y"]))
                hand = list(g.get_hand())
                legal = k < 4 and k < len(hand) and g.is_valid_placement(hand[k], x, y, 0)
                el0 = float(g.get_elixir_for_team(0))
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
                        play["why"] = refusal(legal, el0, hand[k] if k < len(hand) else None)
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
