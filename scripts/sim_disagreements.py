"""S3 failed its gate: the plays where the search and Cole disagree, with the frames, for Cole to say whether the sim or
the review is wrong (docs/simulator.md S3). Read-only page, local (data/reviews is gitignored).

Per play: the bot's frame with its tap (yellow) and the search's best spot (green) drawn on it, then what the sim thinks
happens after each, top-down at +0 / 2 / 4 / 7 / 10 s (blue ours, red theirs, towers with their HP %).

  uv run python scripts/sim_disagreements.py --run s3_v1 -> data/reviews/sim_<run>/index.html
"""
import argparse
import html
import json
from pathlib import Path

import numpy as np

from bot77.sim import cards as sc
from bot77.sim import store
from bot77.sim.geometry import engine_to_ours
from bot77.sim.search import Candidate, trace

AT = (0.0, 2.0, 4.0, 7.0, 10.0)
Y0, Y1 = 2.0, 27.5  # the arena in our units (y), drawn
RIVER = (engine_to_ours(0, 17.5)[1], engine_to_ours(0, 15.5)[1])
TOWERS = {"opp_king": (8.5, 30.5), "opp_l": (3.0, 27.0), "opp_r": (14.0, 27.0),
          "own_king": (8.5, 2.5), "own_l": (3.0, 6.0), "own_r": (14.0, 6.0)}  # engine centres


def short(card: str) -> str:
    words = [w for w in card.replace(".", "").split() if w]
    return (words[0][:3] if len(words) == 1 else "".join(w[0] for w in words[:3])).upper() if words else "?"


def arena_svg(state: dict, mark: tuple | None, title: str) -> str:
    w, h = 18.0, Y1 - Y0
    s = [f'<svg viewBox="0 {Y0} {w} {h}" class="arena"><rect x="0" y="{Y0}" width="18" height="{h}" class="ground"/>',
         f'<rect x="0" y="{RIVER[0]:.2f}" width="18" height="{RIVER[1] - RIVER[0]:.2f}" class="river"/>']
    for bx in (2.0, 14.0):
        s.append(f'<rect x="{bx}" y="{RIVER[0]:.2f}" width="2" height="{RIVER[1] - RIVER[0]:.2f}" class="bridge"/>')
    for key, (ex, ey) in TOWERS.items():
        x, y = engine_to_ours(ex, ey)
        tw = state["towers"][key]
        r = 2.0 if key.endswith("king") else 1.5
        cls = ("own" if key.startswith("own") else "opp") + ("" if tw["hp"] > 0 else " dead")
        s.append(f'<rect x="{x - r:.2f}" y="{y - r * 0.73:.2f}" width="{2 * r}" height="{2 * r * 0.73:.2f}" rx="0.3" '
                 f'class="tower {cls}"/><text x="{x:.2f}" y="{y + 0.35:.2f}" class="hp">{round(100 * tw["frac"])}%</text>')
    if mark:
        s.append(f'<circle cx="{mark[0]:.2f}" cy="{mark[1]:.2f}" r="0.9" class="mark"/>')
    for u in state["units"]:
        if not u["card"] or u["spell"]:
            continue
        s.append(f'<circle cx="{u["x"]:.2f}" cy="{u["y"]:.2f}" r="{0.55 if not u["building"] else 0.9}" '
                 f'class="u {u["team"]}{" fly" if u["flying"] else ""}"><title>{html.escape(u["card"])} {u["hp"]} hp</title></circle>'
                 f'<text x="{u["x"]:.2f}" y="{u["y"] - 0.75:.2f}" class="lbl">{html.escape(short(u["card"]))}</text>')
    s.append(f'<text x="0.4" y="{Y0 + 1.1}" class="t">{html.escape(title)}</text></svg>')
    return "".join(s)


def strip(states: list[dict], mark, label: str) -> str:
    return (f'<div class="strip"><div class="lab">{html.escape(label)}</div>'
            + "".join(arena_svg(st, mark if st["dt"] == 0 else None, f"+{st['dt']:g}s") for st in states) + "</div>")


def tap_map(game: str) -> tuple | None:
    """Tile -> % of the frame for this game's frames, fitted on the bot's own taps (exact to half a pixel)."""
    rows = [a for a in store.read("bot_actions", f"game_id = '{game}'") if a["taps"] and a["x_tiles"] is not None]
    if len(rows) < 3:
        return None
    t = np.array([json.loads(a["taps"])[2:4] for a in rows], float)
    xy = np.array([(a["x_tiles"], a["y_tiles"]) for a in rows], float)
    fx, fy = np.polyfit(xy[:, 0], t[:, 0], 1), np.polyfit(xy[:, 1], t[:, 1], 1)
    W, H = rows[0]["frame_w"], rows[0]["frame_h"]
    return lambda x, y: (100 * np.polyval(fx, x) / W, 100 * np.polyval(fy, y) / H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    res = json.loads(Path(f"data/cache/sim_search_{args.run}.json").read_text())
    margin = res["margin"]
    bpm = {(r["game_id"], r["k"]): r for r in store.read("bot_play_moments")}
    acts = {(a["game_id"], a["k"]): a for a in store.read("bot_actions")}
    out = Path(f"data/reviews/sim_{args.run}")
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in res["results"] if r["status"] == "ok" and r["search"] and r["search"] != r["cole"]]
    rows.sort(key=lambda r: (r["cole"], -(r["gap"] or 0)))
    maps, cards = {}, []
    for n, r in enumerate(rows):
        game, k = r["id"].rsplit(":", 1)
        a = acts.get((game, int(k)), {})
        link = bpm[(game, int(k))]
        m = store.read("moments", f"moment_id = '{link['replay_moment_id']}'")[0]
        tw = store.tower_fracs(m["match_id"], m["t"])
        best = r["best"]
        best_c = Candidate(None if best[0] == "wait" else sc.engine_id(best[0]), best[1], best[2], best[3])
        bot_c = Candidate(None if a.get("card") in (None, "wait") else sc.engine_id(a["card"]), a.get("x_tiles"),
                          a.get("y_tiles"), 0.0)
        st_best = trace(link["replay_moment_id"], best_c, tower_hp=tw, at=AT)
        st_bot = trace(link["replay_moment_id"], bot_c, tower_hp=tw, at=AT)
        if game not in maps:
            maps[game] = tap_map(game)
        f = maps[game]
        shot = a.get("shot") or f"{int(k):03d}_at.jpg"
        at = f"../../raw/live_games/{game}/{shot}"
        marks = ""
        if f and bot_c.x is not None:
            bx, by = f(bot_c.x, bot_c.y)
            marks += f'<div class="ring bot" style="left:{bx:.1f}%;top:{by:.1f}%"></div>'
        if f and best_c.x is not None:
            gx, gy = f(best_c.x, best_c.y)
            when = "now" if not best[3] else f"+{best[3]:g}s"
            marks += (f'<div class="ring best" style="left:{gx:.1f}%;top:{gy:.1f}%"></div>'
                      f'<div class="tag" style="left:{gx:.1f}%;top:{gy:.1f}%">{html.escape(best[0])} {when}</div>')
        best_txt = "wait" if best[0] == "wait" else f"{best[0]} {'now' if not best[3] else f'+{best[3]:g} s'}"
        cards.append(f"""<section class="card"><div class="top"><div class="frame"><img src="{at}" loading="lazy">{marks}</div>
<div class="info"><h2>{html.escape(r['id'])}</h2>
<p>Bot played <b>{html.escape(str(a.get('card')))}</b> <span class="key y">yellow</span></p>
<p>Search's best: <b>{html.escape(best_txt)}</b> <span class="key g">green</span><br>top 2 cards: {html.escape(', '.join(r['top2']))}</p>
<p>Cole: <b>{r['cole']}</b>{' (fix: <b>' + html.escape(str(r['fix'])) + '</b>)' if r['fix'] else ''}{' · blind' if r['blind'] else ''}<br>
<span class="note">{html.escape(r['note'])}</span></p>
<p class="note">Search says <b>{r['search']}</b>: the bot's play scores {r['gap']:.3f} below the best (a mistake above {margin:.3f}); win chance at the moment {r['p0']:.2f}.</p></div></div>
{strip(st_bot, (bot_c.x, bot_c.y) if bot_c.x is not None else None, "Sim: the bot's play")}
{strip(st_best, (best_c.x, best_c.y) if best_c.x is not None else None, "Sim: the search's best")}</section>""")
        print(f"{n + 1}/{len(rows)}", flush=True)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Search vs Cole</title><style>
:root{{--bg:#f6f7f9;--panel:#fff;--ink:#1b1f24;--muted:#5d6673;--line:#e2e5ea;--own:#2f6fe0;--opp:#d6453d;--ground:#e9e4d6;--river:#8cc4e8;--bridge:#b08a5a}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111418;--panel:#1a1e24;--ink:#e8eaed;--muted:#9aa3ad;--line:#2c323a;--ground:#2b2a26;--river:#2f5873;--bridge:#6e5636}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,sans-serif}}
header{{padding:12px 16px;border-bottom:1px solid var(--line);background:var(--panel)}}h1{{font-size:17px;margin:0 0 4px}}h2{{font-size:15px;margin:0 0 6px}}
main{{padding:16px;display:flex;flex-direction:column;gap:16px;max-width:1100px;margin:0 auto}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}}
.top{{display:flex;gap:14px;flex-wrap:wrap}}.frame{{position:relative;width:min(300px,100%)}}.frame img{{width:100%;display:block;border-radius:6px}}
.info{{flex:1;min-width:240px}}.info p{{margin:0 0 8px}}.note{{color:var(--muted)}}
.ring{{position:absolute;width:26px;height:26px;margin:-13px 0 0 -13px;border-radius:50%;border:3px solid}}
.ring.bot{{border-color:#ffd400}}.ring.best{{border-color:#18e06a;background:#18e06a44}}
.tag{{position:absolute;transform:translate(16px,-10px);background:#18e06a;color:#000;font-size:11px;padding:0 4px;border-radius:4px;white-space:nowrap}}
.key{{font-size:11px;padding:0 5px;border-radius:4px;color:#000}}.key.y{{background:#ffd400}}.key.g{{background:#18e06a}}
.strip{{display:flex;gap:4px;align-items:flex-start;margin-top:10px;overflow-x:auto}}.lab{{width:90px;flex:none;font-size:12px;color:var(--muted)}}
.arena{{width:120px;flex:none;border-radius:4px}}.ground{{fill:var(--ground)}}.river{{fill:var(--river)}}.bridge{{fill:var(--bridge)}}
.tower.own{{fill:var(--own);opacity:.55}}.tower.opp{{fill:var(--opp);opacity:.55}}.tower.dead{{opacity:.12}}
.hp{{font-size:0.9px;text-anchor:middle;fill:var(--ink)}}.u.own{{fill:var(--own)}}.u.opp{{fill:var(--opp)}}.u.fly{{fill-opacity:.35;stroke-width:.15}}
.u.fly.own{{stroke:var(--own)}}.u.fly.opp{{stroke:var(--opp)}}.lbl{{font-size:0.75px;text-anchor:middle;fill:var(--ink)}}.t{{font-size:1.1px;fill:var(--ink)}}
.mark{{fill:none;stroke:#18e06a;stroke-width:.2}}
</style></head><body><header><h1>Where the search and Cole disagree ({len(rows)} plays, run {html.escape(args.run)})</h1>
<div class="note">Frame: yellow ring = where the bot tapped, green = where the search would have played (and when). Strips: the simulator's
top-down board after each, at +0 / 2 / 4 / 7 / 10 s; blue ours, red theirs (faded = flying), towers with HP %. For each: is the
review right, or the sim?</div></header><main>{''.join(cards)}</main></body></html>"""
    (out / "index.html").write_text(page)
    print(f"{len(rows)} disagreements -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
