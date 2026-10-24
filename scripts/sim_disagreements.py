"""S3 failed its gate: the plays where the search and Cole disagree, with the frames, for Cole to say whether the sim or
the review is wrong (docs/simulator.md S3). Read-only page, local (data/reviews is gitignored).

  uv run python scripts/sim_disagreements.py --run s3_v1 -> data/reviews/sim_<run>/index.html
"""
import argparse
import html
import json
from pathlib import Path

import lancedb

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
args = ap.parse_args()
res = json.loads(Path(f"data/cache/sim_search_{args.run}.json").read_text())
margin = res["margin"]
db = lancedb.connect("data/lancedb")
acts = {(a["game_id"], a["k"]): a for a in db.open_table("bot_actions").search().limit(10**6)
        .select(["game_id", "k", "shot", "card", "x_tiles", "y_tiles"]).to_arrow().to_pylist()}
out = Path(f"data/reviews/sim_{args.run}")
out.mkdir(parents=True, exist_ok=True)
rows = [r for r in res["results"] if r["status"] == "ok" and r["search"] and r["search"] != r["cole"]]
rows.sort(key=lambda r: (r["cole"], -(r["gap"] or 0)))
cards = []
for r in rows:
    game, k = r["id"].rsplit(":", 1)
    a = acts.get((game, int(k)), {})
    shot = a.get("shot") or f"{int(k):03d}_at.jpg"
    at = f"../../raw/live_games/{game}/{shot}"
    after = at.replace("_at.jpg", "_after.jpg")
    best = r["best"]
    best_txt = "wait" if best[0] == "wait" else f"{best[0]} at ({best[1]:.1f}, {best[2]:.1f}), {'now' if not best[3] else f'+{best[3]:g} s'}"
    cards.append(f"""<div class="card"><div class="imgs"><img src="{at}" loading="lazy"><img src="{after}" loading="lazy"></div>
<div><b>{html.escape(r['id'])}</b> · bot played <b>{html.escape(str(r['bot_card']))}</b></div>
<div class="cole">Cole: <b>{r['cole']}</b>{' (fix: <b>' + html.escape(str(r['fix'])) + '</b>)' if r['fix'] else ''}{' · blind' if r['blind'] else ''}<br>
<span class="note">{html.escape(r['note'])}</span></div>
<div class="sim">Search: <b>{r['search']}</b> (bot's play {r['gap']:.3f} below the best; bad above {margin:.3f})<br>
best: {html.escape(best_txt)} · top 2 cards: {html.escape(', '.join(r['top2']))} · win chance at the moment {r['p0']:.2f}</div></div>""")
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Search vs Cole</title><style>
:root{{--bg:#f6f7f9;--panel:#fff;--ink:#1b1f24;--muted:#5d6673;--line:#e2e5ea}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111418;--panel:#1a1e24;--ink:#e8eaed;--muted:#9aa3ad;--line:#2c323a}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,sans-serif}}
header{{padding:12px 16px;border-bottom:1px solid var(--line);background:var(--panel)}}h1{{font-size:17px;margin:0 0 4px}}
main{{padding:16px;display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:14px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px;display:flex;flex-direction:column;gap:6px}}
.imgs{{display:flex;gap:6px}}.imgs img{{width:50%;border-radius:6px}}.note{{color:var(--muted)}}.sim{{border-top:1px solid var(--line);padding-top:6px}}
</style></head><body><header><h1>Where the search and Cole disagree ({len(rows)} plays, run {html.escape(args.run)})</h1>
<div class="note">Frame at the bot's play and 2 s later. For each: is the review right, or the sim? Spots are our tiles (x 0-18, y down; our river 14.9).</div></header>
<main>{''.join(cards)}</main></body></html>"""
(out / "index.html").write_text(page)
print(f"{len(rows)} disagreements -> {out / 'index.html'}")
