"""Cole's cheat sheet for the lab battles (docs/plan-sim-1to1.md P0.4): one printable page per test (config/lab/*.yaml)
with what to place, on which tile and when, and a board in HIS view (his side at the bottom) with the tile marked on the
play page's grid (scripts/situation_play.py: 18 x 34 cells, column 0 on the left, row 0 at his king's back edge).

  uv run python scripts/lab_sheet.py                      # -> data/lab/cheatsheet.html (local, gitignored)
  uv run python scripts/lab_sheet.py --cole-view flip     # if test L00 shows his screen is flipped, not turned

--cole-view: 'rotate' (default) if Cole's screen is the BlueStacks screen turned 180 degrees (his left lane is the bot's
right lane), 'flip' if only turned upside down (left stays left). Test L00 settles it.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

from bot77 import lab

OUT = Path("data/lab/cheatsheet.html")
CELL = 13


def board_svg(test: lab.LabTest, view: str) -> str:
    """The arena in Cole's view: his towers at the bottom; his tiles filled orange, the bot's outlined blue, numbered
    in time order."""
    W, H = lab.ENGINE_W * CELL, lab.ENGINE_H * CELL
    pad_l, pad_b = 22, 18
    sx = lambda x: pad_l + x * CELL  # noqa: E731  (cell x's left edge, his view)
    sy = lambda y: (lab.ENGINE_H - 1 - y) * CELL  # noqa: E731  (cell y's top edge; his row 0 at the bottom)
    out = [f'<svg viewBox="0 0 {W + pad_l + 4} {H + pad_b}" class="board" role="img" aria-label="board">']
    for y in range(lab.ENGINE_H):
        for x in range(lab.ENGINE_W):
            cls = "own" if y <= lab.OWN_MAX_Y else ("riv" if y in (16, 17) else "opp")
            if cls == "riv" and (x in (2, 3, 14, 15)):
                cls = "brg"
            out.append(f'<rect x="{sx(x)}" y="{sy(y)}" width="{CELL}" height="{CELL}" class="c {cls}"/>')
    # towers (same cells on both sides in his own frame): princess 3x3 at columns 3 / 14, row 6; king 4x4 at 8.5, 2.5
    for (cx, cy, s, who) in ((3, 6, 3, "me"), (14, 6, 3, "me"), (8.5, 2.5, 4, "me"),
                             (3, 27, 3, "bot"), (14, 27, 3, "bot"), (8.5, 30.5, 4, "bot")):
        out.append(f'<rect x="{sx(cx - s / 2 + 0.5)}" y="{sy(cy + s / 2 - 0.5)}" width="{s * CELL}" height="{s * CELL}" '
                   f'class="tw {who}" rx="2"/>')
    for i in range(0, lab.ENGINE_W, 3):
        out.append(f'<text x="{sx(i) + CELL / 2}" y="{H + 13}" class="ax">{i}</text>')
    for j in range(0, lab.ENGINE_H, 3):
        out.append(f'<text x="{pad_l - 4}" y="{sy(j) + CELL - 3}" class="ax" text-anchor="end">{j}</text>')
    order = sorted(test.placements, key=lambda p: (p.t, p.i))
    for n, p in enumerate(order, 1):
        x, y = lab.cole_tile(p.tile, view)
        cls = "mine" if p.side == "cole" else "theirs"
        out.append(f'<rect x="{sx(x) - 1}" y="{sy(y) - 1}" width="{CELL + 2}" height="{CELL + 2}" class="mk {cls}"/>')
        out.append(f'<text x="{sx(x) + CELL / 2}" y="{sy(y) + CELL - 3}" class="num {cls}">{n}</text>')
    out.append("</svg>")
    return "".join(out)


def page(test: lab.LabTest, view: str) -> str:
    rows = []
    for n, p in enumerate(sorted(test.placements, key=lambda p: (p.t, p.i)), 1):
        x, y = lab.cole_tile(p.tile, view)
        if p.side == "cole":
            where = lab.own_tile_words((x, y))
            who = "<b>YOU</b>"
        else:
            where = "bot's side" if not p.spell else "bot's spell"
            who = "bot"
        rows.append(f'<tr class="{"me" if p.side == "cole" else ""}"><td>{n}</td><td class="clk">{lab.clock_text(p.t)}</td>'
                    f'<td>{who}</td><td>{html.escape(p.card)}</td><td class="tile">({x}, {y})</td>'
                    f'<td>{html.escape(where)}{(" · " + html.escape(p.cue)) if p.cue else ""}</td></tr>')
    watch = "".join(f"<li>{html.escape(w)}</li>" for w in test.watch)
    return f"""<section class="page">
<div class="head"><div><div class="id">{test.id}</div><h2>{html.escape(test.name)}</h2>
<div class="mech">{html.escape(test.mechanic)}</div></div></div>
<div class="body"><div class="left">{board_svg(test, view)}
<div class="legend"><span class="sw mine"></span> you <span class="sw theirs"></span> bot · numbers = order · your side at the bottom</div></div>
<div class="right">
<table><thead><tr><th>#</th><th>clock</th><th>who</th><th>card</th><th>tile (you)</th><th>where</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class="note"><b>You:</b> {html.escape(test.cole_notes)}</p>
<p class="note"><b>Settles:</b> {html.escape(test.settles)}</p>
<ul class="watch">{watch}</ul>
<p class="small">Card not in hand at its time? Cycle first: play anything else at the back of the OTHER lane (spells into
the river corner), then place it as soon as you can. Late is fine: the replay records the real time. Keep the test lane clear.</p>
</div></div></section>"""


def front(tests: dict[str, lab.LabTest], view: str) -> str:
    cole_deck = next(iter(tests.values())).decks["cole"]
    rows = "".join(f"<tr><td>{t.id}</td><td>{html.escape(t.name)}</td><td>{html.escape(t.mechanic)}</td></tr>"
                   for t in tests.values())
    turn = ("turned 180 degrees: your LEFT is the bot's RIGHT" if view == "rotate"
            else "flipped top to bottom: your left is the bot's left")
    return f"""<section class="page">
<h1>Lab battles · cheat sheet</h1>
<p>Friendly battles, your phone vs the BlueStacks account. You play your half by this sheet; the bot plays its half by
script (<code>scripts/lab_play.py --test Lnn</code>, started before each battle). One battle per test, in this order.</p>
<p><b>Bring this deck:</b> {html.escape(", ".join(cole_deck))}</p>
<p><b>Board:</b> your side at the bottom, columns 0-17 left to right, rows 0-33 from your king's back edge (the play
page's grid). Your half is rows 0-15, as the engine and the play page have it; the river is rows 16-17. Tiles assume your screen is the bot's {turn}
(test L00 checks it).</p>
<p><b>Clock:</b> times are the battle clock as shown (3:00 counting down).</p>
<table><thead><tr><th>test</th><th>what</th><th>mechanic</th></tr></thead><tbody>{rows}</tbody></table>
</section>"""


CSS = """
:root { --ink:#1d2421; --muted:#5b665f; --line:#d5dbd7; --own:#e8f0e2; --opp:#e2e8f0; --riv:#9cc7e6; --brg:#b08d5f;
  --me:#e07a1f; --bot:#2f6fb5; --bg:#fff; }
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--ink); font:14px/1.45 system-ui, -apple-system, sans-serif; margin:0; padding:16px; }
h1 { font-size:1.5rem; margin:0 0 .5rem; } h2 { margin:.1rem 0; font-size:1.25rem; }
.page { max-width:900px; margin:0 auto 28px; page-break-after:always; break-after:page; }
.id { font:700 .85rem ui-monospace, monospace; color:var(--muted); } .mech { color:var(--muted); }
.body { display:flex; gap:18px; flex-wrap:wrap; margin-top:10px; }
.left { flex:0 0 auto; } .right { flex:1 1 320px; min-width:0; }
.board { width:270px; max-width:100%; height:auto; display:block; }
.c { stroke:#0002; stroke-width:.5; } .own { fill:var(--own); } .opp { fill:var(--opp); } .dead { fill:#eef0ee; }
.riv { fill:var(--riv); } .brg { fill:var(--brg); }
.tw { fill:#0000; stroke-width:2; } .tw.me { stroke:var(--me); } .tw.bot { stroke:var(--bot); }
.mk.mine { fill:var(--me); } .mk.theirs { fill:#fff; stroke:var(--bot); stroke-width:2.5; }
.num { font:700 10px ui-monospace, monospace; text-anchor:middle; } .num.mine { fill:#fff; } .num.theirs { fill:var(--bot); }
.ax { font:9px ui-monospace, monospace; fill:var(--muted); text-anchor:middle; }
.legend { font-size:.8rem; color:var(--muted); margin-top:4px; }
.sw { display:inline-block; width:10px; height:10px; vertical-align:middle; margin-left:6px; }
.sw.mine { background:var(--me); } .sw.theirs { border:2px solid var(--bot); }
table { border-collapse:collapse; width:100%; font-size:.9rem; }
th, td { text-align:left; padding:4px 6px; border-bottom:1px solid var(--line); vertical-align:top; }
tr.me td { background:#fdf0e3; font-weight:600; }
.clk, .tile { font-family:ui-monospace, monospace; white-space:nowrap; }
.note { margin:.6rem 0 .2rem; } .small { font-size:.8rem; color:var(--muted); } .watch { margin:.3rem 0; padding-left:1.1rem; color:var(--muted); }
@media print { body { padding:0; } .page { margin:0; } }
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cole-view", choices=["rotate", "flip"], default="rotate")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    tests = lab.load_tests()
    body = front(tests, a.cole_view) + "".join(page(t, a.cole_view) for t in tests.values())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
                   f"content='width=device-width, initial-scale=1'><title>Lab cheat sheet</title><style>{CSS}</style>"
                   f"</head><body>{body}</body></html>")
    print(f"{len(tests)} tests -> {out} (Cole's view: {a.cole_view})")


if __name__ == "__main__":
    main()
