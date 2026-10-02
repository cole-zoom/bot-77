"""Hand-fitted layouts for wide-screen captures, where the HUD bands don't share one x transform and
auto layout can't fit them (decisions.md, 2026-10-01). Each band is a linear map of Alina's layout,
fitted from measured positions (template-matched chat / Next / elixir drop for the hand bar, the timer
box for the top HUD, colour-blob tower HP bars for the arena):
  hand  x' = hx + hs*x, y' = hy + hs*y      top-right  x' = rx + ts*x, y' = ty + ts*y
  arena x' = ax + axs*x, y' = ay + ays*y    top-left   x' = lx + ts*x
"""
import json
from pathlib import Path
from bot77.autolayout import TOP, HAND

FITS = {
    # Rolex 1342x1920: chat 216,1632; Next 222,1747; elixir 404,1855; own bars x 344/915 y 1182, opp y 260
    "Rolex": dict(size=(1342, 1920), hs=1.02, hx=124, hy=-469, ts=1.0, rx=266, lx=0, ty=-99,
                  axs=0.898, ax=186, ays=0.8874, ay=-80.8, videos=("Pn15Nh1xWoM", "XuM18Vfd-xc")),
    # Mohamed Light 1920x1080 stream overlay, game box x 78-880: chat 227,912; Next 229,976; elixir 330,1035;
    # timer 833,33; tower bars left edges x 300/612, opp top y 159
    "Mohamed Light": dict(size=(1920, 1080), hs=0.56, hx=176.6, hy=-241.6, ts=0.56, rx=275.2, lx=78, ty=-45.4,
                          axs=0.492, ax=212.5, ays=0.486, ay=-27.6, videos=("xONsnIlyu1E",)),
    # Adrian P 1920x1080 edit: game on the left (x 8-900), the hand bar re-scaled and pasted bottom-right under
    # the face cam: chat 1082,850; Next 1086,943; elixir 1230,1028; timer 840,39; own bars x 238/616 y 796
    "Adrian P": dict(size=(1920, 1080), hs=0.81, hx=1009, hy=-818.6, ts=0.675, rx=171.75, lx=8, ty=-55.5,
                     axs=0.594, ax=133.4, ays=0.587, ay=-39.3, videos=("ZXPr1gwF9ys",),
                     # the edit left the ability buttons in the game half; measured by hand (circle centre 745,965)
                     regions={"ability_right": [681, 901, 809, 1029], "ability_left": [99, 901, 227, 1029]}),
}
base = json.loads(Path("config/layouts/alina_portrait.json").read_text())
for creator, F in FITS.items():
    W, H = F["size"]
    AX = lambda x: round(F["ax"] + F["axs"] * x); AY = lambda y: round(F["ay"] + F["ays"] * y)
    regions = {}
    for n, (x0, y0, x1, y1) in base["regions"].items():
        if n in TOP:
            X = (lambda x: round(F["rx"] + F["ts"] * x)) if n in ("timer_label", "timer", "multiplier") else \
                AX if n == "opp_king_hp" else (lambda x: round(F["lx"] + F["ts"] * x))
            Y = lambda y: round(F["ty"] + F["ts"] * y)
        elif n in HAND:
            X = lambda x: round(F["hx"] + F["hs"] * x); Y = lambda y: round(F["hy"] + F["hs"] * y)
        else:
            X, Y = AX, AY
        regions[n] = F.get("regions", {}).get(n) or [max(0, X(x0)), max(0, Y(y0)), min(W, X(x1)), min(H, Y(y1))]
    kx0, ky0, kx1, ky1 = base["katacr_arena"]
    for v in F["videos"]:
        d = {"layout_id": f"auto_{v}", "description": f"{creator} — wide capture, fitted by hand per band",
             "frame_size": [W, H], "regions": regions, "hand": {"ui_scale": F["hs"]},
             "katacr_arena": [AX(kx0), AY(ky0), AX(kx1), AY(ky1)],
             "notes": {"manual": {k: v for k, v in F.items() if k not in ("videos", "size")}, "method": __doc__}}
        Path(f"config/layouts/auto_{v}.json").write_text(json.dumps(d, indent=2))
        print("wrote", d["layout_id"])
