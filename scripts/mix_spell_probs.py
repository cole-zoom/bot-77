"""Mix two spell classifiers' per-match probabilities (scripts/katacr/infer_spell.py outputs) into one directory.

Default (2026-10-15): spell_v2 + spell_v3 averaged, except Goblin Curse, Vines and the ability classes, which come from
v2 alone (v3, trained with Cole's recording clips, under-calls them on POV recordings). Classes only one model knows
come from that model. Tuned on held-back replays + dev; see decisions.md.
"""
import argparse
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--a", default="data/models/spell_v2/spell_probs")
ap.add_argument("--b", default="data/models/spell_v3/spell_probs")
ap.add_argument("--out", default="data/models/spell_mix/spell_probs")
ap.add_argument("--a-only", default="Goblin Curse,Vines,*ability", help="classes taken from --a alone (* = suffix match)")
a = ap.parse_args()

a_only = a.a_only.split(",") if a.a_only else []
take_a = lambda c: any(c.endswith(x[1:]) if x.startswith("*") else c == x for x in a_only)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
n = 0
for fb in sorted(Path(a.b).glob("*.npz")):
    fa = Path(a.a) / fb.name
    zb = np.load(fb)
    if not fa.exists():
        np.savez(out / fb.name, t=zb["t"], p=zb["p"], classes=zb["classes"]); n += 1
        continue
    za = np.load(fa)
    if za["t"].shape != zb["t"].shape or not np.allclose(za["t"], zb["t"]):
        raise SystemExit(f"window times differ: {fb.name}")
    ca, cb = [str(c) for c in za["classes"]], [str(c) for c in zb["classes"]]
    p = zb["p"].copy()
    for i, c in enumerate(cb):
        if c in ca:
            pa = za["p"][:, ca.index(c)]
            p[:, i] = pa if take_a(c) else (pa + zb["p"][:, i]) / 2
    np.savez(out / fb.name, t=zb["t"], p=p.astype(np.float32), classes=zb["classes"]); n += 1
print(n, "matches ->", out)
