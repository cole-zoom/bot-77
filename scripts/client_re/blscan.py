"""Direct callers (BL/B) of the given function addresses. Usage: blscan.py ADDR [ADDR ...]"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from re_common import BASE, image, code_words, funcs, func_of

img = image(); W = code_words(img); starts, ranges = funcs()
targets = {int(a, 16) - BASE: a for a in sys.argv[1:]}
res = {}
for i, w in enumerate(W):
    if (w >> 26) in (0x25, 0x05):
        imm = w & 0x3ffffff
        if imm & (1 << 25): imm -= 1 << 26
        t = i * 4 + imm * 4
        if t in targets:
            f = func_of(starts, i * 4)
            res.setdefault(targets[t], set()).add((hex(f + BASE), ranges[starts.index(f)]))
for t, v in res.items(): print(t, len(v), sorted(v)[:30])
