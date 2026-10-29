"""Functions that load given struct offsets (LDR/LDRB/STR unsigned-offset forms), ranked by how many they hit.

Usage: ldscan.py SPEC [SPEC ...]   SPEC = w:OFF | x:OFF | b:OFF | ws:OFF | bs:OFF (hex OFF; s = store)
       ldscan.py --each SPEC [--max SIZEHEX]   list every function with that access (optionally small ones only)
"""
import collections, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from re_common import BASE, image, code_words, funcs, func_of

PAT = {'w': (0xb9400000, 4), 'x': (0xf9400000, 8), 'b': (0x39400000, 1), 'ws': (0xb9000000, 4), 'bs': (0x39000000, 1)}


def pattern(spec):
    kind, off = spec.split(':'); base, scale = PAT[kind]
    return base | ((int(off, 16) // scale) << 10)


img = image(); W = code_words(img); starts, ranges = funcs()
args = sys.argv[1:]
if args and args[0] == '--each':
    p = pattern(args[1]); mx = int(args[args.index('--max') + 1], 16) if '--max' in args else 1 << 40
    seen = collections.Counter()
    for i, w in enumerate(W):
        if (w & 0xfffffc00) == p:
            seen[func_of(starts, i * 4)] += 1
    for f, c in sorted(seen.items()):
        k = starts.index(f)
        if ranges[k] <= mx: print(hex(f + BASE), 'size', hex(ranges[k]), 'hits', c)
    sys.exit()
pats = {s: pattern(s) for s in args}
hits = collections.defaultdict(set)
for i, w in enumerate(W):
    m = w & 0xfffffc00
    for s, p in pats.items():
        if m == p: hits[func_of(starts, i * 4)].add(s)
for f, v in sorted(hits.items(), key=lambda kv: -len(kv[1]))[:40]:
    if len(v) >= max(2, len(args) - 2): print(hex(f + BASE), len(v), sorted(v))
