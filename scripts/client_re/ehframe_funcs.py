"""Function starts and sizes from .eh_frame_hdr (the binary is stripped; this gives every function boundary).

Usage: ehframe_funcs.py [ADDR ...]  -> writes funcs.json, prints the function containing each ADDR.
"""
import json, os, struct, sys, bisect
sys.path.insert(0, os.path.dirname(__file__))
from re_common import BASE, FUNCS_PATH, image

img = image()
phoff = struct.unpack_from('<Q', img, 32)[0]; phnum = struct.unpack_from('<H', img, 56)[0]
h = None
for i in range(phnum):
    t, fl, off, va = struct.unpack_from('<IIQQ', img, phoff + i * 56)
    if t == 0x6474e550:
        h = va
assert img[h + 1] == 0x1b and img[h + 2] == 0x03 and img[h + 3] == 0x3b, 'unexpected eh_frame_hdr encodings'
cnt = struct.unpack_from('<I', img, h + 8)[0]
starts, ranges = [], []
for i in range(cnt):
    loc, fde = struct.unpack_from('<ii', img, h + 12 + i * 8)
    starts.append(loc + h)
    ranges.append(struct.unpack_from('<I', img, fde + h + 12)[0])
json.dump({'starts': starts, 'ranges': ranges}, open(FUNCS_PATH, 'w'))
print(cnt, 'functions')
for a in sys.argv[1:]:
    x = int(a, 16) - BASE; k = bisect.bisect_right(starts, x) - 1
    print(a, '->', hex(starts[k] + BASE), 'size', hex(ranges[k]))
