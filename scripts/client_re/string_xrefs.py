"""Code sites that take the address of an exact C string (ADRP + ADD pairs), e.g. a field name's loader.

Usage: string_xrefs.py NAME [NAME ...]
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from re_common import BASE, image, code_words

img = image(); W = code_words(img)
targets = {}
for name in sys.argv[1:]:
    for m in re.finditer(b'\x00' + re.escape(name.encode()) + b'\x00', img):
        targets[m.start() + 1] = name
for i, w in enumerate(W):
    if (w & 0x9f000000) != 0x90000000:
        continue
    rd = w & 31
    imm = (((w >> 5) & 0x7ffff) << 2) | ((w >> 29) & 3)
    if imm & (1 << 20): imm -= 1 << 21
    page = ((i * 4) & ~0xfff) + (imm << 12)
    for j in range(1, 10):
        if i + j >= len(W): break
        w2 = W[i + j]
        if (w2 & 0xff800000) == 0x91000000 and ((w2 >> 5) & 31) == rd:
            imm12 = (w2 >> 10) & 0xfff
            if (w2 >> 22) & 1: imm12 <<= 12
            if page + imm12 in targets: print(targets[page + imm12], hex(BASE + i * 4))
            break
