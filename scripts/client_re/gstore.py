"""Loads/stores of absolute globals via ADRP + LDR/STR (e.g. who initialises a game-tag bit mask).

Usage: gstore.py ADDR [ADDR ...]
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from re_common import BASE, image, code_words, funcs, func_of

img = image(); W = code_words(img); starts, _ = funcs()
targets = [int(a, 16) - BASE for a in sys.argv[1:]]
for i, w in enumerate(W):
    if (w & 0x9f000000) != 0x90000000: continue
    rd = w & 31
    imm = (((w >> 5) & 0x7ffff) << 2) | ((w >> 29) & 3)
    if imm & (1 << 20): imm -= 1 << 21
    page = ((i * 4) & ~0xfff) + (imm << 12)
    for t in targets:
        if page != (t & ~0xfff): continue
        for j in range(1, 12):
            if i + j >= len(W): break
            w2 = W[i + j]
            if ((w2 >> 5) & 31) == rd and (w2 & 0x3b000000) == 0x39000000:
                size = w2 >> 30; opc = (w2 >> 22) & 3; off = ((w2 >> 10) & 0xfff) << size
                if page + off == t:
                    print(hex(t + BASE), 'STR' if opc == 0 else 'LDR', hex((i + j) * 4 + BASE),
                          'fn', hex(func_of(starts, i * 4) + BASE))
                break
