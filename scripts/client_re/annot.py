"""Replace UNK_/DAT_ addresses in Ghidra output with the C string they point to. Usage: annot.py FILE > OUT"""
import os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from re_common import BASE, image

img = image()


def s(m):
    a = int(m.group(2), 16) - BASE
    if 0 <= a < len(img):
        e = img.find(b'\x00', a, a + 80); t = img[a:e]
        if e > a and all(32 <= c < 127 for c in t): return '"' + t.decode() + '"'
    return m.group(0)


for line in open(sys.argv[1]):
    sys.stdout.write(re.sub(r'&?(UNK|DAT|s)_([0-9a-f]{10})', s, line))
