"""Rebuild a loadable ELF from the memory dump: each region at its load address, PT_LOADs with offset == vaddr.

Usage: python3 -I scripts/client_re/rebuild_image.py   (writes data/work/ghidra/libg_rebuilt.so)
"""
import glob, os, struct, sys
sys.path.insert(0, os.path.dirname(__file__))
from re_common import DUMP, IMG_PATH, BASE

regions = []
for f in glob.glob(os.path.join(DUMP, '*.bin')):
    a = int(os.path.basename(f).split('_')[0], 16) - BASE
    regions.append((a, open(f, 'rb').read()))
size = max(a + len(b) for a, b in regions)
img = bytearray(size)
for a, b in regions:
    img[a:a + len(b)] = b
phoff = struct.unpack_from('<Q', img, 32)[0]
phnum = struct.unpack_from('<H', img, 56)[0]
for i in range(phnum):
    o = phoff + i * 56
    t, fl, off, va, pa, fs, ms, al = struct.unpack_from('<IIQQQQQQ', img, o)
    if t in (1, 2, 4, 0x6474e550, 0x6474e552):
        fs2 = min(ms, size - va) if t == 1 else fs
        struct.pack_into('<IIQQQQQQ', img, o, t, fl, va, va, pa, fs2, ms, al)
struct.pack_into('<Q', img, 40, 0)          # no section headers (they are not mapped)
struct.pack_into('<HHH', img, 60, 0, 0, 0)
os.makedirs(os.path.dirname(IMG_PATH), exist_ok=True)
open(IMG_PATH, 'wb').write(img)
print('wrote', IMG_PATH, hex(size), 'bytes; base', hex(BASE))
