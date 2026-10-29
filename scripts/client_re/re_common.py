"""Shared paths and constants for the client-library analysis helpers (docs/client-code-notes.md).

Everything these helpers read or write stays under data/ (gitignored): the dump, the rebuilt image, the Ghidra
projects and any decompiler output. Override the locations with CR_RE_DIR (work dir) and CR_DUMP_DIR.
"""
import json, os, struct, bisect

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
WORK = os.environ.get('CR_RE_DIR', os.path.join(REPO, 'data', 'work', 'ghidra'))
DUMP = os.environ.get('CR_DUMP_DIR', os.path.join(REPO, 'data', 'external', 'cr_client', 'libg_dump'))
IMG_PATH = os.path.join(WORK, 'libg_rebuilt.so')
FUNCS_PATH = os.path.join(WORK, 'funcs.json')


def load_base():
    """Load address of libg.so: the first line of the dump's maps.txt."""
    with open(os.path.join(DUMP, 'maps.txt')) as f:
        return int(f.readline().split('-')[0], 16)


BASE = int(os.environ['CR_BASE'], 16) if 'CR_BASE' in os.environ else load_base()


def image():
    return open(IMG_PATH, 'rb').read()


def code_end(img):
    """End of the first (r-x) PT_LOAD segment, from the image's own program headers."""
    phoff = struct.unpack_from('<Q', img, 32)[0]
    phnum = struct.unpack_from('<H', img, 56)[0]
    for i in range(phnum):
        t, fl, off, va, pa, fs, ms, al = struct.unpack_from('<IIQQQQQQ', img, phoff + i * 56)
        if t == 1 and fl & 1:
            return va + fs
    raise ValueError('no executable PT_LOAD')


def code_words(img):
    n = code_end(img) // 4
    return struct.unpack_from('<%dI' % n, img, 0)


def funcs():
    F = json.load(open(FUNCS_PATH))
    return F['starts'], F['ranges']


def func_of(starts, off):
    return starts[bisect.bisect_right(starts, off) - 1]
