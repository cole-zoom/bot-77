"""Card levels from a Battle Log entry image (replay_runner.py's replay_NN_entry.png), 2026-11-15.

The entry shows both decks, 4x2 tiles a side (left = the account that recorded it, which is the replay's bottom
player; right = the opponent), each tile labelled "lvl.N". The sim needs these levels: the engine's cards are level
11 and the bot's games are level 5-7 (decisions.md, 2026-11-15).

- Tiles: the dark tile borders, as connected components (works on BlueStacks and Galaxy A16 entries, whose tiles
  differ by a few pixels in size and place).
- Level: the label's glyphs are bright with a dark outline. The fixed "lvl." prefix is found by template matching;
  the glyphs right of it are matched to digit templates cut from 506 entries (data/cards/level_digits.npz; digits
  1-7 seen; no 8, 9, 0 in our games yet: a tile whose glyphs don't read cleanly gets level None, never a guess).
  Apple's OCR read this font badly (7 as "Z", most labels missed), hence templates.
- Card: the side's deck is known from the hand reading (matches.deck_names), so each tile is assigned to one of
  those cards (Hungarian on art similarity against the cards table's art). Whole-catalogue matching alone was poor.
"""

from __future__ import annotations

from pathlib import Path

import collections
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

DIGITS = Path(__file__).resolve().parents[3] / "data/cards/level_digits.npz"
TW = 100  # tiles are normalised to this width before the label is read
MIN_DIGIT_SCORE = 0.80  # correlation with the digit template; below it the glyph is art, not a digit


def find_tiles(im: np.ndarray) -> list[tuple[int, int, int, int]]:
    """The 16 deck tiles (x, y, w, h): row by row, left to right (first 4 of each row = left player)."""
    v = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 2]
    n, _, st, _ = cv2.connectedComponentsWithStats((v < 80).astype(np.uint8), 8)
    out = [tuple(int(a) for a in st[i, :4]) for i in range(1, n) if 85 <= st[i, 2] <= 130 and 105 <= st[i, 3] <= 145]
    return sorted(out, key=lambda b: (b[1] // 60, b[0]))


def _label_mask(im: np.ndarray, box: tuple[int, int, int, int], enclosed: bool = False) -> np.ndarray:
    """Bright label pixels. enclosed: only bright pixels walled off from the strip's border by the dark outline (for
    labels on light art, Arrows' sky, where plain brightness merges text and background)."""
    x, y, w, h = box
    t = cv2.resize(im[y:y + h, x:x + w], (TW, round(TW * h / w)), interpolation=cv2.INTER_AREA)
    b = t[int((0.64 if enclosed else 0.70) * t.shape[0]): t.shape[0] - 2, 2: TW - 2]
    bright = (cv2.cvtColor(b, cv2.COLOR_BGR2HSV)[:, :, 2] > 165).astype(np.uint8)
    if not enclosed:
        return bright
    n, lab = cv2.connectedComponents(bright, connectivity=4)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    return (bright & ~np.isin(lab, list(edge))).astype(np.uint8)


def _glyphs(m: np.ndarray, prefix: np.ndarray) -> list[np.ndarray]:
    r = cv2.matchTemplate(m.astype(np.float32), prefix.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, _, _, (px, py) = cv2.minMaxLoc(r)
    ph, pw = prefix.shape
    sub = m[max(py - 1, 0): py + ph + 1, px + pw - 1:]
    n, lab, st, _ = cv2.connectedComponentsWithStats(sub, 8)
    cs = sorted(tuple(int(a) for a in st[i]) + (i,) for i in range(1, n) if st[i, 4] >= 4)
    gl = [(gx, cv2.resize((lab[gy:gy + gh, gx:gx + gw] == i).astype(np.uint8) * 255, (12, 18),
                          interpolation=cv2.INTER_AREA))
          for gx, gy, gw, gh, _, i in cs if gh >= 0.6 * ph and gw <= 0.9 * ph]
    out: list[tuple[int, np.ndarray]] = []
    for gx, g in sorted(gl, key=lambda t: t[0]):  # contiguous from the prefix; stray art further right is dropped
        if (not out and gx > 6) or (out and gx - out[-1][0] > 16):
            break
        out.append((gx, g))
    return [g for _, g in out]


def _z(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32).ravel() / 255.0
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-9)


class LevelReader:
    def __init__(self, cards: list[dict]):
        d = np.load(DIGITS)
        self.prefix, self.tm, self.digits = d["prefix"], d["templates"], [int(x) for x in d["digits"]]
        self.art: dict[str, np.ndarray] = {}
        for c in cards:
            fs = [self._art_feat(_decode(c[f"art_{f}"])) for f in ("normal", "evo", "hero") if c.get(f"art_{f}")]
            if fs:
                self.art[c["name"]] = np.array(fs)

    @staticmethod
    def _art_feat(im: np.ndarray) -> np.ndarray:
        h, w = im.shape[:2]
        p = im[int(0.06 * h):int(0.66 * h), int(0.08 * w):int(0.92 * w)]
        p = cv2.resize(p, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        p = (p - p.mean((0, 1))) / (p.std((0, 1)) + 1e-6)
        return p.ravel() / np.sqrt(p.size)

    def level(self, im: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int | None, float]:
        """(level, worst digit score); level None when any glyph isn't a clean digit or there are none."""
        gl = self._glyphs_scored(im, box)
        if not gl or len(gl) > 2 or min(s for _, s in gl) < MIN_DIGIT_SCORE:
            return None, min((s for _, s in gl), default=0.0)
        lv = int("".join(str(d) for d, _ in gl))
        return (lv if 1 <= lv <= 20 else None), min(s for _, s in gl)

    def _glyphs_scored(self, im, box) -> list[tuple[int, float]]:
        best: list[tuple[int, float]] = []
        for enclosed in (False, True):
            out = self._glyphs_scored_mask(_label_mask(im, box, enclosed))
            if out and min(sc for _, sc in out) >= MIN_DIGIT_SCORE:
                return out
            best = best or out
        return best

    def _glyphs_scored_mask(self, m) -> list[tuple[int, float]]:
        out = []
        for g in _glyphs(m, self.prefix):
            s = self.tm @ _z(g)
            out.append((self.digits[int(s.argmax())], float(s.max())))
        # a trailing art fragment that isn't a digit after a clean first digit: keep the clean part
        while len(out) > 1 and out[-1][1] < MIN_DIGIT_SCORE:
            out.pop()
        return out

    def read(self, im: np.ndarray, decks: dict[str, list[str]]) -> list[dict] | None:
        """decks: {"bottom": [...], "top": [...]} card names (the left / right player's). None if the tiles aren't
        all found. One row per tile: side, card (None if the side's deck lists fewer cards than tiles), level."""
        T = find_tiles(im)
        if len(T) != 16:
            return None
        rows = []
        for side, cols in (("bottom", range(0, 4)), ("top", range(4, 8))):
            boxes = [T[r * 8 + c] for r in (0, 1) for c in cols]
            deck = [c for c in decks.get(side, []) if c in self.art]
            cards: list[str | None] = [None] * len(boxes)
            score = [0.0] * len(boxes)
            if deck:
                tf = [self._art_feat(im[y:y + h, x:x + w]) for x, y, w, h in boxes]
                S = np.array([[float((self.art[c] @ f).max()) for c in deck] for f in tf])
                ri, ci = linear_sum_assignment(-S)
                for i, j in zip(ri, ci):
                    cards[i], score[i] = deck[j], S[i, j]
            for b, card, sc in zip(boxes, cards, score):
                lv, ds = self.level(im, b)
                rows.append({"side": side, "card": card, "level": lv, "art_score": round(sc, 3),
                             "digit_score": round(ds, 3)})
        return rows


def _decode(png: bytes) -> np.ndarray:
    im = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    if im.shape[2] == 4:
        a = im[:, :, 3:] / 255.0
        im = (im[:, :, :3] * a + 128 * (1 - a)).astype(np.uint8)
    return im


def name_clusters(cluster_entries: dict[int, list], entry_decks: dict, purity: float = 0.80) -> dict[int, tuple]:
    """Name tile clusters by the hand-read decks of the entries they occur in, surest first, each name explaining
    away: once a cluster is named, another tile of the same entry can't be that card (2026-11-16: the Giant's tiles,
    always beside a Mini P.E.K.K.A in the bot's deck, were named Mini P.E.K.K.A 94% vs Giant 90%, so no Giant ever got
    a level and every Giant played at 11).
    cluster_entries: cluster -> entry keys of its tiles (one per tile); entry_decks: entry -> deck card names.
    -> {cluster: (name, purity, tiles with a known deck)}."""
    taken: dict = collections.defaultdict(set)  # entry -> names claimed by its named tiles
    names: dict[int, tuple] = {}
    while True:
        best = None
        for j, es in cluster_entries.items():
            if j in names:
                continue
            known = [e for e in es if e in entry_decks]
            c = collections.Counter(n for e in known for n in set(entry_decks[e]) - taken[e])
            if not c:
                continue
            (name, n), *rest = c.most_common(2)
            p = n / len(known)
            margin = p - (rest[0][1] / len(known) if rest else 0.0)
            if p >= purity and (best is None or (p, margin) > best[0]):
                best = ((p, margin), j, name, len(known))
        if best is None:
            return names
        (p, _), j, name, n = best
        names[j] = (name, p, n)
        for e in cluster_entries[j]:
            taken[e].add(name)
