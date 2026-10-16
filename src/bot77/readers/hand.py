"""Read the POV player's hand: which card is in each of the 4 slots and the next slot,
plus each card's state (greyed out, selected, which art form)."""

from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

import cv2
import numpy as np

from bot77.layout import Layout

SLOTS = ("slot1", "slot2", "slot3", "slot4")

# Fraction of the API art used for matching: the middle of the picture, clear of the
# card frame and of the elixir drop that the game draws over the bottom of the card.
ART_PATCH = (0.15, 0.85, 0.15, 0.62)  # x0, x1, y0, y1

MIN_SCORE = 0.45  # best match below this -> not a confident card read
MIN_MARGIN = 0.20  # best minus runner-up (a different card) below this -> ambiguous
GREY_SATURATION = 40  # mean HSV saturation of the matched art below this -> greyed out
LCN_SIGMA = 5.0  # px; local contrast normalisation window

# Sizes at a canonical UI scale: Alina's native 1080x2340 capture. A layout's `ui_scale`
# says how big the game UI is in its source frames relative to that; templates are built
# at that size (downscaling the art beats upscaling small crops).
CANONICAL = {
    "slot_art_scale": 0.73,  # API art -> hand slot
    "next_art_scales": (0.31, 0.33, 0.35),  # the next card is tiny, so search a few sizes
    "evo_art_scale": 0.80,  # charged evo / hero cards are drawn ~10% larger
    "art_top_in_slot": 62,  # px from the slot region's top to an unselected card's art
    "selected_raise_px": 31,
}


def normalise(gray: np.ndarray, sigma: float = LCN_SIGMA) -> np.ndarray:
    """Local contrast normalisation. Unaffordable cards are drawn grey with a slanted
    wipe: one side washed out, the other plain grey. Normalising each small neighbourhood
    makes matching indifferent to that, and to the game's own desaturation."""
    g = gray.astype(np.float32)
    mean = cv2.GaussianBlur(g, (0, 0), sigma)
    var = cv2.GaussianBlur(g * g, (0, 0), sigma) - mean * mean
    return (g - mean) / (np.sqrt(np.maximum(var, 0)) + 8.0)


@dataclass(frozen=True)
class Template:
    card_id: int
    name: str
    form: str  # normal | evo | hero
    slot: np.ndarray  # normalised patch at hand-slot scale
    next: tuple[np.ndarray, ...]  # normalised patches at next-slot scales


@dataclass(frozen=True)
class SlotRead:
    state: str  # card | empty | unknown
    card_id: int | None = None
    name: str | None = None
    form: str | None = None
    score: float = 0.0
    margin: float = 0.0
    greyed: bool | None = None
    selected: bool | None = None


ART_OVERRIDES = Path("data/cards/art_overrides")
EVO_ART_SCALES_EXTRA = (0.84, 0.88, 0.92)  # besides CANONICAL['evo_art_scale'] (0.80)


def _decode_art(png: bytes) -> np.ndarray:
    im = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    if im.shape[2] == 4:
        alpha = im[:, :, 3:] / 255.0
        im = (im[:, :, :3] * alpha + 128 * (1 - alpha)).astype(np.uint8)
    return im


def _art_patch(im: np.ndarray) -> np.ndarray:
    h, w = im.shape[:2]
    x0, x1, y0, y1 = ART_PATCH
    return cv2.cvtColor(im[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)], cv2.COLOR_BGR2GRAY)


def load_templates(cards: list[dict], ui_scale: float = 1.0, next_scales: tuple[float, ...] | None = None) -> list[Template]:
    """`cards` are rows of the LanceDB `cards` table (need id, name, art_normal/evo/hero). `next_scales` overrides
    CANONICAL['next_art_scales'] for a capture whose Next card is drawn at a different size relative to the hand
    (the replay viewer: 0.35-0.39, decisions.md 2026-10-12)."""
    c, sigma = dict(CANONICAL), LCN_SIGMA * ui_scale
    if next_scales:
        c["next_art_scales"] = tuple(next_scales)
    out = []
    for card in cards:
        for form in ("normal", "evo", "hero"):
            png = card.get(f"art_{form}")
            if not png:
                continue
            patch = _art_patch(_decode_art(png))
            # evo/hero cards are drawn larger than normal ones, and larger than the original calibration in newer
            # captures (Cole's 2026-10-06 recording: best match at 0.88, not 0.80, so every evo/hero card read as
            # nothing and its plays never reached the event log). Try several sizes; the best score per card wins.
            for evo_scale in ((c["evo_art_scale"], *EVO_ART_SCALES_EXTRA) if form != "normal" else (None,)):
                grow = evo_scale / c["slot_art_scale"] if evo_scale else 1.0
                resize = lambda s: normalise(
                    cv2.resize(patch, None, fx=s * grow * ui_scale, fy=s * grow * ui_scale, interpolation=cv2.INTER_AREA), sigma)
                out.append(Template(card["id"], card["name"], form,
                                    resize(c["slot_art_scale"]), tuple(resize(s) for s in c["next_art_scales"])))
    # In-game art overrides: art patches cut from real frames, for cards whose API icon no longer matches what the
    # hand shows (evo Skeleton Army shows the Skeleton General since an update; its API evo icon is the old skulls).
    # Saved at slot scale for an evo/hero card drawn at OVERRIDE_DRAWN_SCALE (see scripts in decisions.md, 2026-10-07).
    names = {card["name"]: card["id"] for card in cards}
    for f in sorted(ART_OVERRIDES.glob("*__*.png")):
        name, form = f.stem.split("__")
        if name not in names:
            continue
        patch = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        rs = lambda k: normalise(cv2.resize(patch, None, fx=k * ui_scale, fy=k * ui_scale, interpolation=cv2.INTER_AREA), sigma)
        out.append(Template(names[name], name, form, rs(1.0), tuple(rs(n / c["slot_art_scale"]) for n in c["next_art_scales"])))
    return out


class HandReader:
    def __init__(self, layout: Layout, templates: list[Template]):
        self.layout = layout
        self.templates = templates
        self.ui_scale = layout.hand.get("ui_scale", 1.0)

    @classmethod
    def from_cards(cls, layout: Layout, cards: list[dict]) -> HandReader:
        return cls(layout, load_templates(cards, layout.hand.get("ui_scale", 1.0), layout.hand.get("next_art_scales")))

    def restrict(self, card_ids: set[int]) -> HandReader:
        """A reader that only considers the given cards (e.g. the inferred deck)."""
        return HandReader(self.layout, [t for t in self.templates if t.card_id in card_ids])

    def read(self, frame: np.ndarray) -> dict[str, SlotRead]:
        reads = {slot: self._read_slot(frame, slot, "slot") for slot in SLOTS}
        reads["next"] = self._read_slot(frame, "next", "next")
        return reads

    def _read_slot(self, frame: np.ndarray, region: str, size: str) -> SlotRead:
        s = self.layout.scale(frame.shape[1])
        crop = self.layout.crop(frame, region)
        if abs(s - 1.0) > 1e-3:  # same layout at a different resolution
            crop = cv2.resize(crop, None, fx=1 / s, fy=1 / s, interpolation=cv2.INTER_AREA)
        gray = normalise(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), LCN_SIGMA * self.ui_scale)

        best: dict[int, tuple[float, Template, tuple[int, int], np.ndarray]] = {}
        for t in self.templates:
            for patch in (t.slot,) if size == "slot" else t.next:
                if patch.shape[0] > gray.shape[0] or patch.shape[1] > gray.shape[1]:
                    continue
                _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(gray, patch, cv2.TM_CCOEFF_NORMED))
                if t.card_id not in best or score > best[t.card_id][0]:
                    best[t.card_id] = (score, t, loc, patch)
        if not best:
            return SlotRead("unknown")

        ranked = sorted(best.values(), key=lambda b: b[0], reverse=True)
        score, t, (x, y), patch = ranked[0]
        margin = score - ranked[1][0] if len(ranked) > 1 else score
        if score < MIN_SCORE or margin < MIN_MARGIN:
            state = "empty" if _looks_empty(crop) else "unknown"
            return SlotRead(state, score=score, margin=margin)

        matched = crop[y:y + patch.shape[0], x:x + patch.shape[1]]
        saturation = cv2.cvtColor(matched, cv2.COLOR_BGR2HSV)[:, :, 1].mean()
        selected = None
        if size == "slot" and t.form == "normal":
            selected = y < (CANONICAL["art_top_in_slot"] - CANONICAL["selected_raise_px"] / 2) * self.ui_scale
        return SlotRead("card", t.card_id, t.name, t.form, score, margin,
                        greyed=bool(saturation < GREY_SATURATION), selected=selected)


def _looks_empty(crop: np.ndarray) -> bool:
    """An empty slot is the flat blue placeholder with a faint crown."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, w = crop.shape[:2]
    centre = hsv[h // 4:3 * h // 4, w // 4:3 * w // 4]
    blue = (centre[:, :, 0] > 95) & (centre[:, :, 0] < 125) & (centre[:, :, 1] > 120)
    return bool(blue.mean() > 0.8)


def infer_deck(reads: list[dict[str, SlotRead]], size: int = 8) -> list[int]:
    """The `size` cards seen most often across a batch of hand reads (from the full reader).
    A match's whole deck shows up in the hand within the first couple of cycles."""
    counts: dict[int, int] = {}
    for frame in reads:
        for r in frame.values():
            if r.state == "card":
                counts[r.card_id] = counts.get(r.card_id, 0) + 1
    return [cid for cid, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:size]]
