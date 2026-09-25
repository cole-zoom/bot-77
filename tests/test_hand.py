import json
from pathlib import Path

import cv2
import lancedb
import numpy as np
import pytest

from bot77.layout import Layout
from bot77.readers.hand import HandReader, infer_deck

FIXTURES = Path("tests/fixtures/hand")
CASES = json.loads((FIXTURES / "labels.json").read_text())


@pytest.fixture(scope="module")
def cards():
    table = lancedb.connect("data/lancedb").open_table("cards")
    return table.search().where("kind = 'deck_card'").limit(500).to_arrow().to_pylist()


@pytest.fixture(scope="module")
def readers(cards):
    layouts = {m["layout"] for m in CASES.values()}
    return {lid: HandReader.from_cards(Layout.load(lid), cards) for lid in layouts}


@pytest.fixture(scope="module")
def reader(readers):
    return readers["alina_portrait"]


def full_frame(name: str, meta: dict) -> np.ndarray:
    """Fixtures store only the hand area; paste it back where it was in the source frame."""
    crop = cv2.imread(str(FIXTURES / name))
    w, h = meta["frame_size"]
    frame = np.zeros((h, w, 3), np.uint8)
    y, x = meta["y_offset"], meta.get("x_offset", 0)
    frame[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
    return frame


@pytest.mark.parametrize("name", sorted(CASES))
def test_hand_reads(readers, name):
    meta = CASES[name]
    reads = readers[meta["layout"]].read(full_frame(name, meta))
    for slot, expected in meta["labels"].items():
        r = reads[slot]
        if expected == "?":
            assert r.state == "unknown", f"{slot}: expected unreadable, got {r}"
        elif expected is None:
            assert r.state == "empty", f"{slot}: expected empty, got {r}"
        else:
            assert (r.state, r.name) == ("card", expected), f"{slot}: {r}"
    for slot, flags in meta["flags"].items():
        r = reads[slot]
        assert r.greyed == ("G" in flags), f"{slot} greyed: {r}"
        assert (r.form == "evo") == ("evo" in flags), f"{slot} form: {r}"
        assert (r.form == "hero") == ("hero" in flags), f"{slot} form: {r}"
        if "S" in flags:
            assert r.selected, f"{slot} selected: {r}"


def test_deck_restricted_reader_agrees(reader):
    alina = {n: m for n, m in CASES.items() if m["layout"] == "alina_portrait"}
    all_reads = [reader.read(full_frame(n, m)) for n, m in alina.items()]
    deck = infer_deck(all_reads)
    assert len(deck) == 8
    small = reader.restrict(set(deck))
    for (name, meta), full in zip(alina.items(), all_reads):
        for slot, r in small.read(full_frame(name, meta)).items():
            if full[slot].state == "card":
                assert (r.state, r.card_id, r.form) == ("card", full[slot].card_id, full[slot].form), (name, slot)
