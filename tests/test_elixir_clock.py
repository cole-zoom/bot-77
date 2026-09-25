import json
from pathlib import Path

import pytest

from bot77.layout import Layout
from bot77.readers.clock import elapsed, parse_clock, parse_phase
from bot77.readers.elixir import read_elixir
from tests.test_hand import CASES, full_frame


@pytest.mark.parametrize("name", sorted(n for n, m in CASES.items() if "elixir" in m))
def test_elixir_whole_matches_number(name):
    meta = CASES[name]
    r = read_elixir(full_frame(name, meta), Layout.load(meta["layout"]))
    assert r.whole == meta["elixir"], r
    # the fractional sliver can fill a whole segment just before the number ticks up
    assert r.consistent and meta["elixir"] <= r.elixir < meta["elixir"] + 1.1, r


@pytest.mark.parametrize("text, seconds", [("2:20", 140), ("0:04", 4), ("1:5O", 110), ("Time 0.33", 33), ("12", None), ("1:75", None)])
def test_parse_clock(text, seconds):
    assert parse_clock(text) == seconds


@pytest.mark.parametrize("text, phase", [("Time left:", "regulation"), ("OveRtiMe", "overtime"), ("", None)])
def test_parse_phase(text, phase):
    assert parse_phase(text) == phase


def test_elapsed_ladder_and_other_modes():
    assert elapsed("regulation", 140) == 40
    assert elapsed("overtime", 60) == 240
    assert elapsed("overtime", 60, mode="princess_gambit") is None
