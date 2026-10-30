"""What the lab battles showed the sim already gets right (decisions.md, lab night 1, 2026-10-29): each test plays a
lab test's plan in the bare engine (bot77.sim.labcmp.bare_run, the play page's physics) and checks Cole's read or the
replay's numbers, so an engine change that breaks a matched behaviour fails here."""
import math

import pytest

pytest.importorskip("clash_royale_env")

from bot77 import lab  # noqa: E402
from bot77.sim import labcmp  # noqa: E402

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _test(tid: str) -> lab.LabTest:
    return lab.load_tests(ROOT / "config/lab")[tid]


def _track(frames, card, team):
    """[(t, unit)] for the first unit of that card and team, while it lives."""
    out, uid = [], None
    for f in frames:
        us = [u for u in f["units"] if u["card"] == card and u["team"] == team and (uid is None or u["id"] == uid)]
        if us:
            uid = us[0]["id"]
            out.append((f["t"], us[0]))
        elif uid is not None:
            break
    return out


def _at(track, t):
    return min(track, key=lambda tu: abs(tu[0] - t))[1]


def test_l21_hog_ignores_a_hut_dropped_behind_it():
    # Cole: "it did not turn back to the hut". The Hog keeps on the tower and never targets the Hut.
    fr = labcmp.bare_run(_test("L21"), t_end=45)
    hog, hut = _track(fr, "Hog Rider", "opp"), _track(fr, "Goblin Hut", "own")
    assert hog and hut
    assert all(u["target"] != hut[0][1]["id"] for _, u in hog)
    assert min(u["y"] for _, u in hog) < 9.0 and abs(hog[-1][1]["x"] - 14.0) < 0.6  # at its tower
    assert fr[-1]["towers"]["own_r"] < 0.8


def test_l09_hog_goes_for_a_centre_hut_and_takes_it_down_first():
    # Cole: "to the hut, destroyed the hut, but died too".
    fr = labcmp.bare_run(_test("L09"), t_end=60)
    hog, hut = _track(fr, "Hog Rider", "opp"), _track(fr, "Goblin Hut", "own")
    assert any(u["target"] == hut[0][1]["id"] for _, u in hog)
    assert hut[-1][0] < hog[-1][0]  # the Hut goes before the Hog
    assert hog[-1][0] < 50.0  # and the Hog dies too


def test_l02_minipekka_beats_the_knight_with_decent_hp():
    # Cole: "mini pekka won with a decent amount of hp" (the replay: the Knight went at 34.5 s, the Mini P.E.K.K.A
    # lived on to the tower).
    fr = labcmp.bare_run(_test("L02"), t_end=40)
    mp, kn = _track(fr, "Mini P.E.K.K.A", "own"), _track(fr, "Knight", "opp")
    assert kn[-1][0] < mp[-1][0]
    left = _at(mp, kn[-1][0] + 0.5)["hp"] / mp[0][1]["hp"]
    assert left >= 0.35


def test_l04_giant_walks_the_lane_like_the_replay():
    # r58: the Giant at play + 2 / 4 / 6 / 8 s, engine tiles (the sim matched within 0.22).
    fr = labcmp.bare_run(_test("L04"), t_end=35)
    g = _track(fr, "Giant", "own")
    for dt, (x, y) in {2: (2.8, 13.9), 4: (2.7, 15.9), 6: (2.7, 17.8), 8: (2.7, 19.8)}.items():
        u = _at(g, 25.0 + dt)
        assert math.hypot(u["x"] - x, u["y"] - y) < 0.5, (dt, u["x"], u["y"])


def test_l21_goblin_hut_wakes_on_the_passing_hog():
    # r44 / r54: the first Spear Goblin 0.7-1.2 s after the Hut lands, three in all while the Hog is near.
    fr = labcmp.bare_run(_test("L21"), t_end=45)
    sg = [f["t"] for f in fr if any(u["card"] == "Spear Goblins" and u["team"] == "own" for u in f["units"])]
    assert sg and sg[0] - 29.5 <= 1.4
