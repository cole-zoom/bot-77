"""The play page's tile grid (scripts/situation_play.py, P0.1 of docs/plan-sim-1to1.md): the tiles it lights up are the
tiles the engine accepts, a click plays exactly at the tile's centre, and the terrain it draws is the engine's."""
import math

import clash_royale_env as cre
import pytest

from bot77.sim import placement as pl

DECK = [2, 81, 5, 43, 9, 95, 7, 3]  # the play page's deck (CLASH_DECK): Giant ... Goblin Hut, Fireball, Arrows


def fresh(seed=0):
    g = cre.ClashRoyaleEnv(DECK, DECK, 3600)
    g.seed(seed)
    g.reset()
    return g


def click_to_cell(px, py, width, height):
    """The page's cellAt (canvas pixel from the top-left -> tile), in Python."""
    fx, fy = px / width * 18, (height - py) / height * 34
    return min(17, max(0, math.floor(fx))), min(33, max(0, math.floor(fy)))


def cell_to_pixel(x, y, width, height):
    """The page's X / Y (engine point -> canvas pixel): cell i covers [i - 0.5, i + 0.5]."""
    return (x + 0.5) * width / 18, height - (y + 0.5) * height / 34


def test_a_click_anywhere_in_a_tile_plays_at_its_centre():
    w, h = 362.64, 685.0  # the board's size on a 1440 x 900 laptop window
    for x, y in [(0, 0), (9, 9), (17, 33), (3, 15), (14, 16)]:
        cx, cy = cell_to_pixel(x, y, w, h)
        for dx, dy in [(0, 0), (-0.49, 0.49), (0.49, -0.49)]:  # inside the tile, near its corners
            assert click_to_cell(cx + dx * w / 18, cy + dy * h / 34, w, h) == (x, y)


def test_the_lit_tiles_are_exactly_the_tiles_the_engine_accepts():
    g = fresh()
    for k, card in enumerate(g.get_hand()):
        mask = pl.legal_mask(g, card)
        for y in range(34):
            for x in range(18):
                s = g.snapshot()
                spent = s.get_elixir_spent(0)
                s.step(k, float(x), float(y), 1)
                played = s.get_elixir_spent(0) > spent + 1e-6
                assert played == (mask[y * 18 + x] == "1"), (card, x, y)


@pytest.mark.parametrize("card, cell", [(95, (9, 9)), (95, (4, 11)), (2, (12, 13)), (9, (0, 15))])
def test_a_play_stands_at_the_clicked_tile(card, cell):
    for seed in range(40):  # a deal with the card in hand
        g = fresh(seed)
        if card in g.get_hand():
            break
    else:
        pytest.skip("card never dealt")
    g.step(list(g.get_hand()).index(card), float(cell[0]), float(cell[1]), 1)
    mine = [u for u in g.get_units() if u.team == 0 and not u.tower and u.card_id == card]
    assert mine and (mine[0].x, mine[0].y) == pytest.approx(cell)


def test_the_terrain_is_the_engines():
    b = pl.board_geometry(fresh())
    assert b["river"] == [15.5, 17.5] and b["own_half_max_y"] == 15.0
    assert b["bridge_cols"] == [2, 3, 14, 15]  # Board::isOnBridge at the cell centres, 2.5 / 14.5 +- 1
    dead = {(i % 18, i // 18) for i, c in enumerate(b["dead"]) if c == "1"}
    assert (0, 0) in dead and (8, 0) not in dead and (17, 33) in dead and len(dead) == 24
    assert pl.card_spec(95)["placement_radius"] == 1.0 and pl.card_spec(7)["spell_radius"] == 2.5


def test_no_pocket_in_the_engine():
    """The engine never opens the pocket (fork: test_placement_legality_is_static). If it starts to, the page
    shows it with no change (it reads is_valid_placement), and this test says so."""
    g = fresh()
    before = pl.legal_mask(g, 2)
    g.destroy_tower(1, 1)
    assert pl.legal_mask(g, 2) == before
