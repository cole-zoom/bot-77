"""The play page's tile grid (scripts/situation_play.py, P0.1 of docs/plan-sim-1to1.md): the tiles it lights up are the
tiles the engine accepts, a click plays exactly at the tile's centre, and the terrain it draws is the engine's."""
import http.client
import json
import math
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

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


# --- the revamped page (2026-10-29): engine clock, sprites for everything, the next card, and the recording unchanged

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "scripts" / "play_page"
ENGINE_H = ROOT / "third_party" / "clash-royale-env" / "include" / "core"


def test_the_tick_comes_from_the_engine():
    """The page paces itself and its clock with placement.TICK_S, which the engine's double-elixir tick fixes."""
    assert pl.TICK_S == pytest.approx(120.0 / cre.ClashRoyaleEnv.DOUBLE_ELIXIR_TICK) == pytest.approx(0.1)
    assert pl.CLOCK["double_at"] == pytest.approx(120.0) and pl.CLOCK["regulation_end"] == 180.0
    assert pl.CLOCK["tick_s"] == pl.TICK_S


def test_the_recorded_play_time_is_unchanged():
    """t = round(n * TICK_S, 3) is exactly the old round(t + 0.1, 1) chain at 0.1 s a tick."""
    t = 0.0
    for n in range(1, 6001):
        t = round(t + 0.1, 1)
        assert round(n * pl.TICK_S, 3) == t


def engine_unit_names():
    """Every name the engine can put on the board: its registry, plus the bodies cards spawn (negative ids, only in the
    registry source) and the crown towers."""
    names = {cre.get_card_info(i)["name"] for i in cre.get_all_card_ids()}
    src = "".join(p.read_text() for p in ENGINE_H.glob("*.h"))
    names |= set(re.findall(r'\(-\d+, "([^"]+)"', src))
    return names | {"King Tower", "Princess Tower"}


def sprite_tables():
    js = (PAGE / "sprites.js").read_text()
    sprites = js[js.index("const SPRITES = {"):js.index("};", js.index("const SPRITES = {"))]
    spells = js[js.index("const SPELLS = {"):js.index("};", js.index("const SPELLS = {"))]
    return (dict(re.findall(r'^\s*"([^"]+)": \["[A-Z]", "([^"]+)"', sprites, re.M)),
            dict(re.findall(r'"([^"]+)": \["#[0-9a-f]{6}", "[a-z]+", "([^"]+)"\]', spells)))


def test_every_unit_the_engine_can_spawn_has_its_own_sprite():
    sprites, spells = sprite_tables()
    spell_cards = {cre.get_card_info(i)["name"] for i in cre.get_all_card_ids() if cre.get_card_info(i)["is_spell"]}
    assert spell_cards - set(spells) == set(), "spells without an area look"
    missing = engine_unit_names() - spell_cards - set(sprites)
    assert missing == set(), f"units without a sprite: {sorted(missing)}"
    assert len(set(sprites.values())) == len(sprites), "two sprites share a label"
    assert len(engine_unit_names()) > 140  # the registry plus ~40 spawned bodies


def test_no_game_assets_in_the_page():
    """The art and sound are drawn and synthesised in code: no image or audio files, no client paths."""
    assert {p.suffix for p in PAGE.iterdir()} <= {".html", ".css", ".js"}
    for p in PAGE.iterdir():
        assert "cr_client" not in p.read_text() and "data:image" not in p.read_text() and "data:audio" not in p.read_text()


def test_the_next_card_is_the_one_that_comes():
    g = fresh(3)
    for step in range(16):
        nxt = pl.next_card(g)
        g.set_elixir_for_team(0, 10.0)
        hand = list(g.get_hand())
        k = next(k for k in range((step * 3) % 4, (step * 3) % 4 + 4) if pl.legal_mask(g, hand[k % 4]).find("1") >= 0) % 4
        cell = pl.legal_mask(g, hand[k]).index("1")
        g.step(k, float(cell % 18), float(cell // 18), 1)
        if list(g.get_hand())[k] != hand[k]:   # it went (a slot can be on cooldown)
            assert list(g.get_hand())[k] == nxt
        for _ in range(12):
            g.step(4, 0.0, 0.0, 1)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


REPLAY = r'''
import json, os, sys
import numpy as np
ROOT = sys.argv[1]
os.environ.update({"CLASH_DECK": "2,81,5,43,9,95,7,3", "RL_CLONE_CKPT": ROOT + "/data/models/bc_v2/frozen.pth",
                   "RL_PRO_DECKS": ROOT + "/data/cache/rl/pro_decks.json", "RL_SITUATION_SHARE": "1"})
sys.path.insert(0, ROOT + "/third_party/clash-royale-env"); os.chdir(ROOT + "/third_party/clash-royale-env")
import python_ai, torch
import clash_royale_env as cre
cre.set_client_collision(True)  # the page's physics (its log line's "engine")
from bot77.rl import situations as sit
from bot77.rl.clone_env import CloneEnv
torch.set_num_threads(2)
tpl, seed, actions = sys.argv[2], int(sys.argv[3]), json.loads(sys.argv[4])
t = {f.__name__: f for f in sit.TEMPLATES + sit.REVIEW_TEMPLATES + sit.HELD_OUT_TEMPLATES}[tpl]
env = CloneEnv({"scenario_seed": 0, "decision_ticks": 1})
env.forced_template = t; env.rng = np.random.default_rng(seed); env.scenario_rng = env.rng; torch.manual_seed(seed)
env.reset()
for k, x, y in actions:   # the engine action a page play maps to: card slot, the tile's centre
    obs, r, term, trunc, info = env.step({"card_index": k, "target_x": np.array([x], np.float32), "target_y": np.array([y], np.float32),
                                          "activate_ability_slot1": 0, "activate_ability_slot2": 0})
    if info.get("rl_situation_end") or term or trunc:
        break
print(json.dumps(env.last_situation, default=float))
'''


@pytest.mark.skipif(not (ROOT / "data/models/rl_clones_v3/final.pth").exists() or
                    not (ROOT / "data/models/bc_v2/frozen.pth").exists(), reason="needs the local model checkpoints")
def test_the_page_records_and_plays_exactly_what_it_did(tmp_path):
    """Cole's games are answer keys. Through the page's own HTTP API: the log line keeps its format, and replaying the
    same per-tick actions straight into CloneEnv (a play = slot + tile centre, no play = slot 4) gives the same result."""
    port, log = free_port(), tmp_path / "play.jsonl"
    srv = subprocess.Popen([sys.executable, str(ROOT / "scripts/situation_play.py"), "--port", str(port), "--log", str(log)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT)
    url = f"http://127.0.0.1:{port}"

    def post(path, body):
        req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=120).read())
    try:
        for _ in range(120):
            try:
                board = json.loads(urllib.request.urlopen(url + "/board", timeout=2).read())
                break
            except OSError:
                time.sleep(0.5)
        assert board["tick_s"] == pl.TICK_S and (board["w"], board["h"]) == (pl.W, pl.H)
        cat = json.loads(urllib.request.urlopen(url + "/catalog").read())
        assert len(cat["cards"]) == len(cre.get_all_card_ids())
        for path in ("/", "/static/app.js", "/static/sprites.js", "/static/sound.js", "/static/style.css"):
            assert urllib.request.urlopen(url + path).status == 200
        conn = http.client.HTTPConnection("127.0.0.1", port)   # no way out of the page's folder
        conn.request("GET", "/static/../situation_play.py")
        assert conn.getresponse().status == 404
        tpl, seed = "hog_rider", 2026
        st, actions, n = post("/new", {"template": tpl, "seed": seed})["state"], [], 0
        while True:
            n += 1
            body = {}
            if n in (3, 12, 30):
                k = n % 4
                cell = st["hand"][k]["legal"].find("1")
                body = {"slot": k, "x": 9.0, "y": 30.0} if n == 12 else {"slot": k, "x": float(cell % 18), "y": float(cell // 18)}
            actions.append([body["slot"], body["x"], body["y"]] if body else [4, 0.0, 0.0])
            out = post("/step", body)
            st = out["state"]
            assert {"target", "targetable", "deploy"} <= set(st["units"][0]) and "next" in st
            if st["done"]:
                break
        line = json.loads(log.read_text().splitlines()[-1])
        assert set(line) == {"when", "situation", "seed", "plays", "result", "engine"} and line["seed"] == seed
        assert line["engine"] == {"client_collision": True}
        assert [set(p) - {"why"} for p in line["plays"]] == [{"t", "card", "x", "y", "landed"}] * 3
        assert [p["t"] for p in line["plays"]] == [0.3, 1.2, 3.0] and line["plays"][1]["landed"] is False
        assert set(line["result"]) == {"you", "scripted", "model"}
    finally:
        srv.terminate()
        srv.wait(10)
    replay = subprocess.run([sys.executable, "-c", REPLAY, str(ROOT), tpl, str(seed), json.dumps(actions)],
                            capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert replay.returncode == 0, replay.stderr[-2000:]
    assert json.loads(replay.stdout.strip().splitlines()[-1]) == line["result"]["you"]
