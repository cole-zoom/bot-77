"""frame_states: one row per frame of every game, the hands, elixir, board, towers, what happened next and (bot games)
the bot's own view, for mining failure moments with queries (docs/frame-states.md; Cole 2026-11-02: "Should we build a
lance table with columns for whats on the board and whats in the hand to make mining easier?", "why not every frame?").

  uv run --no-sync python scripts/build_frame_states.py --moments data/reviews/mined_v2_2026-10-31/moments.json

Bot games only for now (source replay_bot): the pairing of each live game with its Battle Log replay comes from a
miner output (scripts/mine_moments.py), which also tags rows with the modes that fired (`modes`). Cole's labels come
from his loss reviews (data/reviews/cole/*.json) through scripts/failure_modes.py. A row per hand read (0.1 s); units
are the latest detection at or before it (every 0.25 s; `units_age`). Plays are seen in the replay when the card
leaves the hand, ~1-2 s after the bot's tap. Replaces the table each run (--append to add).
"""
import argparse
import bisect
import json
import sys
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_loss_review as blr  # noqa: E402
import failure_modes as fm  # noqa: E402
import mine_moments as mm  # noqa: E402

from bot77.sim.geometry import ours_to_engine  # noqa: E402
from bot77.sim.validate import load_game, tower_frac  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN_MODEL = {"run40v4": "bc_school_v4/epoch03_cal20", "run40v4b": "bc_school_v4/epoch03_cal20",
             "run40v4c": "bc_school_v4/epoch03_cal20"}
TABLE = "frame_states"  # board_states is A's per-decision table (scripts/build_board_states.py, 2026-10-22)
DECK = ["Giant", "Battle Ram", "Mini P.E.K.K.A", "Mega Minion", "Bomber", "Goblin Hut", "Fireball", "Arrows"]
TOWER_KEYS = ("own_l", "own_r", "own_king", "opp_l", "opp_r", "opp_king")
GRID_X, GRID_Y = 6, 8  # the vector's board grid (3 x 4.25 tiles a cell)
DIM = 176

UNIT = pa.struct([("card", pa.string()), ("team", pa.string()), ("x", pa.float32()), ("y", pa.float32()),
                  ("lane", pa.string()), ("half", pa.string()), ("worth", pa.float32()), ("air", pa.bool_())])
PLAY = pa.struct([("card", pa.string()), ("x", pa.float32()), ("y", pa.float32())])
PLAY_NEXT = pa.struct([("card", pa.string()), ("x", pa.float32()), ("y", pa.float32()), ("in_s", pa.float32())])
SCHEMA = pa.schema([
    ("state_id", pa.string()), ("game", pa.string()), ("replay_id", pa.string()), ("source", pa.string()),
    ("model", pa.string()), ("result", pa.string()), ("t", pa.float32()), ("clock", pa.string()),
    ("phase", pa.string()), ("multiplier", pa.int8()), ("changed", pa.bool_()),
    ("our_hand", pa.list_(pa.string())), ("their_hand", pa.list_(pa.string())),
    ("our_next", pa.string()), ("their_next", pa.string()),
    ("our_elixir", pa.float32()), ("their_elixir", pa.float32()), ("elixir_lead", pa.float32()),
    ("their_cycle", pa.list_(pa.struct([("card", pa.string()), ("seconds_ago", pa.float32())]))),
    ("our_counters_in_hand", pa.list_(pa.string())), ("their_wincons_in_hand", pa.list_(pa.string())),
    ("units", pa.list_(UNIT)), ("units_age", pa.float32()),
    ("our_left_value", pa.float32()), ("our_right_value", pa.float32()),
    ("their_left_value", pa.float32()), ("their_right_value", pa.float32()),
    ("threat_left", pa.float32()), ("threat_right", pa.float32()),
    ("their_wall_left", pa.float32()), ("their_wall_right", pa.float32()),
    ("best_arrows_value", pa.float32()), ("best_arrows_xy", pa.list_(pa.float32())),
    ("best_fireball_value", pa.float32()), ("best_fireball_xy", pa.list_(pa.float32())),
    ("tower_hp", pa.struct([(k, pa.float32()) for k in TOWER_KEYS])),
    ("towers_ours", pa.int8()), ("towers_theirs", pa.int8()), ("king_awake", pa.bool_()), ("tempo", pa.string()),
    ("our_play_now", PLAY), ("their_play_now", PLAY), ("our_play_next", PLAY_NEXT), ("their_play_next", PLAY_NEXT),
    ("our_spell_hit", pa.float32()), ("play_nearest_foe", pa.float32()),
    ("our_tower_loss_6s", pa.float32()), ("our_tower_loss_12s", pa.float32()),
    ("their_tower_loss_6s", pa.float32()), ("their_tower_loss_12s", pa.float32()),
    ("bot_hand_read", pa.list_(pa.string())), ("blind", pa.bool_()), ("p_wait", pa.float32()),
    ("p_cards", pa.list_(pa.float32())), ("value", pa.float32()),
    ("vector", pa.list_(pa.float32(), DIM)), ("modes", pa.list_(pa.string())),
    ("cole_label", pa.struct([("mode", pa.string()), ("verdict", pa.string()), ("should", pa.string()),
                              ("note", pa.string())])),
])


def hud(db, match_id: str) -> list[tuple[float, list[str], float, str | None]]:
    """(video t, hand, elixir, next) for one side of a replay (mine_moments.hud plus the next card)."""
    mm.hud(db, match_id[:0] + "_")  # loads the card names
    rows = db.open_table("hud_states").search().where(f"match_id = '{match_id}'").select(
        ["t_video", "hand", "elixir", "next"]).limit(10**6).to_arrow().to_pylist()
    nm = lambda c: mm._NAMES.get(c, "?") if c is not None else "?"
    return [(r["t_video"], [nm(c) for c in (r["hand"] or [])], r["elixir"], nm(r["next"]) if r["next"] else None)
            for r in sorted(rows, key=lambda r: r["t_video"])]


def lane_of(x: float) -> str:
    return "left" if x < 9 else "right"


def board(g, tv: float):
    """The detection frame at or before video time tv: (frame time, units as dicts)."""
    i = bisect.bisect_right(g.frame_ts, tv) - 1
    if i < 0:
        return None, []
    ft = g.frame_ts[i]
    out = []
    for u in g.frames.get(ft, []):
        if not u["card"] or u["team"] not in ("own", "opp"):
            continue
        x, y = ours_to_engine(u["x"], u["y"])
        out.append({"card": u["card"], "team": u["team"], "x": float(x), "y": float(y), "lane": lane_of(x),
                    "half": "ours" if y < mm.RIVER else "theirs", "worth": float(mm.worth(u["card"])),
                    "air": u["card"] in mm.AIR_UNITS})
    return ft, out


_SHARE: dict[tuple[str, str], float] = {}


def share(card: str, spell: str) -> float:
    """The part of a unit's HP one spell takes (1 = it dies): Arrows 3 x 123 = 369, Fireball 689 (engine stats). Its
    worth times this is what the spell wins on it: Arrows on a lone Valkyrie (1,907 HP) is 0.8 of her 4 elixir, not 4."""
    if (card, spell) not in _SHARE:
        import clash_royale_env as cre
        from bot77.sim import cards as sc
        e, s_ = sc.engine_id(card), sc.engine_id(spell)
        if e is None or s_ is None:
            _SHARE[card, spell] = 1.0
        else:
            st, sp = dict(cre.get_card_stats(int(e))), dict(cre.get_card_stats(int(s_)))
            dmg = sp["damage"] * max(1, sp.get("spell_hits") or 1)
            _SHARE[card, spell] = min(1.0, dmg / st["hp"]) if st.get("hp") else 1.0
    return _SHARE[card, spell]


def spell_value(units: list[dict], spell: str, x: float, y: float) -> float:
    r = mm.SPELLS[spell][0]
    return float(sum(u["worth"] * share(u["card"], spell) for u in units
                     if u["team"] == "opp" and (u["x"] - x) ** 2 + (u["y"] - y) ** 2 <= r * r))


def best_spell(units: list[dict], spell: str) -> tuple[float, list[float] | None]:
    """The elixir one spell would win, centred on one of their units (on our half or up to SPELL_UP_TO), and where."""
    foes = [u for u in units if u["team"] == "opp"]
    best = (0.0, None)
    for c in foes:
        if c["y"] > mm.SPELL_UP_TO:
            continue
        v = spell_value(units, spell, c["x"], c["y"])
        if v > best[0]:
            best = (v, [round(c["x"], 2), round(c["y"], 2)])
    return best


def tempo(t: float, lead: float, ours_up: int, theirs_up: int, ours_over: float) -> str:
    """The game's tempo, roughly by bot77.rl.situations' switches: desperate from 2:00 with fewer towers, press with
    a lead of 3+ or a push of ours over the river, even when behind on elixir, else chill."""
    if t >= 120 and ours_up < theirs_up:
        return "desperate"
    if lead >= 3 or ours_over >= 4:
        return "press"
    return "even" if lead < 0 else "chill"


def vector(units, our_hand, our_next, their_hand, our_el, their_el, hp, t, mult, play=None) -> list[float]:
    """176 floats: a 6x8 grid of worth per team (96, at half weight), our deck in hand / next (16), their hand hashed
    (24), elixir (3), towers (6), time and speed (2), lane values (8); then the play at this row, at double weight
    (2026-11-02: the board alone found Cole's pointless pushes no better than chance): the card (8), its spot (2), and
    in its lane their troops on our half, their wall, our troops (3), and the elixir lead (1)."""
    v = np.zeros(DIM, np.float32)
    for u in units:
        gx = min(GRID_X - 1, max(0, int(u["x"] / 18 * GRID_X)))
        gy = min(GRID_Y - 1, max(0, int(u["y"] / 34 * GRID_Y)))
        v[(0 if u["team"] == "own" else 48) + gy * GRID_X + gx] += u["worth"] / 6
    for i, c in enumerate(DECK):
        v[96 + i] = c in our_hand
        v[104 + i] = c == our_next
    for c in their_hand:
        if c and c != "?":
            v[112 + _bucket(c)] = 1
    v[136], v[137], v[138] = our_el / 10, their_el / 10, (our_el - their_el) / 10
    for i, k in enumerate(TOWER_KEYS):
        v[139 + i] = hp[k]
    v[145], v[146] = t / 300, mult / 3
    for i, (team, ln) in enumerate((("own", "left"), ("own", "right"), ("opp", "left"), ("opp", "right"))):
        v[147 + i] = sum(u["worth"] for u in units if u["team"] == team and u["lane"] == ln) / 10
        v[151 + i] = sum(u["worth"] for u in units if u["team"] == team and u["lane"] == ln and u["half"] == "ours") / 10
    if play and play["card"] in DECK:
        v[155 + DECK.index(play["card"])] = 2
        if play["x"] is not None:
            ln = lane_of(play["x"])
            v[163], v[164] = 2 * play["x"] / 18, 2 * play["y"] / 34
            lv = lambda team, half=None: sum(u["worth"] for u in units if u["team"] == team and u["lane"] == ln
                                             and (half is None or u["half"] == half)) / 5
            v[165], v[166], v[167] = 2 * lv("opp", "ours"), 2 * lv("opp", "theirs"), 2 * lv("own")
        v[168] = 2 * (our_el - their_el) / 10
    return v.tolist()


def _bucket(card: str) -> int:
    import zlib
    return zlib.crc32(card.encode()) % 24


def cole_labels() -> dict[tuple[str, float], dict]:
    out = {}
    for f in sorted((ROOT / "data/reviews/cole").glob("loss_review_*.json")):
        for g in json.loads(f.read_text())["games"]:
            for m in g.get("moments", []):
                modes = fm.COLE.get((g["gid"], m["t"]), [])
                out[(g["gid"], m["t"])] = {"mode": modes[0] if modes else None, "verdict": "mistake",
                                          "should": m.get("should") or None, "note": m.get("note") or None}
    return out


def build_game(db, gm: dict, replay_id: str, moments: list[dict], labels: dict) -> list[dict]:
    g = load_game(replay_id, "bot", "replay_bot")
    ours = hud(db, g.game_id)
    theirs = hud(db, g.game_id.replace("_m01", "_top_m01"))
    if not ours:
        return []
    plays = [p for p in g.plays if p["t_match"] is not None and p["card"]]
    mine = [p for p in plays if p["side"] == "bottom"]
    their = [p for p in plays if p["side"] == "top"]
    spells = mm.our_spells(gm["gid"])

    def spot(p):
        if p.get("spot"):
            return [float(v) for v in ours_to_engine(*p["spot"])]
        if p["side"] == "bottom" and p["card"] in mm.SPELLS:  # the spot the bot tapped (bot_actions)
            s_ = min((s_ for s_ in spells if s_["card"] == p["card"]), key=lambda s_: abs(p["t_match"] - s_["t"] - 1),
                     default=None)
            if s_ is not None and -1 <= p["t_match"] - s_["t"] <= 4:
                return [float(s_["x"]), float(s_["y"])]
        return [None, None]
    stuck = mm.blank_stretches(gm["gid"])
    dec_f = blr.GAMES / gm["gid"] / "decisions.json"
    dec = sorted((r for r in json.loads(dec_f.read_text()) if r.get("clock") is not None), key=lambda r: r["clock"]) \
        if dec_f.exists() else []
    dec_clock = [r["clock"] for r in dec]
    their_ts = [p["t_match"] for p in their]
    rows, prev, cache = [], None, {}
    for tv, hand, el, nxt_us in ours:
        t = tv - g.offset
        if t < 0:
            continue
        th = mm.at(theirs, tv) or (tv, [], 0.0, None)
        our_hand, their_hand = list(hand[:4]), list(th[1][:4])
        our_el, their_el = float(el or 0), float(th[2] or 0)
        ft, units = board(g, tv)
        if ft not in cache:
            cache.clear()
            cache[ft] = (best_spell(units, "Arrows"), best_spell(units, "Fireball"))
        (av, axy), (fv, fxy) = cache[ft]
        val = lambda team, ln, half=None: float(sum(u["worth"] for u in units if u["team"] == team and u["lane"] == ln
                                                    and (half is None or u["half"] == half)))
        hp = {key: float(tower_frac(g, key, tv) if tower_frac(g, key, tv) is not None else 1.0) for key in TOWER_KEYS}
        up_o, up_t = int(hp["own_l"] > 0) + int(hp["own_r"] > 0), int(hp["opp_l"] > 0) + int(hp["opp_r"] > 0)
        wincons = [c for c in their_hand if c in mm.WINCON]
        counters = mm.COUNTERS_AIR if wincons and all(c in mm.AIR for c in wincons) else mm.COUNTERS_GROUND
        i = bisect.bisect_right(their_ts, t)
        cycle = [{"card": p["card"], "seconds_ago": float(t - p["t_match"])} for p in their[max(0, i - 8):i]][::-1]
        now = lambda side: next(({"card": p["card"], "x": spot(p)[0], "y": spot(p)[1]} for p in side
                                 if t - 0.05 < p["t_match"] <= t + 0.05), None)
        nxt = lambda side: next(({"card": p["card"], "x": spot(p)[0], "y": spot(p)[1], "in_s": float(p["t_match"] - t)}
                                 for p in side if p["t_match"] > t + 0.05), None)
        loss = lambda keys, dt: float(sum(max(0.0, hp[key] - (tower_frac(g, key, tv + dt) or hp[key])) for key in keys))
        hit = None
        for sp in spells:
            if abs(sp["t"] - t) <= 0.05 + 1e-6:
                hit = max(spell_value(board(g, tv + d)[1], sp["card"], sp["x"], sp["y"]) for d in (0.3, 0.8, 1.3))
        d = None
        if dec:
            j = bisect.bisect_left(dec_clock, t)
            cand = [dec[x] for x in (j - 1, j) if 0 <= x < len(dec)]
            d = min(cand, key=lambda r: abs(r["clock"] - t)) if cand else None
            if d is not None and abs(d["clock"] - t) > 1.5:
                d = None
        pn = now(mine)
        near = None  # our play's distance to their nearest ground unit worth 1.5+ (a drop "on top")
        if pn and pn["x"] is not None:
            ds = [((u["x"] - pn["x"]) ** 2 + (u["y"] - pn["y"]) ** 2) ** 0.5 for u in units
                  if u["team"] == "opp" and not u["air"] and u["worth"] >= 1.5]
            near = float(min(ds)) if ds else None
        lead = our_el - their_el
        over = val("own", "left", "theirs") + val("own", "right", "theirs")
        key_now = (tuple(our_hand), tuple(their_hand), round(our_el, 1), round(their_el, 1), ft)
        row = {
            "state_id": f"{gm['gid']}@{t:.1f}", "game": gm["gid"], "replay_id": replay_id, "source": "replay_bot",
            "model": RUN_MODEL.get(gm.get("run")), "result": gm["result"], "t": t, "clock": mm.mmss(t),
            "phase": "overtime" if t > 180 else "regulation", "multiplier": 1 if t < 120 else 2,
            "changed": key_now != prev,
            "our_hand": our_hand, "their_hand": their_hand, "our_next": nxt_us,
            "their_next": th[3],
            "our_elixir": our_el, "their_elixir": their_el, "elixir_lead": lead,
            "their_cycle": cycle,
            "our_counters_in_hand": [c for c in our_hand if c in counters], "their_wincons_in_hand": wincons,
            "units": units, "units_age": float(tv - ft) if ft is not None else None,
            "our_left_value": val("own", "left"), "our_right_value": val("own", "right"),
            "their_left_value": val("opp", "left"), "their_right_value": val("opp", "right"),
            "threat_left": val("opp", "left", "ours"), "threat_right": val("opp", "right", "ours"),
            "their_wall_left": val("opp", "left", "theirs"), "their_wall_right": val("opp", "right", "theirs"),
            "best_arrows_value": av, "best_arrows_xy": axy, "best_fireball_value": fv, "best_fireball_xy": fxy,
            "tower_hp": hp, "towers_ours": up_o, "towers_theirs": up_t,
            "king_awake": hp["opp_king"] < 1 or up_t < 2, "tempo": tempo(t, lead, up_o, up_t, over),
            "our_play_now": pn, "play_nearest_foe": near, "their_play_now": now(their), "our_play_next": nxt(mine),
            "their_play_next": nxt(their), "our_spell_hit": hit,
            "our_tower_loss_6s": loss(("own_l", "own_r"), 6), "our_tower_loss_12s": loss(("own_l", "own_r"), 12),
            "their_tower_loss_6s": loss(("opp_l", "opp_r"), 6), "their_tower_loss_12s": loss(("opp_l", "opp_r"), 12),
            "bot_hand_read": [mm._NAMES.get(c, "?") if c is not None else "?" for c in d["hand"]] if d else None,
            "blind": any(a <= t <= b for a, b in stuck),
            "p_wait": float(d["p"][4]) if d else None, "p_cards": [float(x) for x in d["p"][:4]] if d else None,
            "value": float(d["value"]) if d else None,
            "vector": vector(units, our_hand, nxt_us, their_hand, our_el, their_el, hp, t, 1 if t < 120 else 2, pn),
            "modes": sorted({m["kind"] for m in moments if abs(m["t"] - t) <= 0.05 + 1e-6}),
            "cole_label": next((lab for (gid, lt), lab in labels.items() if gid == gm["gid"] and abs(lt - t) <= 0.05 + 1e-6),
                               None),
        }
        prev = key_now
        rows.append(row)
    # a moment or note whose time falls between rows goes on the nearest one
    for kind_t, field in [((m["t"], m["kind"]), "modes") for m in moments] + \
                         [((lt, lab), "cole_label") for (gid, lt), lab in labels.items() if gid == gm["gid"]]:
        tt, val_ = kind_t
        if not rows:
            break
        r = min(rows, key=lambda r: abs(r["t"] - tt))
        if field == "modes" and val_ not in r["modes"]:
            r["modes"] = sorted(r["modes"] + [val_])
        elif field == "cole_label" and r["cole_label"] is None:
            r["cole_label"] = val_
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moments", required=True, help="a miner output (moments.json): the game -> replay pairing")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--games", nargs="*", help="only these live game ids")
    a = ap.parse_args()
    data = json.loads(Path(a.moments).read_text())
    db = lancedb.connect(ROOT / "data/lancedb")
    labels = cole_labels()
    first = not a.append
    total = 0
    for gid, gm in data["games"].items():
        if a.games and gid not in a.games:
            continue
        ms = [m for m in data["moments"] if m["game"] == gid]
        if not ms:
            continue
        rows = build_game(db, gm, ms[0]["replay"], ms, labels)
        if not rows:
            print(f"{gid}: no rows")
            continue
        t = pa.Table.from_pylist(rows, SCHEMA)
        if first:
            tbl = db.create_table(TABLE, t, mode="overwrite")
            first = False
        else:
            tbl = db.open_table(TABLE)
            tbl.add(t)
        total += len(rows)
        print(f"{gid} vs {gm['opp']:18s} {len(rows)} rows ({sum(r['changed'] for r in rows)} changed, "
              f"{sum(r['blind'] for r in rows)} blind)", flush=True)
    tbl = db.open_table(TABLE)
    for col in ("game", "source"):
        try:
            tbl.create_scalar_index(col, replace=True)
        except Exception as e:  # noqa: BLE001
            print(f"index {col}: {e}")
    print(f"{total} rows -> {TABLE} ({tbl.count_rows()} in the table)")


if __name__ == "__main__":
    main()
