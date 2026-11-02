"""Find each failure mode in frame_states (docs/frame-states.md) and score it against Cole's notes, all in LanceDB.

  uv run --no-sync python scripts/query_modes.py [--mode missed_spell] [--show 5] [--k 40]

Two searches per mode:
  rule    a filter (QUERIES): the mode written as conditions on the row.
  hybrid  vector search from the rows at Cole's notes of that mode, prefiltered to the rows the mode can be about
          (SCOPE). Scored leave-one-out: each note must be found from his *other* notes of the mode.
Hits in one game less than GAP s apart are one moment. Found = a moment within MATCH_S of a note with that mode (main
or second). Rows where the live reader had a card locked out (`blind`) are never shown. Keep the queries here and in
docs/frame-states.md the same.
"""
import argparse
import sys
from pathlib import Path

import lancedb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import failure_modes as fm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GAP, MATCH_S = 4.0, 5.0
BASE = "NOT blind AND source = 'replay_bot'"


def lane(col: str, cond: str, suf: str = "") -> str:
    """`col` of the lane our play is in (LanceDB's filter has no CASE)."""
    return (f"((our_play_now.x < 9 AND {col}_left{suf} {cond}) OR (our_play_now.x >= 9 AND {col}_right{suf} {cond}))")


QUERIES = {
    "missed_spell": """
        ((best_arrows_value >= 3.5 AND array_has(our_hand, 'Arrows') AND our_elixir >= 3
          AND NOT coalesce(our_play_next.card = 'Arrows' AND our_play_next.in_s <= 3, false))
      OR (best_fireball_value >= 5 AND array_has(our_hand, 'Fireball') AND our_elixir >= 4
          AND NOT coalesce(our_play_next.card = 'Fireball' AND our_play_next.in_s <= 3, false)))""",
    "pointless_push": f"""
        our_play_now.card IN ('Giant', 'Battle Ram') AND our_play_now.y >= 12.5 AND towers_ours >= towers_theirs
        AND {lane('threat', '< 2')} AND {lane('our', '< 3', '_value')}
        AND (elixir_lead < -0.5 OR {lane('their_wall', '>= 6')})""",
    "spent_counter": """
        our_play_now.card IN ('Mini P.E.K.K.A', 'Goblin Hut', 'Mega Minion')
        AND array_length(their_wincons_in_hand) > 0 AND array_length(our_counters_in_hand) = 0""",
    "wasted_spell": """
        our_play_now.card IN ('Arrows', 'Fireball')
        AND ((our_play_now.card = 'Arrows' AND our_spell_hit < 2.5) OR (our_play_now.card = 'Fireball' AND our_spell_hit < 3.5))""",
    "too_early": f"""
        our_play_now.card IN ('Mini P.E.K.K.A', 'Mega Minion', 'Bomber') AND our_play_now.y < 15.5
        AND threat_left + threat_right < 1 AND {lane('their', '>= 3', '_value')}""",
    "bad_placement": """
        (our_play_now.card IN ('Mini P.E.K.K.A', 'Bomber') AND play_nearest_foe <= 1.5)
     OR (our_play_now.card IN ('Mini P.E.K.K.A', 'Mega Minion', 'Bomber') AND our_play_now.y < 15.5
         AND ((our_play_now.x < 9 AND threat_left = 0 AND threat_right >= 3)
           OR (our_play_now.x >= 9 AND threat_right = 0 AND threat_left >= 3)))""",
    "no_defence": """
        ((threat_left >= 3 AND tower_hp.own_l > 0) OR (threat_right >= 3 AND tower_hp.own_r > 0))
        AND our_elixir >= 3 AND coalesce(our_play_next.in_s > 4, true) AND our_tower_loss_6s >= 0.05""",
    "missed_finish": """
        ((tower_hp.opp_l > 0 AND tower_hp.opp_l <= 0.07) OR (tower_hp.opp_r > 0 AND tower_hp.opp_r <= 0.07))
        AND array_has(our_hand, 'Fireball') AND our_elixir >= 4
        AND NOT coalesce(our_play_next.card IN ('Fireball', 'Arrows') AND our_play_next.in_s <= 5, false)""",
}
# the rows each mode can be about (the hybrid search's prefilter), and whether a note sits on one of our plays
SCOPE = {
    "missed_spell": ("best_arrows_value >= 2 OR best_fireball_value >= 3", False),
    "pointless_push": ("our_play_now.card IN ('Giant', 'Battle Ram', 'Mini P.E.K.K.A', 'Mega Minion', 'Bomber')", True),
    "spent_counter": ("our_play_now.card IN ('Mini P.E.K.K.A', 'Goblin Hut', 'Mega Minion')", True),
    "wasted_spell": ("our_play_now.card IN ('Arrows', 'Fireball')", True),
    "too_early": ("our_play_now.card IN ('Mini P.E.K.K.A', 'Mega Minion', 'Bomber')", True),
    "bad_placement": ("our_play_now.card IN ('Mini P.E.K.K.A', 'Mega Minion', 'Bomber')", True),
    "no_defence": ("threat_left + threat_right >= 2", False),
    "missed_finish": ("(tower_hp.opp_l > 0 AND tower_hp.opp_l <= 0.15) OR (tower_hp.opp_r > 0 AND tower_hp.opp_r <= 0.15)", False),
}
COLS = ["game", "t", "clock"]


def group(rows: list[dict], mode: str) -> list[dict]:
    out, last = [], {}
    for r in sorted(rows, key=lambda r: (r["game"], r["t"])):
        if r["game"] in last and r["t"] - last[r["game"]] < GAP:
            last[r["game"]] = r["t"]
            continue
        last[r["game"]] = r["t"]
        out.append({"game": r["game"], "t": round(float(r["t"]), 1), "clock": r["clock"], "kind": mode})
    return out


def rule(tbl, mode: str) -> list[dict]:
    return group(tbl.search().where(f"{BASE} AND ({QUERIES[mode]})").select(COLS).limit(10**6).to_list(), mode)


def note_row(tbl, game: str, t: float, on_play: bool) -> dict | None:
    """The row a note points at: our play nearest it (within MATCH_S) for a play mode, else the nearest row."""
    w = f"game = '{game}' AND t >= {t - MATCH_S} AND t <= {t + MATCH_S}" + (" AND our_play_now IS NOT NULL" if on_play else "")
    rows = tbl.search().where(w).select(COLS + ["vector", "blind"]).limit(1000).to_list()
    return min(rows, key=lambda r: abs(r["t"] - t)) if rows else None


def hybrid(tbl, mode: str, examples: list[dict], k: int) -> list[dict]:
    scope, _ = SCOPE[mode]
    hits = []
    for ex in examples:
        hits += tbl.search(ex["vector"]).where(f"{BASE} AND ({scope})", prefilter=True).select(COLS).limit(k).to_list()
    return group(hits, mode)


def found(ms: list[dict], key: tuple[str, float]) -> bool:
    return any(m["game"] == key[0] and abs(m["t"] - key[1]) <= MATCH_S for m in ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=None)
    ap.add_argument("--show", type=int, default=0, help="print this many of Cole's notes it misses, per mode")
    ap.add_argument("--k", type=int, default=40, help="hybrid: nearest rows per example")
    a = ap.parse_args()
    tbl = lancedb.connect(ROOT / "data/lancedb").open_table("frame_states")
    games = {r["game"] for r in tbl.search().select(["game"]).limit(10**7).to_list()}
    notes = {key: v for key, v in fm.COLE.items() if key[0] in games}
    print(f"{tbl.count_rows()} rows, {len(games)} games; Cole's notes in them: {len(notes)} of {len(fm.COLE)}")
    tot = {"rule": 0, "hybrid": 0, "either": 0, "n": 0}
    for mode in QUERIES:
        if a.mode and mode != a.mode:
            continue
        want = [key for key, v in notes.items() if mode in v]
        rows = {key: note_row(tbl, *key, SCOPE[mode][1]) for key in want}
        r_ms = rule(tbl, mode)
        r_hit = [key for key in want if found(r_ms, key)]
        h_hit, h_n = [], 0
        for key in want:  # leave one out
            others = [rows[o] for o in want if o != key and rows[o] is not None and not rows[o]["blind"]]
            if not others:
                continue
            ms = hybrid(tbl, mode, others, a.k)
            h_n += len(ms)
            if found(ms, key):
                h_hit.append(key)
        either = set(r_hit) | set(h_hit)
        for name, v in (("rule", len(r_hit)), ("hybrid", len(h_hit)), ("either", len(either)), ("n", len(want))):
            tot[name] += v
        print(f"  {fm.MODES[mode][0]:22s} rule {len(r_hit)}/{len(want)} ({len(r_ms) / len(games):.1f} a game)   "
              f"hybrid {len(h_hit)}/{len(want)} ({h_n / max(1, len(want)) / len(games):.1f} a game)   either {len(either)}/{len(want)}")
        for key in [key for key in want if key not in either][:a.show]:
            print(f"      missed {key[0]} @ {key[1]}")
    print(f"  all: rule {tot['rule']}, hybrid {tot['hybrid']}, either {tot['either']} of {tot['n']} (note x mode pairs)")


if __name__ == "__main__":
    main()
