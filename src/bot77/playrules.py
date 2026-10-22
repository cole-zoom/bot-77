"""Play rules v2 for opponent deploys (decisions.md, 2026-10-10), from the development-set error analysis
(scripts/analyze_dev.py on 22 paired matches; never the benchmark):

- continuation: a later "birth" of a card that is still around is not a new play. Goblinstein's doctor wanders back
  into view 15-60 s after the play; Spirit Empress switching ground <-> air is a new class, not a new card;
- long-lived spawners: Tombstone's skeletons, Goblin Drill's and Suspicious Bush's goblins appear long after the
  spawner's own detections stop, so "spawned" uses a time window instead of a live track;
- summons that are hero abilities (Tomb Queen = hero Tombstone's ability) are reported as "__ability", the label the
  truth uses for ability activations;
- our own deck: the POV player's cards are known exactly (hand reader), and their units near the river get called
  enemy; a candidate whose card is in our deck needs to start deep in the enemy half to count;
- timing: units appear ~1 s after the card leaves the hand (deploy time), spells about when they are cast; the event
  time is moved back by a per-card offset measured on the development set (config/play_timing.json).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DEFAULT_RULES = {
    "continuation_s": {"Goblinstein": 60.0, "Spirit Empress": 40.0, "Goblin Machine": 20.0, "Goblin Demolisher": 20.0},
    # only these classes continue a card (the doctor wanders back; a new monster is a new play); None = any class
    "continuation_classes": {"Goblinstein": ["goblinstein-doctor"], "Spirit Empress": None, "Goblin Machine": None,
                             "Goblin Demolisher": None},
    "continuation_tiles": 9.0,
    "long_spawners": {"Tombstone": [["Skeletons"], 45.0, 6.0], "Goblin Drill": [["Goblins"], 30.0, 6.0],
                      "Suspicious Bush": [["Goblins"], 30.0, 6.0], "Goblin Hut": [["Spear Goblins"], 60.0, 6.0],
                      "Witch": [["Skeletons"], 20.0, 6.0], "Graveyard": [["Skeletons"], 12.0, 6.0]},
    "ability_classes": ["tomb-queen"],  # adding dark-prince-hero-dismounted cost a play on the dev set
    # Mirror: the same card again within this gap and with fewer than mirror_max_between plays in between (the cycle).
    # Only a clocked repeat (a real new placement) and only when the deck has room for Mirror: no Mirror once
    # mirror_deck_full other cards each have >= mirror_deck_min_w deploys (weighted like infer_deck; 0 = no check).
    # A repeat that fails either test can't be a new play by the cycle, so it is dropped as a duplicate.
    # V5 detections, 2026-10-14: dev Mirror calls 73 (9 right) -> 16ish; see decisions.md for the sweep.
    "mirror_max_gap_s": 25.0, "mirror_need_clock": True, "mirror_max_between": 2,
    "mirror_deck_full": 8, "mirror_deck_min_w": 2.0, "mirror_drop_rest": True,
    "own_deck_max_y": 11.0,  # enemy half is y < 14.9 tiles; our units drift up to the river
    # ...but only if we played that card this recently (None: any time). Replays (2026-10-14): both players often run
    # the same meta cards, and the any-time rule deleted 95 real opponent plays near the river in 60 replays
    "own_deck_recent_s": None,
    "timing": True,
    "timing_file": None,  # None: config/play_timing.json; replays use config/play_timing_replay.json (see REPLAY_RULES)
    # deck inference (bot77.spawns.infer_deck): a deploy with no clock counts this much (clocked plays and spells: 1).
    # Replays 2026-10-14 (70 tuning matches): 0.6 vs 1.0 = right +1.3, wrong -3.1, fakes -2.4/match
    "deck_w_unclocked": 0.6,
    # spell stage (bot77.spells): directory of per-match classifier probabilities (scripts/katacr/infer_spell.py)
    # POV recordings: v2+v5 mix (v5 learned opponent Curse / Vines / Void from Cole's keyed live games; dev 430 -> 433);
    # replays keep the v2+v3 mix (REPLAY_SPELL_PROBS; held-back 1212 vs 1206 with v2+v5), 2026-10-18
    "spell_probs": "data/models/spell_mix25/spell_probs",
    "spell_thr": 0.7, "spell_group_s": 1.2, "spell_class_thr": {}, "spell_any_own": False,  # mix makes the old BB/Void cutoffs unnecessary
    # detector classes ignored entirely: "rage" fires on any spell circle (Poison, Curse, Tornado...) and its fake Rage
    # plays took a deck slot in 141 of ~400 replays (failure mining, 2026-10-16); the spell stage still finds Rage
    # spawns._on_own_half: tiles past the river a new enemy unit may be first seen (None = spawns.OWN_HALF_MARGIN, 0.8).
    # Replays use REPLAY_HALF_MARGIN: fast units (Hog Rider, Royal Hogs, Prince) and pocket plays are first seen 1-3
    # tiles past the river (failure mining, 2026-10-16); POV recordings lose with a wider margin
    "half_margin": None,
    # bot77.elixir: drop plays the opponent couldn't have afforded (start: elixir at the match start)
    "elixir_check": {"on": False, "tol": 1.0, "start": 5.0, "burst_s": 12.0},
    # bot77.fakefilter: learned P(fake) per play (scripts/train_fake_filter.py); thr for replays, thr_pov for recordings
    "fake_filter": {"model": "data/models/fake_filter_v4/model.pkl", "thr": 0.6, "thr_pov": 0.6},  # v4: + 33 keyed live games, POV weight 3 (2026-10-20)
    "pocket": True,  # spawns.towers_down: after our princess tower falls, allow POCKET_TILES past the river in that lane
    "det_ignore": ["clone"],
    "composite_peak": {"Skeletons": ["skeleton", 5, "Skeleton Army"], "Minions": ["minion", 4, "Minion Horde"]},  # spawns.composites_by_peak: {single card: [det class, min units in one frame, composite]}
    "det_shadow": {"Rage": 3.0},  # {card: s}: drop a detector-only play of card when another spell is this close in time
    "split_min_dets": 0,  # spawns._split: two cards in one group, each seen this often (0 = off), 2026-10-15
    "twin_new": False,  # a second unit of a card alive elsewhere at once is a new play (spawns.Deploy.twin, 2026-10-15)
    "mirror_repeat": False,  # the same card twice in a row -> the second is Mirror (2026-10-15)
    "mirror_min_votes": 0, "mirror_min_share": 0.0,  # ... only when both reads have this much card evidence
    "spell_unnamed_own": True,  # our unnamed plays explain a spell call (2026-10-15)
    "spell_ability_thr": 0.7,  # ability classes ("<card> ability") of the spell stage, 2026-10-15
    "cycle_guess": False,  # bot77.opponent: card for a clock with no recognisable unit
    "drop_cardless": True,  # applied after the deck: candidates with no deck card among their votes
    "timing_default": {"troop": 1.1, "building": 1.1, "spell": 0.1},
    # deploy clocks (bot77.clocks); applied in bot77.opponent before these rules
    "clock": {"on": True, "min_conf": 0.3, "min_dets": 2, "use_side": True, "side_margin": 1.0,
              "win_before_s": 0.5, "win_after_s": 2.5, "tiles": 3.0, "unit_lag_s": 1.0,
              "offset_s": 1.0,  # event time = clock start - this (median lag 0.85 / 1.10 s on the two dev halves)
              "drop_unclocked": False, "keep_unclocked": [], "drop_unclocked_repeat_s": 0.0,
              "new_from_clock": True, "vote_s": 3.0, "vote_conf": 0.3, "vote_tiles": 2.5,
              "keep_cardless": False,  # True + cycle_guess: +5 plays but +5 fakes/match
              "pov_window": None,  # [lo, hi]: drop clocks starting this soon after one of our plays (no effect)
              "min_y": 1.5,  # HUD trophy icon
              "unit_team_s": 0.0, "unit_team_min": 1.0},  # team from the landing units' badges: -8 plays
}
TIMING_FILE = Path("config/play_timing.json")
# Replays (TV Royale / Battle Log): the truth is read off the top player's hand, and the deploy clock starts at that
# moment (median +0.05 s) instead of ~1 s after it as in POV recordings; timing measured on replays (decisions.md,
# 2026-10-14). The own-deck rule only for cards we played recently: both replay players often run the same meta cards.
REPLAY_TIMING_FILE = Path("config/play_timing_replay.json")


REPLAY_HALF_MARGIN = 3.1
REPLAY_SPELL_PROBS = "data/models/spell_mix256/spell_probs"  # v2+v5+v6 (held-back 1215 -> 1224 right vs v2+v3, 2026-10-21)


def replay_rules(rules: dict | None) -> dict:
    t = json.loads(REPLAY_TIMING_FILE.read_text())
    r = dict(rules or {})
    r.setdefault("timing_file", str(REPLAY_TIMING_FILE))
    r.setdefault("timing_default", t["default"])
    r.setdefault("own_deck_recent_s", 10.0)
    r.setdefault("half_margin", REPLAY_HALF_MARGIN)
    r.setdefault("spell_probs", REPLAY_SPELL_PROBS)
    r["clock"] = {"offset_s": t["clock_offset_s"], **r.get("clock", {})}
    return r


def merged_rules(rules: dict | None) -> dict:
    r = {**DEFAULT_RULES, **(rules or {})}
    r["clock"] = {**DEFAULT_RULES["clock"], **(rules or {}).get("clock", {})}
    return r


def apply_rules(events: list, pov_plays: list[tuple[float, str]], ctype: dict[str, str], rules: dict | None = None) -> list:
    r = merged_rules(rules)
    events = sorted(events, key=lambda e: e.t)
    pov_deck = {c for _, c in pov_plays}
    kept = []
    for e in events:
        # hero ability summons
        if any(c in r["ability_classes"] for c in e.classes) and len(e.classes) == 1:
            e.card, e.card_votes = "__ability", {"__ability": sum(e.card_votes.values())}
            kept.append(e); continue
        card = e.card
        # continuation of a card still on the field
        win = r["continuation_s"].get(card)
        cc = r["continuation_classes"].get(card)
        if cc is not None and not set(e.classes) <= set(cc):
            win = None
        if win and not (r["twin_new"] and e.twin) and any(k.card == card and 0 < e.t - k.t <= win and
                       ((k.x - e.x) ** 2 + (k.y - e.y) ** 2) ** 0.5 <= r["continuation_tiles"] for k in kept):
            continue
        # long-lived spawners
        spawned = False
        for sp, (kids, sw, sr) in r["long_spawners"].items():
            if card in kids and any(k.card == sp and 0 < e.t - k.t <= sw and
                                    ((k.x - e.x) ** 2 + (k.y - e.y) ** 2) ** 0.5 <= sr for k in kept):
                spawned = True; break
        if spawned:
            continue
        # our own cards drifting near the river
        if card in pov_deck and e.y > r["own_deck_max_y"] and (
                r["own_deck_recent_s"] is None or any(c == card and 0 <= e.t - t <= r["own_deck_recent_s"] for t, c in pov_plays)):
            continue
        kept.append(e)
    if r["mirror_max_gap_s"] > 0:
        # the cycle: a played card comes back to hand only after 4 other plays, so the same card again with fewer
        # plays in between is Mirror (which copies the last card). With clocks, only a clocked repeat (a real new
        # placement, not the old unit walking back) counts.
        seq = [e for e in kept if e.card not in ("__ability", None)]
        room = True
        if r["mirror_deck_full"]:
            w = Counter()
            for e in seq:
                if not e.subunit_only:
                    clocked = getattr(e, "clock_t", None) is not None or ctype.get(e.card) == "spell"
                    w[e.card] += 1.0 if clocked else r["deck_w_unclocked"]
            if sum(1 for c, v in w.items() if c != "Mirror" and v >= r["mirror_deck_min_w"]) >= r["mirror_deck_full"]:
                room = False  # eight cards already: no room for Mirror
        dup = set()
        for i, e in enumerate(seq):
            prev = next((j for j in range(i - 1, -1, -1) if seq[j].card == e.card), None)
            if prev is None or e.t - seq[prev].t > r["mirror_max_gap_s"] or i - prev - 1 >= r["mirror_max_between"]:
                continue
            twin = r["twin_new"] and e.twin  # two copies seen at once: a new placement, so by the cycle a Mirror
            if not twin and (not room or (r["mirror_need_clock"] and getattr(e, "clock_t", None) is None)):
                dup.add(id(e))
                continue
            e.card, e.card_votes = "Mirror", {"Mirror": sum(e.card_votes.values())}
        if r["mirror_drop_rest"]:
            kept = [e for e in kept if id(e) not in dup]
    if r["timing"]:
        tf = Path(r["timing_file"]) if r["timing_file"] else TIMING_FILE
        table = json.loads(tf.read_text()) if tf.exists() else {}
        table = table.get("cards", table)  # the replay file nests the per-card table
        for e in kept:
            e.t_seen = e.t  # first unit seen; scoring against labels made at unit appearance uses this
            if getattr(e, "clock_t", None) is not None and r["clock"]["on"]:
                e.t = e.clock_t - r["clock"]["offset_s"]
            elif e.card and e.card != "__ability":
                e.t -= table.get(e.card, r["timing_default"].get(ctype.get(e.card, "troop"), 1.0))
    return kept
