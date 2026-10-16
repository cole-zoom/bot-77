"""Spell stage (decisions.md, 2026-10-14): opponent spell plays from a clip classifier, merged with the troop plays.

The detector looks at single frames 4 times a second and never learned Zap or Arrows (0 of 43 seen on the replay test
set, even at 15 fps). A video classifier (R(2+1)D-18 over 1.6 s of the arena at 10 fps, scripts/katacr/train_spell.py)
scores a window every 0.3 s; this module turns those probabilities into spell plays and merges them:

- peaks: the best spell class per window above `thr`, grouped over `group_s`, one play per group at its peak;
- our own spells (the POV hand log) are removed: the classifier sees both players' spells;
- detector spell plays near a classifier spell play are replaced by it; units a barrel spawns are dropped;
- the merged list then goes through deck inference like any other play.

Abilities (2026-10-15): classes named "<card> ability" (champion / hero abilities: Archer Queen's cloak, Golden
Knight's dash, ...) become "__ability" plays, kept only when the opponent played that card earlier and we didn't use
the same ability then. They stay out of deck inference (opponent.py sets them aside).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bot77.spawns import Deploy

PLAY_OFFSET_S = 0.1  # a window starting at s means the card left the hand near s + 0.1 (train_spell.POS_OFF)
ABILITY = " ability"  # class-name suffix for ability classes
SPAWNS = {"Barbarian Barrel": {"Barbarians"}, "Goblin Barrel": {"Goblins"}, "Royal Delivery": {"Royal Recruits"}}


def load_probs(probs_dir: str | Path, match_id: str):
    f = Path(probs_dir) / f"{match_id}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    return z["t"], z["p"], [str(c) for c in z["classes"]]


def spell_peaks(t: np.ndarray, p: np.ndarray, classes: list[str], thr: float = 0.5, group_s: float = 1.2,
                class_thr: dict[str, float] | None = None) -> list[tuple[float, str, float]]:
    """(play time, spell, peak prob) per group of windows whose best spell clears its threshold."""
    sp = p[:, 1:]  # column 0 is "__none"
    best = sp.argmax(1); bp = sp.max(1)
    hits = [(float(t[i]), classes[1 + best[i]], float(bp[i])) for i in range(len(t))
            if bp[i] >= (class_thr or {}).get(classes[1 + best[i]], thr)]
    out: list[tuple[float, str, float]] = []
    for h in hits:
        if out and h[1] == out[-1][1] and h[0] - out[-1][3] <= group_s:
            g = out[-1]
            out[-1] = (h[0], h[1], h[2], h[0]) if h[2] > g[2] else (g[0], g[1], g[2], h[0])
        else:
            out.append((h[0], h[1], h[2], h[0]))
    return [(s + PLAY_OFFSET_S, c, pk) for s, c, pk, _ in out]


def merge_spells(enemy: list, peaks: list[tuple[float, str, float]], pov: list[tuple[float, str]], is_spell,
                 own_window: tuple[float, float] = (-1.0, 2.0), replace_s: float = 1.5, spawn_s: float = 4.0,
                 any_own: bool = False, own_abilities: list[tuple[float, str]] = (), ability_thr: float = 0.7,
                 unnamed_own: list[float] = ()) -> list:
    """Add classifier spell (and ability) plays to the troop plays (Deploy list); see the module docstring.
    `own_abilities`: (t, card) of our ability uses."""
    abil = [(t, c[:-len(ABILITY)], pk) for t, c, pk in peaks if c.endswith(ABILITY) and pk >= ability_thr]
    peaks = [x for x in peaks if not x[1].endswith(ABILITY)]
    abil = [(t, c, pk) for t, c, pk in abil
            if any(e.card == c and e.t < t for e in enemy)  # their champion / hero is (or was) on the field
            and not any(c == mc and own_window[0] <= t - mt <= own_window[1] for mt, mc in own_abilities)]
    # the classifier sees our spells too. any_own: any of our spells (or Mirror) at that moment explains the call, even
    # under another name (our Vines read as Poison). Dev +0.2, held-back replays -1.0 (both sides play the same spells
    # together there), so off by default (2026-10-15).
    mine = [(t, c) for t, c in pov if c and (is_spell(c) or c == "Mirror")]
    keep = [(t, c, pk) for t, c, pk in peaks
            if not any((any_own or c == mc) and own_window[0] <= t - mt <= own_window[1] for mt, mc in mine)]
    # unnamed_own: times of our plays the hand reader couldn't name (our Lightning in replays, 2026-10-15) explains a call
    keep = [(t, c, pk) for t, c, pk in keep if not any(own_window[0] <= t - mt <= own_window[1] for mt in unnamed_own)]
    out = []
    for e in enemy:
        if e.card and is_spell(e.card) and any(abs(e.t - t) <= replace_s for t, _, _ in keep):
            continue  # the classifier's call replaces the detector's
        if any(e.card in SPAWNS.get(c, ()) and 0 <= e.t - t <= spawn_s for t, c, _ in keep):
            continue  # the barrel's troops, not a play
        out.append(e)
    for t, c, pk in keep:
        d = Deploy("enemy", t, 9.0, 8.0, c, {c: max(1, int(round(10 * pk)))}, 1, False, False, ["spell-stage"])
        d.t_seen = t
        out.append(d)
    for t, c, pk in abil:
        d = Deploy("enemy", t, 9.0, 8.0, "__ability", {"__ability": max(1, int(round(10 * pk)))}, 1, False, False,
                   ["spell-stage", f"{c}{ABILITY}"])
        d.t_seen = t
        out.append(d)
    return sorted(out, key=lambda e: e.t)
