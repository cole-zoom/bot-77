"""The playbook (step 1, 2026-10-21): how strong players answer each card, mined from keyed replays.

interactions: one row per (play by one player, the other player's next play within ANSWER_S): the cards, the delay,
where each was placed and the answer's position relative to the play, in the answering player's frame (y grows toward
their own king, x mirrored for the top player so "left" is the same lane on both screens).
playbook: interactions aggregated per opponent card (and its zone): the answers, their share, median delay and offset.

Positions: each hand-log play (exact card and time) is matched to the first unit of that card C detected for that
player just after it (bot77.spawns.deploys over the replay's detections); spells use the detection too.
"""
from __future__ import annotations

ANSWER_S = 8.0
MATCH_WINDOW = (-0.5, 4.0)  # unit first seen this long after the play
RIVER_Y = 14.9


def placed(plays: list[dict], deploys: list, team: str) -> list[dict]:
    """Attach (x, y) to each play from the nearest later deploy of the same card by this team."""
    used, out = set(), []
    for p in plays:
        best = None
        for i, d in enumerate(deploys):
            if i in used or d.team != team or d.card != p["card"]:
                continue
            dt = d.t - p["t"]
            if MATCH_WINDOW[0] <= dt <= MATCH_WINDOW[1] and (best is None or dt < best[0]):
                best = (dt, i)
        if best:
            used.add(best[1]); d = deploys[best[1]]
            out.append({**p, "x": d.x, "y": d.y})
        else:
            out.append({**p, "x": None, "y": None})
    return out


def to_frame(x: float | None, y: float | None, side: str) -> tuple[float | None, float | None]:
    """A position in the answering player's frame: the bottom player's screen as is, the top player's turned around
    (so y grows toward the answerer's king and x is the answerer's left-right)."""
    if x is None:
        return None, None
    return (x, y) if side == "bottom" else (18.0 - x, 2 * RIVER_Y - y)


def zone(y: float | None) -> str:
    """Where a play landed, in its answerer's frame: their half, the bridge, or deep on the other side."""
    if y is None:
        return "unknown"
    return "our_half" if y > RIVER_Y + 1.5 else "bridge" if y > RIVER_Y - 3.0 else "their_half"


def interactions(bottom: list[dict], top: list[dict]) -> list[dict]:
    """bottom / top: placed plays of each player. One row per play and the other player's next play (<= ANSWER_S)."""
    rows = []
    for side_o, plays_o, side_a, plays_a in (("top", top, "bottom", bottom), ("bottom", bottom, "top", top)):
        for p in plays_o:
            nxt = next((q for q in plays_a if 0 < q["t"] - p["t"] <= ANSWER_S), None)
            if nxt is None:
                continue
            ox, oy = to_frame(p["x"], p["y"], side_a)
            ax, ay = to_frame(nxt["x"], nxt["y"], side_a)
            rows.append({"opp_card": p["card"], "opp_side": side_o, "opp_x": ox, "opp_y": oy, "opp_zone": zone(oy),
                         "opp_lane": None if ox is None else ("left" if ox < 9 else "right"),
                         "answer": nxt["card"], "delay_s": nxt["t"] - p["t"], "ans_x": ax, "ans_y": ay,
                         "dx": None if ax is None or ox is None else ax - ox,
                         "dy": None if ay is None or oy is None else ay - oy,
                         "same_lane": None if ax is None or ox is None else (ax < 9) == (ox < 9)})
    return rows
