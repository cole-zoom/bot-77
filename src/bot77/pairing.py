"""Pair two recordings of the same friendly battle (both players recording) and turn each side's
POV plays into exact opponent labels for the other.

The match clock runs at real speed in every game, so ANY two overlapping matches line up at some
constant offset — the clock alone can't pair them (first attempt, 2026-09-29). The recordings' wall-clock
start times fix it: the same game must sit at offset = start_A - start_B (plus a little clock skew). Then every POV play in B is an
opponent play in A at t_A = t_B - offset (and vice versa). The POV detector is 100% on 490 labelled
plays (decisions.md), so these labels need no human review.
"""

from __future__ import annotations

import json
from pathlib import Path

import lancedb
import numpy as np

MAX_SPREAD_S = 1.5  # clock-derived offsets within a true pair agree to about this
MAX_SKEW_S = 20  # the two devices' wall clocks / recording starts may differ by this much
MIN_COMMON_S = 30  # the two matches must share at least this much match time
NEXT_HOLD_FRAMES = 3  # a "next" card counts once read this many frames in a row
NEXT_MATCH_S = (-0.8, 1.5)  # a next-card advance belongs to a logged play this close before / after it
CYCLE_AHEAD = 3  # 8-card cycle: the card played at one advance is shown as "next" 3 advances later (91% on 400 replays)


def _clock(db, match_id: str) -> dict[int, float]:
    """match second -> first video time it was on screen."""
    rows = db.open_table("hud_states").search().where(f"match_id = '{match_id}'").limit(10**7) \
        .select(["t_video", "t_match"]).to_arrow().to_pylist()
    out: dict[int, float] = {}
    for r in sorted(rows, key=lambda r: r["t_video"]):
        if r["t_match"] is not None:
            out.setdefault(int(r["t_match"]), r["t_video"])
    return out


def pair_matches(video_a: str, video_b: str, start_offset_s: float, db_dir: Path = Path("data/lancedb")) -> list[dict]:
    """`start_offset_s`: recording start of A minus start of B, in seconds of wall-clock time."""
    db = lancedb.connect(db_dir)
    ms = lambda v: db.open_table("matches").search().where(f"video_id = '{v}'").to_arrow().to_pylist()
    pairs = []
    for ma in ms(video_a):
        ca = _clock(db, ma["match_id"])
        best = None
        for mb in ms(video_b):
            cb = _clock(db, mb["match_id"])
            common = sorted(set(ca) & set(cb))
            if len(common) < MIN_COMMON_S:
                continue
            offs = np.array([cb[e] - ca[e] for e in common])
            med = float(np.median(offs))
            spread = float(np.median(np.abs(offs - med)))
            if (spread <= MAX_SPREAD_S and abs(med - start_offset_s) <= MAX_SKEW_S
                    and (best is None or abs(med - start_offset_s) < abs(best["offset_s"] - start_offset_s))):
                best = {"a": ma["match_id"], "b": mb["match_id"], "offset_s": med, "spread_s": spread, "common_s": len(common)}
        if best:
            pairs.append(best)
    return pairs


def next_card_plays(db, match_id: str, plays: list[dict], names: dict[int, str]) -> list[dict]:
    """Plays the POV detector missed, found from the "next" card advancing (decisions.md, 2026-10-15). Two cards
    played ~0.3 s apart can log as one play (Bandit then Wall Breakers, tv_2026-10-12_r46 top at 11.9 s: the Bandit's
    slot never read as a card), but "next" still advances once per card. An advance with no logged play near it is a
    play; its card is the one shown as "next" CYCLE_AHEAD advances later, named only when the logged plays on both
    sides agree with that rule there (so the advance sequence is aligned). The same rule names logged plays the hand
    couldn't name (card None; those carry "names_play", the index into `plays`). `plays`: logged play rows (t_video,
    card)."""
    rows = db.open_table("hud_states").search().where(f"match_id = '{match_id}'").limit(10**7) \
        .select(["t_video", "next"]).to_arrow().to_pylist()
    seq, cur, cand, run, tf = [], None, None, 0, None  # (time of the advance, new next card)
    for r in sorted(rows, key=lambda r: r["t_video"]):
        n = r["next"] if r["next"] is not None and r["next"] >= 0 else None
        if n is None:
            continue
        if n == cand:
            run += 1
        else:
            cand, run, tf = n, 1, r["t_video"]
        if run == NEXT_HOLD_FRAMES and n != cur:
            seq.append((tf, n)); cur = n
    used, owner = set(), {}  # advance index -> logged play index
    for k in range(1, len(seq)):
        c = [(abs(seq[k][0] - p["t_video"]), j) for j, p in enumerate(plays)
             if j not in used and NEXT_MATCH_S[0] <= seq[k][0] - p["t_video"] <= NEXT_MATCH_S[1]]
        if c:
            j = min(c)[1]; used.add(j); owner[k] = j
    later = lambda k: names.get(seq[k + CYCLE_AHEAD][1]) if k + CYCLE_AHEAD < len(seq) else None
    agrees = lambda k: k in owner and plays[owner[k]]["card"] is not None and later(k) == plays[owner[k]]["card"]
    out = []
    for k in range(1, len(seq)):
        if k in owner and plays[owner[k]]["card"] is None and later(k) and agrees(k - 1) and agrees(k + 1):
            out.append({"t_video": round(plays[owner[k]]["t_video"], 2), "card": later(k), "names_play": owner[k]})
            continue  # a logged play the hand couldn't name
        if k in owner or not later(k):
            continue
        if agrees(k - 1) and agrees(k + 1):
            out.append({"t_video": round(seq[k][0], 2), "card": later(k)})
            continue
        # the burst's logged play took the other card's advance (r46: Wall Breakers logged at the Bandit's advance)
        for n, outer, other in ((k - 1, k - 2, k + 1), (k + 1, k + 2, k - 1)):
            if (n in owner and plays[owner[n]]["card"] == later(k) and later(n) and later(n) != later(k)
                    and agrees(outer) and agrees(other)):
                out.append({"t_video": round(seq[min(n, k)][0], 2) if n < k else round(seq[k][0], 2), "card": later(n)})
                break
    return out


def opponent_truth_from_pair(pair: dict, db_dir: Path = Path("data/lancedb")) -> tuple[dict, dict]:
    """(truth for A's opponent, truth for B's opponent), in the opponent-truth format."""
    db = lancedb.connect(db_dir)

    def pov(mid):
        return db.open_table("events").search().where(f"match_id = '{mid}'").limit(10**5) \
            .select(["t_video", "kind", "card", "form", "confidence"]).to_arrow().to_pylist()

    names = {c["id"]: c["name"] for c in db.open_table("cards").search().limit(5000).select(["id", "name"]).to_arrow().to_pylist()}

    def build(target_mid, source_mid, sign):
        vid = target_mid.rsplit("_m", 1)[0]
        items = []
        rows = pov(source_mid)
        for x in next_card_plays(db, source_mid, [e for e in rows if e["kind"] == "play"], names):
            items.append({"match_id": target_mid, "t_video": round(x["t_video"] - sign * pair["offset_s"], 2),
                          "label": x["card"], "form": None, "source": f"next-card advance in {source_mid}",
                          "source_confidence": "medium"})
        for e in rows:
            label = "__ability" if e["kind"] == "ability" else e["card"]
            if not label:
                continue
            items.append({"match_id": target_mid, "t_video": round(e["t_video"] - sign * pair["offset_s"], 2),
                          "label": label, "form": e["form"], "source": f"POV of {source_mid}",
                          "source_confidence": e["confidence"]})
        items.sort(key=lambda x: x["t_video"])
        return {"video_id": vid, "side": "opponent", "matches": [target_mid], "items": items,
                "paired_with": source_mid, "offset_s": pair["offset_s"], "auto": True}

    return build(pair["a"], pair["b"], 1), build(pair["b"], pair["a"], -1)


def write_pair_truth(video_a: str, video_b: str, start_offset_s: float, out_dir: Path = Path("data/ground_truth"),
                     db_dir: Path = Path("data/lancedb"), log=print) -> list[dict]:
    pairs = pair_matches(video_a, video_b, start_offset_s, db_dir)
    for p in pairs:
        ta, tb = opponent_truth_from_pair(p, db_dir)
        for t in (ta, tb):
            mid = t["matches"][0]
            (out_dir / f"opponent_truth_auto_{mid}.json").write_text(json.dumps(t, indent=1))
        log(f"{p['a']} <-> {p['b']}: offset {p['offset_s']:.2f}s (spread {p['spread_s']:.2f}), "
            f"{p['common_s']} s in common, {len(ta['items'])} / {len(tb['items'])} opponent plays")
    return pairs


def write_replay_truth(video_bottom: str, video_top: str, out_dir: Path = Path("data/ground_truth"),
                       db_dir: Path = Path("data/lancedb"), log=print) -> list[Path]:
    """Opponent truth from a replay recording (decisions.md, 2026-10-12): the same file processed twice, with the
    bottom hand (layout replay_bottom, video `video_bottom`) and the top hand (replay_top, `video_top`). The top
    player's plays are the bottom player's opponent truth, at offset 0 (same video). Written as
    opponent_truth_replay_<match>.json so the paired dev set (opponent_truth_auto_*) stays fixed."""
    db = lancedb.connect(db_dir)
    mids = lambda v: sorted(r["match_id"] for r in db.open_table("matches").search().where(f"video_id = '{v}'")
                            .limit(100).select(["match_id"]).to_arrow().to_pylist())
    bottom, top = mids(video_bottom), mids(video_top)
    out = []
    for a, b in zip(bottom, top):
        truth, _ = opponent_truth_from_pair({"a": a, "b": b, "offset_s": 0.0}, db_dir)
        truth["source"] = "replay"
        path = out_dir / f"opponent_truth_replay_{a}.json"
        path.write_text(json.dumps(truth, indent=1))
        log(f"{a}: {len(truth['items'])} opponent plays from {b}")
        out.append(path)
    if len(bottom) != len(top):
        log(f"warning: {video_bottom} has {len(bottom)} matches, {video_top} has {len(top)}")
    return out


def _align(pa: list, pb: list) -> list[tuple[float, float]]:
    """In-order card alignment of two play lists [(t, card)] as (t_a, t_b) anchors: the longest common subsequence, so
    a play missing on one side (a replay recording that started late, 2026-10-20) is skipped instead of pulling the
    match onto a later copy of the same card the way a greedy scan does."""
    n, m = len(pa), len(pb)
    L = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            L[i][j] = L[i + 1][j + 1] + 1 if pa[i][1] == pb[j][1] else max(L[i + 1][j], L[i][j + 1])
    out, i, j = [], 0, 0
    while i < n and j < m:
        if pa[i][1] == pb[j][1]:
            out.append((pa[i][0], pb[j][0])); i += 1; j += 1
        elif L[i + 1][j] >= L[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


LIVE_MATCH_S = 1.5  # a live POV play and the replay's bottom-hand play of it agree to this
LIVE_MIN_AGREE, LIVE_MIN_PLAYS = 0.7, 5


def pair_live_replay(live_video: str, replay_video: str, db_dir: Path = Path("data/lancedb")) -> list[dict]:
    """Pair Cole's live recordings with the in-game replays he records after each game (2026-10-16).

    The wall clock can't pair them (the replay is recorded later) and the match clock alone can't either (any two
    matches line up at some offset). The game itself does: the replay's bottom hand is Cole's, so his live POV plays
    must reappear as the replay's bottom-hand plays at the clock offset. A pair needs >= LIVE_MIN_AGREE of the live
    plays (>= LIVE_MIN_PLAYS) matched by card within LIVE_MATCH_S. `replay_video`: the replay processed with layout
    replay_bottom. Returns [{"a": live match, "b": replay match, "offset_s", "agree", "n"}]."""
    db = lancedb.connect(db_dir)
    ms = lambda v: db.open_table("matches").search().where(f"video_id = '{v}'").to_arrow().to_pylist()
    plays = lambda mid: [(e["t_video"], e["card"]) for e in db.open_table("events").search()
                         .where(f"match_id = '{mid}' AND kind = 'play'").limit(10**4).select(["t_video", "card"])
                         .to_arrow().to_pylist() if e["card"]]
    pairs = []
    for ma in ms(live_video):
        ca, pa = _clock(db, ma["match_id"]), plays(ma["match_id"])
        if len(pa) < LIVE_MIN_PLAYS:
            continue
        best = None
        for mb in ms(replay_video):
            cb = _clock(db, mb["match_id"])
            common = sorted(set(ca) & set(cb))
            if len(common) < MIN_COMMON_S:
                continue
            off = float(np.median([cb[e] - ca[e] for e in common]))
            pb = plays(mb["match_id"])
            # the clock shows whole seconds, so its offset is off by up to ~2 s (2026-10-17: 4 of 20 games missed the
            # 1.5 s window by that); refine it on the plays themselves, which also makes the key's times exact
            score = lambda o: sum(1 for t, c in pa if any(c2 == c and abs((t2 - o) - t) <= 0.5 for t2, c2 in pb))
            # the clock can be off by more than that (2026-10-20: overtime replays, same play sequence but < 30%
            # agreement): also try offsets implied by the plays themselves (each early live play vs each replay
            # play of the same card) and keep whichever lines up the most plays
            seeds = [off] + [t2 - t for t, c in pa[:6] for t2, c2 in pb if c2 == c]
            best_seed = max(seeds, key=lambda s: (score(s), -abs(s - off)))
            off = max((best_seed + d / 10 for d in range(-30, 31)), key=lambda o: (score(o), -abs(o - best_seed)))
            hit = sum(1 for t, c in pa if any(c2 == c and abs((t2 - off) - t) <= LIVE_MATCH_S for t2, c2 in pb))
            agree, anchors = hit / len(pa), None
            if agree < LIVE_MIN_AGREE:
                # the live recording can lose seconds mid-game (iPhone Mirroring stalls; 2026-10-20: the offset
                # drifted 8 s over one match in small jumps), so no single offset fits: line the plays up in order
                # by card and map times piecewise between those anchors
                anchors = _align(sorted(pa), sorted(pb))
                gaps = [b - a for a, b in anchors]
                steps = [abs(g2 - g1) for g1, g2 in zip(gaps, gaps[1:])]
                # a different game with the same deck also lines up by card order, but its gaps jump all the time; a
                # stalled recording of the same game holds a steady gap with rare jumps (2026-10-20: true pair 30/32
                # plays aligned, median step 0.7 s, 76% of steps <= 2 s, max 8.3 s; five false pairs <= 72% aligned,
                # median 4.3-9.0 s, 14-29% <= 2 s, max 18-34 s)
                smooth = len(anchors) >= max(3, 0.85 * len(pa)) and len(steps) >= 2 and \
                    float(np.median(steps)) <= 1.5 and sum(x <= 2 for x in steps) >= 0.6 * len(steps) and \
                    max(steps) <= 15 and abs(float(np.median(gaps)) - off) <= 40
                if smooth:
                    to_live = lambda tb: float(np.interp(tb, [b for _, b in anchors], [a for a, _ in anchors]))
                    hit = sum(1 for t, c in pa if any(c2 == c and abs(to_live(t2) - t) <= LIVE_MATCH_S for t2, c2 in pb))
                    agree = hit / len(pa)
            if agree >= LIVE_MIN_AGREE and (best is None or agree > best["agree"]):
                best = {"a": ma["match_id"], "b": mb["match_id"], "offset_s": off, "agree": round(agree, 3), "n": len(pa),
                        "anchors": anchors if anchors and agree >= LIVE_MIN_AGREE and hit / len(pa) == agree else None}
        if best:
            pairs.append(best)
    return pairs


def write_live_pair(p: dict, replay_top_video: str, out_dir: Path = Path("data/ground_truth"),
                    db_dir: Path = Path("data/lancedb"), log=print) -> Path | None:
    """Write the answer key for one live match from its paired replay (a pair_live_replay result)."""
    db = lancedb.connect(db_dir)
    top = {r["match_id"].rsplit("_m", 1)[1]: r["match_id"] for r in db.open_table("matches").search()
           .where(f"video_id = '{replay_top_video}'").limit(100).select(["match_id"]).to_arrow().to_pylist()}
    b_top = top.get(p["b"].rsplit("_m", 1)[1])
    if not b_top:
        log(f"{p['b']}: no top-hand match in {replay_top_video}")
        return None
    if p.get("anchors"):  # piecewise time map (the live recording lost time mid-game)
        truth, _ = opponent_truth_from_pair({"a": p["a"], "b": b_top, "offset_s": 0.0}, db_dir)
        ab, aa = [b for _, b in p["anchors"]], [a for a, _ in p["anchors"]]
        for i in truth["items"]:
            i["t_video"] = round(float(np.interp(i["t_video"], ab, aa)), 2)
        truth["time_map"] = p["anchors"]
    else:
        truth, _ = opponent_truth_from_pair({"a": p["a"], "b": b_top, "offset_s": p["offset_s"]}, db_dir)
    truth.update(source="live+replay", replay=p["b"], agree=p["agree"])
    path = out_dir / f"opponent_truth_live_{p['a']}.json"
    path.write_text(json.dumps(truth, indent=1))
    how = "piecewise time map" if p.get("anchors") else f"offset {p['offset_s']:.2f} s"
    log(f"{p['a']} <-> {p['b']}: {how}, {p['agree']:.0%} of {p['n']} own plays agree, {len(truth['items'])} opponent plays")
    return path


def write_live_truth(live_video: str, replay_video: str, replay_top_video: str, out_dir: Path = Path("data/ground_truth"),
                     db_dir: Path = Path("data/lancedb"), log=print) -> list[Path]:
    """Opponent truth for a live recording from its replay: the top player's plays (the replay processed with
    replay_top, `replay_top_video`) moved onto the live video's time line. Written as opponent_truth_live_<match>.json
    (its own set: eval_dev.py --set live), so the dev set stays fixed."""
    db = lancedb.connect(db_dir)
    top = {r["match_id"].rsplit("_m", 1)[1]: r["match_id"] for r in db.open_table("matches").search()
           .where(f"video_id = '{replay_top_video}'").limit(100).select(["match_id"]).to_arrow().to_pylist()}
    out = []
    for p in pair_live_replay(live_video, replay_video, db_dir):
        b_top = top.get(p["b"].rsplit("_m", 1)[1])
        if not b_top:
            log(f"{p['b']}: no top-hand match in {replay_top_video}"); continue
        if p.get("anchors"):  # piecewise time map (the live recording lost time mid-game)
            truth, _ = opponent_truth_from_pair({"a": p["a"], "b": b_top, "offset_s": 0.0}, db_dir)
            ab, aa = [b for _, b in p["anchors"]], [a for a, _ in p["anchors"]]
            for i in truth["items"]:
                i["t_video"] = round(float(np.interp(i["t_video"], ab, aa)), 2)
            truth["time_map"] = p["anchors"]
        else:
            truth, _ = opponent_truth_from_pair({"a": p["a"], "b": b_top, "offset_s": p["offset_s"]}, db_dir)
        truth.update(source="live+replay", replay=p["b"], agree=p["agree"])
        path = out_dir / f"opponent_truth_live_{p['a']}.json"
        path.write_text(json.dumps(truth, indent=1))
        log(f"{p['a']} <-> {p['b']}: offset {p['offset_s']:.2f} s, {p['agree']:.0%} of {p['n']} own plays agree, "
            f"{len(truth['items'])} opponent plays")
        out.append(path)
    return out
