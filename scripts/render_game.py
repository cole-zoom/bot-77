"""Watch the pipeline on one game: the video with live detector boxes, both hands as read, and a running
play-by-play of opponent plays (truth vs what we detected) beside it. Stays local (game frames, gitignored).

  uv run python scripts/render_game.py <match_id> --weights <best.pt> --events <opponent_events table>
      [--fps 15] [--out data/reviews/playback_<match>.mp4]
"""
import argparse, json, subprocess
from pathlib import Path

import cv2
import lancedb
import numpy as np

from bot77.detect.yolo import Bot77Detector
from bot77.layout import Layout
from bot77.review import MATCH_WINDOW_S
from bot77.video import iter_frames, probe

PANEL_W = 820
GREEN, ORANGE, RED, GREY, BLUE, WHITE = (80, 200, 80), (0, 165, 255), (60, 60, 230), (150, 150, 150), (230, 160, 60), (240, 240, 240)


def match_plays(truth, events):
    """Same one-to-one matching as bot77.review.score_opponent: card matches first, then any card."""
    used_t, used_e, pairs = set(), set(), {}
    for exact in (True, False):
        for i, e in sorted(enumerate(events), key=lambda ie: ie[1]["t_video"]):
            if i in used_e:
                continue
            c = [(abs(t["t_video"] - e["t_video"]), k) for k, t in enumerate(truth) if k not in used_t
                 and abs(t["t_video"] - e["t_video"]) <= MATCH_WINDOW_S and (t["label"] == e["card"]) == exact]
            if c:
                _, k = min(c); used_t.add(k); used_e.add(i); pairs[i] = k
    rows = []
    for k, t in enumerate(truth):
        i = next((i for i, kk in pairs.items() if kk == k), None)
        if i is None:
            rows.append((t["t_video"], "MISSED", t["label"], None, GREY))
        else:
            e = events[i]
            ok = e["card"] == t["label"]
            rows.append((t["t_video"], "RIGHT" if ok else "WRONG", t["label"], e, GREEN if ok else ORANGE))
    rows += [(e["t_video"], "FAKE", e["card"], e, RED) for i, e in enumerate(events) if i not in pairs]
    return sorted(rows, key=lambda r: r[0])


def hand_text(db, match_id, t, names):
    r = db.open_table("hud_states").search().where(f"match_id = '{match_id}' AND t_video <= {t:.2f} AND t_video > {t - 1.0:.2f}") \
        .limit(20).to_arrow().to_pylist()
    if not r:
        return None
    r = max(r, key=lambda x: x["t_video"])
    cards = []
    for cid, st, g in zip(r["hand"], r["hand_state"], r["greyed"]):
        n = names.get(cid, "?") if cid >= 0 else st
        cards.append(n + (" (grey)" if g else ""))
    return cards, names.get(r["next"], "?") if r["next"] >= 0 else "-", r["elixir"]


def put(img, txt, xy, col=WHITE, s=0.62, th=1, outline=True):
    if outline:
        cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, s, col, th, cv2.LINE_AA)


def putp(img, txt, xy, col=WHITE, s=0.62, th=1):
    put(img, txt, xy, col, s, th, outline=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("match_id"); ap.add_argument("--weights", required=True); ap.add_argument("--events", required=True)
    ap.add_argument("--fps", type=float, default=15); ap.add_argument("--out", default=None)
    ap.add_argument("--truth", default=None); ap.add_argument("--dur", type=float, default=None); ap.add_argument("--top-match", default=None)
    a = ap.parse_args()
    db = lancedb.connect("data/lancedb")
    names = {c["id"]: c["name"] for c in db.open_table("cards").search().limit(5000).to_arrow().to_pylist()}
    m = db.open_table("matches").search().where(f"match_id = '{a.match_id}'").to_arrow().to_pylist()[0]
    v = db.open_table("videos").search().where(f"video_id = '{m['video_id']}'").to_arrow().to_pylist()[0]
    lay = Layout.load(v["layout_id"])
    tf = a.truth or sorted(Path("data/ground_truth").glob(f"opponent_truth_*{a.match_id}.json"))[0]
    truth = [i for i in json.loads(Path(tf).read_text())["items"] if i["match_id"] == a.match_id]
    events = db.open_table(a.events).search().where(f"match_id = '{a.match_id}'").limit(10**5) \
        .select(["t_video", "card", "classes"]).to_arrow().to_pylist()
    rows = match_plays(truth, events)
    ours = [(e["t_video"], e["card"]) for e in db.open_table("events").search().where(
        f"match_id = '{a.match_id}' AND kind = 'play'").limit(10**4).select(["t_video", "card"]).to_arrow().to_pylist()]
    top = a.top_match or (f"{m['video_id']}_top_m{a.match_id.rsplit('_m', 1)[1]}" if m["video_id"].startswith(("tv_", "bl_")) else None)
    det = Bot77Detector(Path(a.weights))
    info = probe(Path(v["path"]))
    s = 1000 / info["height"]  # video scaled to 1000 px high
    vw, vh = int(info["width"] * s) // 2 * 2, 1000
    out = Path(a.out or f"data/reviews/playback_{a.match_id}.mp4"); out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{vw + PANEL_W}x{vh}",
                            "-r", str(a.fps), "-i", "-", "-ss", str(m["t_start"]), "-t", str(a.dur or m["t_end"] - m["t_start"]),
                            "-i", v["path"], "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "22",
                            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)], stdin=subprocess.PIPE)
    tot = {k: sum(r[1] == k for r in rows) for k in ("RIGHT", "WRONG", "FAKE", "MISSED")}
    for t, f in iter_frames(Path(v["path"]), a.fps, start=m["t_start"], duration=a.dur or m["t_end"] - m["t_start"]):
        img = cv2.resize(f, (vw, vh), interpolation=cv2.INTER_AREA)
        for d in det.detect(f, lay):
            if d.name.endswith("bar") or d.name.startswith("bar"):
                continue
            x0, y0, x1, y1 = (int(c * s) for c in d.box)
            col = RED if d.team == "enemy" else BLUE
            if "tower" in d.name:  # towers: thin box, no label
                cv2.rectangle(img, (x0, y0), (x1, y1), col, 1)
                continue
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            put(img, f"{d.name} {d.conf:.2f}", (x0, max(y0 - 4, 12)), col, 0.42)
        # flash the current opponent play on the video
        for tt, kind, card, e, col in rows:
            if 0 <= t - tt <= 1.5:
                put(img, f"{kind}: {card}" + (f" (we said {e['card']})" if kind == "WRONG" else ""), (10, vh // 2), col, 0.8, 2)
        panel = np.full((vh, PANEL_W, 3), 24, np.uint8)
        putp(panel, f"{a.match_id}   t={t:6.1f}s", (14, 30), WHITE, 0.65)
        putp(panel, f"opponent plays: right {tot['RIGHT']}  wrong {tot['WRONG']}  fake {tot['FAKE']}  missed {tot['MISSED']}",
            (14, 58), WHITE, 0.5)
        y = 92
        for label, mid in (("OPPONENT hand (top)", top), ("OUR hand (bottom)", a.match_id)):
            h = hand_text(db, mid, t, names) if mid else None
            putp(panel, label, (14, y), WHITE, 0.55); y += 24
            if h:
                cards, nxt, el = h
                putp(panel, " | ".join(cards), (14, y), (200, 220, 255), 0.45); y += 22
                putp(panel, f"next: {nxt}   elixir: {el:.1f}", (14, y), (200, 220, 255), 0.45); y += 30
            else:
                putp(panel, "(not read)", (14, y), GREY, 0.45); y += 30
        putp(panel, "play by play (opponent; blue = ours)", (14, y), WHITE, 0.55); y += 8
        cv2.line(panel, (14, y), (PANEL_W - 14, y), GREY, 1); y += 24
        feed = [(r[0], r) for r in rows] + [(tt, (tt, "OURS", c, None, BLUE)) for tt, c in ours if c]
        feed = sorted([x for x in feed if x[0] <= t + 0.01], key=lambda x: x[0])[-26:]
        for _, (tt, kind, card, e, col) in feed:
            mm, ss = divmod(tt - m["t_start"], 60)
            extra = ""
            if kind == "WRONG":
                extra = f" -> we said {e['card']}"
            elif kind in ("RIGHT",) and e is not None:
                extra = f"  ({e['t_video'] - tt:+.1f}s)" + ("  [spell stage]" if "spell-stage" in (e.get("classes") or []) else "")
            elif kind == "FAKE" and e is not None and "spell-stage" in (e.get("classes") or []):
                extra = "  [spell stage]"
            putp(panel, f"{int(mm)}:{ss:04.1f}  {kind:6s} {card}{extra}", (14, y), col, 0.5); y += 24
        enc.stdin.write(np.concatenate([img, panel], 1).tobytes())
    enc.stdin.close(); enc.wait()
    print("wrote", out, tot)


if __name__ == "__main__":
    main()
