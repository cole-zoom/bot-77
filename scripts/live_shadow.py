"""B v0 offline check: run the live stack (bot77.live) over one of Cole's recorded, keyed games and compare what the
bot would do with what Cole did. Cole's real plays are fed to the policy as its own history (shadow mode), so the
state matches the game on screen.

Reports: card agreement at Cole's play moments (the live readers end to end, vs A on stored data); how often the bot
wants to play vs Cole's play rate; placement vs where Cole's unit appeared (same card); the opponent tracker vs the
answer key. Saves a few frames with the bot's intended spot drawn.

  uv run python scripts/live_shadow.py cole_2026-10-20_b_m03 [--fps 2.5] [--out data/reviews/shadow]
"""
import argparse, json, math
from pathlib import Path

import cv2
import lancedb

from bot77.detect.yolo import Bot77Detector
from bot77.layout import Layout
from bot77.live.policy import Policy
from bot77.live.state import LiveReader, OpponentTracker
from bot77.video import iter_frames

ap = argparse.ArgumentParser()
ap.add_argument("match"); ap.add_argument("--fps", type=float, default=2.5)
ap.add_argument("--layout", default="cole_iphone_mirroring")
ap.add_argument("--weights", default="data/models/bot77_detector/v6l/best.pt")
ap.add_argument("--out", default="data/reviews/shadow")
a = ap.parse_args()

db = lancedb.connect("data/lancedb")
cards = db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).to_arrow().to_pylist()
m = db.open_table("matches").search().where(f"match_id = '{a.match}'").to_arrow().to_pylist()[0]
video = Path(db.open_table("videos").search().where(f"video_id = '{m['video_id']}'").to_arrow().to_pylist()[0]["path"])
own = sorted(({"t": e["t_video"], "card": e["card"]} for e in db.open_table("events").search()
              .where(f"match_id = '{a.match}' AND kind = 'play'").limit(10**4).to_arrow().to_pylist() if e["card"]),
             key=lambda p: p["t"])
key = json.loads(Path(f"data/ground_truth/opponent_truth_live_{a.match}.json").read_text())["items"]
layout = Layout.load(a.layout)
reader = LiveReader(layout, cards, Bot77Detector(Path(a.weights)))
policy, opp = Policy(cards), OpponentTracker()
out = Path(a.out) / a.match; out.mkdir(parents=True, exist_ok=True)

decisions, units_by_t, fed, saved = [], {}, 0, 0
span = m["t_end"] - m["t_start"]
for t, f in iter_frames(video, a.fps, start=m["t_start"], duration=span):
    while fed < len(own) and own[fed]["t"] <= t:  # Cole's real plays become the policy's own history
        policy.record_own(own[fed]["t"], own[fed]["card"]); fed += 1
    st = reader.read(f, t)
    opp.update(st)
    units_by_t[round(t, 2)] = [u for u in st.units if u.team == "friendly" and u.card]
    policy_min_gap = policy.min_gap_s; policy.min_gap_s = 0.0  # shadow: Cole's own timing decides the gaps
    act = policy.decide(st, opp.plays)
    policy.min_gap_s = policy_min_gap
    decisions.append((t, st, act))
    if act is not None and saved < 8 and act.reason == "defend":
        f = f.copy()  # frames from iter_frames are read-only
        tile, ox, oy = reader.tile_px(f.shape[1])
        p = (int(ox + act.x * tile), int(oy + act.y * tile))
        cv2.circle(f, p, int(tile * 0.8), (0, 255, 255), 6)
        cv2.putText(f, f"{act.card} ({act.reason})", (p[0] - 120, p[1] - 50), 0, 1.4, (0, 255, 255), 4)
        cv2.imwrite(str(out / f"{t:07.1f}_{act.card.replace(' ', '_')}.jpg"), cv2.resize(f, (540, 1170)))
        saved += 1

# 1. card choice at Cole's play moments (the decision just before each play)
names = {c["id"]: c["name"] for c in cards}
agree = n = 0
for p in own:
    before = [d for d in decisions if p["t"] - 1.0 <= d[0] < p["t"] - 0.1 and d[1].in_battle and None not in d[1].hand]
    if not before:
        continue
    t, st, _ = before[-1]
    if names.get(st.hand[0]) is None or p["card"] not in [names.get(c) for c in st.hand]:
        continue
    h = {"hand": st.hand, "next": st.next, "elixir": st.elixir, "t_match": st.elapsed_s, "multiplier": st.multiplier}
    from bot77.moves import candidates
    rows = candidates(h, t, [q for q in own if q["t"] < t], opp.plays, policy.last_played, policy.cards, policy.names)
    s = policy.a.scores(rows)
    n += 1; agree += names.get(st.hand[max(range(4), key=lambda k: s[k])]) == p["card"]
# 2. when: bot triggers vs Cole's plays
trig = [d for d in decisions if d[2] is not None]
minutes = span / 60
covered = sum(1 for p in own if any(abs(d[0] - p["t"]) <= 1.5 for d in trig))
# 3. where: bot spot vs Cole's unit of the same card appearing within 3 s after his play
dist, lane_ok, nwhere = [], 0, 0
for p in own:
    near = [d for d in trig if abs(d[0] - p["t"]) <= 1.5 and d[2].card == p["card"]]
    spawn = [u for tt, us in units_by_t.items() if 0 <= tt - p["t"] <= 3 for u in us if u.card == p["card"]]
    if near and spawn:
        u, act = spawn[0], near[0][2]
        dist.append(math.hypot(u.x - act.x, u.y - act.y)); lane_ok += (u.x < 9) == (act.x < 9); nwhere += 1
# 4. opponent tracker vs the key
kp = [i for i in key if i["label"] not in ("__ability",)]
hit = sum(1 for q in opp.plays if any(i["label"] == q["card"] and 0 <= q["t"] - i["t_video"] <= 4 for i in kp))
print(f"{a.match}: {len(decisions)} frames at {a.fps} fps, deck read: {[names.get(c) for c in reader.deck]}")
print(f"  card choice at Cole's {n} plays: bot agrees {agree}/{n} ({agree / max(n, 1):.0%})")
print(f"  when: bot wants to play {len(trig) / minutes:.1f}x/min (Cole plays {len(own) / minutes:.1f}x/min); "
      f"{covered}/{len(own)} of Cole's plays have a bot trigger within 1.5 s")
if nwhere:
    print(f"  where (same card, {nwhere} cases): median {sorted(dist)[len(dist) // 2]:.1f} tiles from Cole's unit, same lane {lane_ok}/{nwhere}")
print(f"  opponent tracker: {len(opp.plays)} plays seen, {hit} match the key ({len(kp)} in the key)")
print(f"  reasons: { {r: sum(1 for d in trig if d[2].reason == r) for r in ('defend', 'full elixir', 'idle')} }; frames -> {out}")
