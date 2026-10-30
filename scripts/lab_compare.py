"""After a lab session (docs/plan-sim-1to1.md P0.4): the recorded friendly battles against the simulator.

  # 1. pull the battles' replays (the BlueStacks Battle Log, newest first; --no-known: lab rematches look identical)
  uv run python scripts/replay_runner.py -n <battles played> --no-known
  # 2. mark them as lab games, run the usual pipeline on them, map each to its test, compare
  uv run python scripts/lab_compare.py --date 2026-10-29 --ingest
  # later runs (pipeline done): compare again, e.g. after an engine fix
  uv run python scripts/lab_compare.py --date 2026-10-29
  uv run python scripts/lab_compare.py --map L01=bl_2026-10-29_r05_m01 --map L02=bl_2026-10-29_r04_m01
  # the self-check on an existing held-back replay, treated as a lab test
  uv run python scripts/lab_compare.py --selftest bl_2026-10-12_r18_m01

--ingest writes the date's replay ids to config/lab_replays.json (scripts/build_sources.py then gives them source and
split 'lab', so they never reach training), then runs ingest_replays.py and the table builds (as the after-session
scripts do) and tower_hp for the lab games. Mapping: each replay goes to the test whose scripted placements it fits
best (same card and side, within 12 s), or --map.

Per test (bot77.sim.labcmp): landing tile vs script, who died when (real vs sim), tower HP over time, positions at
play + 1 / 2 / 4 / 6 / 8 s; printed, saved to data/lab/results/<test>_<match>.json, and a frame-pair viewer (replay frame
| sim board at the same moment, the replay's detections as rings on the sim board) at data/lab/viewer/<test>_<match>.html.
All local (data/lab is gitignored).
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import yaml

from bot77 import lab
from bot77.sim import labcmp, store
from bot77.sim.validate import load_game

ROOT = Path(__file__).resolve().parents[1]
LAB_REPLAYS = ROOT / "config/lab_replays.json"
RESULTS = Path("data/lab/results")
VIEWER = Path("data/lab/viewer")
SESSIONS = ROOT / "config/lab/sessions"
SESSION_MATCH_DT = 12.0  # a session map says when each run was; its placements match real plays this close
CHAIN = ["ingest_replays", "build_sources", "build_detections", "build_plays", "build_moments"]


def run_py(script: str, *args: str) -> None:
    print(f"=== {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / f"{script}.py"), *args], cwd=ROOT, check=True)


def mark_lab(date: str, only: str | None) -> list[str]:
    """The date's Battle Log recordings -> config/lab_replays.json (merged with what's there)."""
    ids = sorted(f"bl_{date}_r{p.stem.split('_')[1]}" for p in (ROOT / "data/raw/replays" / date).glob("replay_*.mp4"))
    if only:
        keep = set()  # "3-22" or "43-46,52-60"
        for part in only.split(","):
            a, _, b = part.partition("-")
            keep |= set(range(int(a), int(b or a) + 1))
        ids = [i for i in ids if int(i.rsplit("_r", 1)[1]) in keep]
    cur = json.loads(LAB_REPLAYS.read_text()) if LAB_REPLAYS.exists() else {"_doc": "", "replays": []}
    cur["_doc"] = ("Battle Log replays that are lab battles (scripted friendly battles, scripts/lab_play.py): "
                   "source and split 'lab' in build_sources.py, never training data.")
    cur["replays"] = sorted(set(cur["replays"]) | set(ids))
    LAB_REPLAYS.write_text(json.dumps(cur, indent=1) + "\n")
    return ids


def ingest(date: str, only: str | None) -> None:
    ids = mark_lab(date, only)
    print(f"{len(ids)} lab replays on {date}: {', '.join(ids)}")
    for s in CHAIN:
        run_py(s)
    run_py("build_tower_hp", "--sources", "lab", "--splits", "lab")
    run_py("build_tower_hp", "--merge")


def game(match_id: str):
    m = store.read("matches", f"match_id = '{match_id}'")
    if not m:
        sys.exit(f"{match_id} is not in `matches` (run with --ingest first?)")
    assert m[0]["split"] not in ("test", "heldout"), "never on test / held-out"
    return load_game(match_id, m[0]["split"], m[0]["source"]), m[0]


def auto_map(tests: dict[str, lab.LabTest], date: str) -> dict[str, str]:
    """Each of the date's lab replays -> the test it fits best (greedy on the fit score)."""
    lab_ids = set(json.loads(LAB_REPLAYS.read_text())["replays"]) if LAB_REPLAYS.exists() else set()
    ms = [m["match_id"] for m in store.read("matches", f"match_id LIKE 'bl_{date}_r%_m01'")]
    ms = [m for m in ms if "_top_" not in m and m.removesuffix("_m01") in lab_ids]  # never the ladder games
    scores = []
    for mid in ms:
        g, _ = game(mid)
        for tid, t in tests.items():
            s = labcmp.score(t, g)
            if s > 0:
                scores.append((s, tid, mid))
    out, used = {}, set()
    for s, tid, mid in sorted(scores, reverse=True):
        if tid not in out and mid not in used and s >= 0.5:
            out[tid] = mid
            used.add(mid)
    for mid in ms:
        if mid not in used:
            print(f"  {mid}: no test fits")
    return dict(sorted(out.items()))


def fmt_events(e: dict) -> str:
    if e["first"] is None:
        return "-"
    gone = ", ".join(f"{t:.1f}" for t in e["drops"]) or "-"
    return f"in {e['first']:.1f}, max {e['max']}, gone {gone}"


def print_report(rep: dict) -> None:
    print(f"\n### {rep['test']} {rep['name']}  ({rep['match_id']}, engine {rep['engine']})")
    if "status" in rep:
        print(f"run: shift {rep['shift']:+.1f} s, {rep['status']}; Cole: {rep['read'] or '-'}")
    print("landing (engine tiles):")
    for r in rep["landing"]:
        where = (f"seen {r['first_seen_t']:.1f} s at ({r['real_tile'][0]:.1f}, {r['real_tile'][1]:.1f}), "
                 f"off ({r['dx']:+.1f}, {r['dy']:+.1f})") if "real_tile" in r else ""
        when = "NOT FOUND in the replay" if r["t_real"] is None else f"played {r['t_real']:.1f} s ({r['late_s']:+.1f} vs plan)"
        print(f"  {r['side']:10s} {r['card']:16s} tile {tuple(r['tile'])}  {when}  {where}")
    print("bodies (game s; real | sim):")
    for b in rep["bodies"]:
        print(f"  {b['team']:3s} {b['card']:16s} real: {fmt_events(b['real']):42s} sim: {fmt_events(b['sim'])}")
    print("towers changed (HP fraction real / sim):")
    for r in rep["towers"]:
        ch = [f"{k} {v['real'] if v['real'] is not None else '?'}/{v['sim']}" for k, v in r.items()
              if k != "t" and (v["real"] not in (None, 1.0) or v["sim"] != 1.0)]
        if ch:
            print(f"  {r['t']:6.1f}  " + "  ".join(ch))
    print("positions (scripted units, mean error in engine tiles):")
    for r in rep["positions"]:
        e = f"{r['err']:.2f}  sim {r['sim']} real {r['real']}" if "err" in r else "unmatched"
        print(f"  {r['card']:16s} +{r['dt']:.0f} s  bodies sim {r['n_sim']} real {r['n_real']}  {e}")
    if rep["skipped"]:
        print("plays the sim couldn't take: " + ", ".join(f"{s['side']} {s['card']} @{s['t']:.1f} ({s['why']})" for s in rep["skipped"]))


def viewer(lr: labcmp.LabRun, rep: dict, mrow: dict, step: float = 0.5, max_frames: int = 100) -> Path:
    """data/lab/viewer/<test>_<match>.html: replay frame | sim board, every `step` s around the test."""
    name = f"{lr.test.id}_{lr.game.game_id}"
    d = VIEWER / name
    d.mkdir(parents=True, exist_ok=True)
    plays = [r["play"]["t_match"] for r in lr.matched if r["play"] is not None] or [p.t for p in lr.test.placements]
    t0, t1 = min(plays) - 1.0, max(plays) + min(lr.test.tail_s, 40.0)
    v = store.read("videos", f"video_id = '{mrow['video_id']}'")
    cap = cv2.VideoCapture(str(ROOT / v[0]["path"])) if v and v[0].get("path") else None
    rows, next_t = [], t0
    for f in lr.frames:
        if f.t < next_t or f.t > t1 or len(rows) >= max_frames:
            continue
        next_t = f.t + step - 1e-6
        k = len(rows)
        real_ok = False
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_MSEC, f.vt * 1000)
            ok, im = cap.read()
            if ok:
                im = cv2.resize(im, (300, round(im.shape[0] * 300 / im.shape[1])))
                cv2.imwrite(str(d / f"{k:03d}_real.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 80])
                real_ok = True
        cv2.imwrite(str(d / f"{k:03d}_sim.png"), labcmp.draw_board(f))
        rows.append({"t": round(f.t, 2), "clock": lab.clock_text(f.t), "real": f"{name}/{k:03d}_real.jpg" if real_ok else None,
                     "sim": f"{name}/{k:03d}_sim.png",
                     "units": f"sim {len(f.sim)} · replay {len(f.real)}"})
    out = VIEWER / f"{name}.html"
    out.write_text(_html(lr, rep, rows))
    return out


def _html(lr: labcmp.LabRun, rep: dict, rows: list[dict]) -> str:
    esc = html.escape
    def land_row(r: dict) -> str:
        played = "" if r["t_real"] is None else f"{r['t_real']:.1f}"
        late = "" if r["late_s"] is None else f"{r['late_s']:+.1f}"
        seen = "" if "real_tile" not in r else str(tuple(r["real_tile"]))
        return (f"<tr><td>{esc(r['side'])}</td><td>{esc(r['card'])}</td><td>{tuple(r['tile'])}</td>"
                f"<td>{played}</td><td>{late}</td><td>{seen}</td></tr>")
    land = "".join(land_row(r) for r in rep["landing"])
    bod = "".join(f"<tr><td>{esc(b['team'])}</td><td>{esc(b['card'])}</td><td>{esc(fmt_events(b['real']))}</td>"
                  f"<td>{esc(fmt_events(b['sim']))}</td></tr>" for b in rep["bodies"])
    pos = "".join(f"<tr><td>{esc(r['card'])}</td><td>+{r['dt']:.0f} s</td><td>{r['n_real']} / {r['n_sim']}</td>"
                  f"<td>{r.get('err', '')}</td></tr>" for r in rep["positions"])
    tw = []
    for r in rep["towers"]:
        ch = [f"{k} {v['real'] if v['real'] is not None else '?'} / {v['sim']}" for k, v in r.items()
              if k != "t" and (v["real"] not in (None, 1.0) or v["sim"] != 1.0)]
        if ch:
            tw.append(f"<tr><td>{r['t']:.1f}</td><td>{esc('  ·  '.join(ch))}</td></tr>")
    data = json.dumps(rows)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lab {esc(lr.test.id)} viewer</title><style>
:root {{ --bg:#f6f7f5; --panel:#fff; --ink:#1d2421; --muted:#5b665f; --line:#d5dbd7; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#121815; --panel:#1b231e; --ink:#e6ece7; --muted:#9aa89e; --line:#2c3830; }} }}
body {{ background:var(--bg); color:var(--ink); font:14px/1.45 system-ui, sans-serif; margin:0; padding:16px; }}
.wrap {{ max-width:1000px; margin:0 auto; }} h1 {{ font-size:1.3rem; margin:0 0 4px; }} .muted {{ color:var(--muted); }}
.pair {{ display:flex; gap:12px; flex-wrap:wrap; align-items:flex-start; margin:12px 0; }}
.pair figure {{ margin:0; }} .pair img {{ display:block; max-width:100%; border-radius:6px; border:1px solid var(--line); }}
figcaption {{ font-size:.8rem; color:var(--muted); }}
input[type=range] {{ width:100%; max-width:640px; }}
table {{ border-collapse:collapse; font-size:.85rem; margin:6px 0 14px; }} th, td {{ text-align:left; padding:3px 8px; border-bottom:1px solid var(--line); }}
.tbls > div {{ overflow-x:auto; margin-bottom:6px; }} td {{ white-space:nowrap; }}
</style></head><body><div class="wrap">
<h1>{esc(lr.test.id)} · {esc(lr.test.name)}</h1>
<div class="muted">{esc(lr.test.mechanic)} · replay {esc(lr.game.game_id)} · engine {esc(rep['engine'])} · ←/→ step, the sim board shows its units filled and the replay's detections as rings (blue BlueStacks, orange Cole), tower labels sim/real HP</div>
<div><input type="range" id="sl" min="0" max="{max(0, len(rows) - 1)}" value="0"> <span id="lab"></span></div>
<div class="pair"><figure><img id="real" alt="replay frame"><figcaption>replay</figcaption></figure>
<figure><img id="sim" alt="sim board"><figcaption>simulator</figcaption></figure></div>
<div class="tbls">
<div><b>Landing</b><table><tr><th>side</th><th>card</th><th>tile</th><th>played</th><th>late</th><th>seen at</th></tr>{land}</table></div>
<div><b>Bodies (real | sim)</b><table><tr><th>team</th><th>card</th><th>real</th><th>sim</th></tr>{bod}</table></div>
<div><b>Positions</b><table><tr><th>card</th><th>at</th><th>real / sim bodies</th><th>error (tiles)</th></tr>{pos}</table></div>
<div><b>Towers (HP fraction real / sim)</b><table>{''.join(tw)}</table></div>
</div></div>
<script>
const R = {data}; const sl = document.getElementById("sl");
function show(i) {{ const r = R[i]; if (!r) return; document.getElementById("real").src = r.real || "";
  document.getElementById("real").style.display = r.real ? "" : "none";
  document.getElementById("sim").src = r.sim; document.getElementById("lab").textContent = `${{r.t.toFixed(1)}} s · clock ${{r.clock}} · ${{r.units}}`; }}
sl.addEventListener("input", () => show(+sl.value));
document.addEventListener("keydown", e => {{ if (e.key === "ArrowRight") sl.value = Math.min(+sl.max, +sl.value + 1);
  else if (e.key === "ArrowLeft") sl.value = Math.max(0, +sl.value - 1); else return; show(+sl.value); }});
show(0);
</script></body></html>"""


def load_session(date: str) -> list[dict]:
    """config/lab/sessions/<date>.yaml: each run's test, replay, time shift, status and Cole's read."""
    return yaml.safe_load((SESSIONS / f"{date}.yaml").read_text())["runs"]


def compare(test: lab.LabTest, match_id: str, towers: str, no_viewer: bool, run: dict | None = None) -> dict:
    g, mrow = game(match_id)
    if run is not None:
        test = test.shifted(float(run["shift"]))
    lr = labcmp.run(test, g, towers=towers, match_dt=SESSION_MATCH_DT if run is not None else labcmp.MATCH_DT,
                    play_dt=(run or {}).get("play_dt"))
    rep = labcmp.report(lr)
    if run is not None:
        rep.update({"shift": run["shift"], "status": run["status"], "read": run.get("read", ""),
                    "play_dt": run.get("play_dt", {})})
    print_report(rep)
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"_s{round(float(run['shift']))}" if run is not None else ""
    (RESULTS / f"{test.id}_{match_id}{tag}.json").write_text(json.dumps(rep, indent=1, default=str))
    if not no_viewer:
        print(f"viewer: {viewer(lr, rep, mrow)}")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="the session's date (data/raw/replays/<date>/): auto-map its replays to tests")
    ap.add_argument("--only", help="with --ingest: only these replay numbers of the date, e.g. 3-22 or 43-46,52-60")
    ap.add_argument("--ingest", action="store_true", help="mark the date's replays as lab games and run the pipeline first")
    ap.add_argument("--session", metavar="DATE", help="run the session map config/lab/sessions/DATE.yaml")
    ap.add_argument("--all", action="store_true", help="--session: also the void runs")
    ap.add_argument("--map", action="append", default=[], help="TEST=MATCH_ID (repeatable)")
    ap.add_argument("--test", action="append", default=[], help="only these tests")
    ap.add_argument("--selftest", metavar="MATCH_ID", help="treat a recorded replay as a lab test (its own first plays)")
    ap.add_argument("--self-t0", type=float, default=0.0, help="--selftest: take the plays from this game time on")
    ap.add_argument("--towers", default="liquipedia_l11")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--switch", action="append", default=[], metavar="NAME=0|1",
                    help="an engine fidelity switch for this run (clash_royale_env.set_NAME), e.g. princess_first_shot_delay=1")
    a = ap.parse_args()
    import clash_royale_env as cre
    cre.set_client_collision(True)  # the play page's physics (scripts/situation_play.py)
    for kv in a.switch:
        name, _, v = kv.partition("=")
        getattr(cre, f"set_{name}")(v == "1")
    if a.selftest:
        g, _ = game(a.selftest)
        t = labcmp.synthetic_test(g, t0=a.self_t0)
        print(f"self-check: {len(t.placements)} plays of {a.selftest} as a lab test: " +
              ", ".join(f"{p.side} {p.card} {p.tile} @{p.t:.1f}" for p in t.placements))
        compare(t, a.selftest, a.towers, a.no_viewer)
        return
    if a.ingest:
        if not a.date:
            sys.exit("--ingest needs --date")
        ingest(a.date, a.only)
    tests = lab.load_tests()
    if a.test:
        tests = {k: v for k, v in tests.items() if k in a.test}
    if a.session:
        for r in load_session(a.session):
            if r["test"] in tests and (a.all or r["status"] != "void"):
                compare(tests[r["test"]], r["replay"], a.towers, a.no_viewer, r)
        return
    mapping = dict(m.split("=", 1) for m in a.map)
    if a.date and not mapping:
        mapping = auto_map(tests, a.date)
        print("mapping: " + ", ".join(f"{k}={v}" for k, v in mapping.items()))
    if not mapping:
        sys.exit("nothing to compare: give --date or --map")
    for tid, mid in mapping.items():
        if not re.match(r"^[A-Za-z0-9_\-]+$", mid):
            sys.exit(f"bad match id {mid!r}")
        compare(tests[tid], mid, a.towers, a.no_viewer)


if __name__ == "__main__":
    main()
