"""A long live test that keeps every game learnable (2026-10-29, Cole: "play 75 ... would it be possible to learn from
these games? if so, set it up so that we do").

The Battle Log only holds the last 25 battles, and its replays are the only full truth of a live game (both hands,
both elixir bars, every play; scripts/replay_runner.py). So the games go in chunks: play a chunk with
scripts/live_play.py, stop, record that chunk's replays, move the videos to the LaCie (laptop disk is ~94% full; a live
game is ~0.45 GB, a replay ~0.12 GB), and go again until the total is reached. A replay failure never stops the games:
it's logged and the next chunk starts.

  uv run --no-sync python scripts/live_with_replays.py --ckpt X --games 75 --chunk 20 [--adopt]

--adopt: a live_play.py run is already going (started by hand with --games N): wait for --chunk of its games, stop it
with the STOP file, then carry on from there. Progress: data/live/<tag>_supervisor.log.
"""
import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data/live"
STOP = LIVE / "STOP"
GAMES = ROOT / "data/raw/live_games"
REPLAYS = ROOT / "data/raw/replays"
LACIE = Path("/Volumes/LaCie/bot-77-data/raw")
GAME_LINE = re.compile(r"■ Game (\d+): (\S*)")


def log(msg: str, f: Path) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with f.open("a") as fh:
        fh.write(line + "\n")


def results(live_log: Path) -> list[str]:
    return [m.group(2) for m in GAME_LINE.finditer(live_log.read_text(errors="ignore"))] if live_log.exists() else []


def live_running() -> bool:
    return subprocess.run(["pgrep", "-f", "scripts/live_play.py"], capture_output=True).returncode == 0


def to_lacie(sup: Path) -> None:
    """Videos older than 2 min (finished) -> the LaCie, a symlink left behind (memory: storage on the LaCie)."""
    if not LACIE.parent.exists():
        log("LaCie not mounted: videos stay on the laptop", sup)
        return
    moved = 0
    for p in list(GAMES.glob("*/game.mkv")) + list(REPLAYS.glob("2026-*/replay_*.mp4")) + list(REPLAYS.glob("2026-*/replay_*.mkv")):
        if p.is_symlink() or time.time() - p.stat().st_mtime < 120:
            continue
        dst = LACIE / p.relative_to(ROOT / "data/raw")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        if dst.stat().st_size == p.stat().st_size:
            p.unlink()
            p.symlink_to(dst)
            moved += 1
    log(f"moved {moved} videos to the LaCie", sup)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=75)
    ap.add_argument("--chunk", type=int, default=20, help="games per chunk (< 25, the Battle Log's length)")
    ap.add_argument("--tag", default="run75")
    ap.add_argument("--adopt", default=None, help="the log of a live_play.py run already going: its first chunk")
    ap.add_argument("--decision-s", default="0.1")
    a = ap.parse_args()
    sup = LIVE / f"{a.tag}_supervisor.log"
    record: list[str] = []
    chunk_no = 0
    live_log = Path(a.adopt) if a.adopt else None
    while len(record) < a.games:
        chunk_no += 1
        want = min(a.chunk, a.games - len(record))
        if live_log is None or chunk_no > 1:
            live_log = LIVE / f"{a.tag}_chunk{chunk_no:02d}.log"
            STOP.unlink(missing_ok=True)
            cmd = ["uv", "run", "--no-sync", "python", "scripts/live_play.py", "--stream", "--policy", "selfplay",
                   "--ckpt", a.ckpt, "--decision-s", a.decision_s, "--games", str(want), "--hours", "4"]
            log(f"chunk {chunk_no}: {want} games -> {live_log.name}", sup)
            subprocess.Popen(cmd, cwd=ROOT, stdout=live_log.open("w"), stderr=subprocess.STDOUT)
            time.sleep(20)
        else:
            log(f"chunk {chunk_no}: adopting the run already going ({live_log.name}), stopping it after {want} games", sup)
        while live_running():
            if len(results(live_log)) >= want and not STOP.exists():
                STOP.touch()  # live_play finishes the game in progress, then exits
                log(f"chunk {chunk_no}: {want} games done, STOP sent", sup)
            time.sleep(30)
        got = results(live_log)
        record += got
        w = record.count("WIN")
        log(f"chunk {chunk_no}: {got.count('WIN')}-{len(got) - got.count('WIN')}; total {w}-{len(record) - w} "
            f"over {len(record)} games", sup)
        STOP.unlink(missing_ok=True)
        if not got:
            log("no games in this chunk: stopping (check the live log)", sup)
            break
        if (LIVE / "STOP_ALL").exists():
            log("STOP_ALL found: no replays, no more chunks", sup)
            break
        time.sleep(10)
        # the chunk's replays: newest len(got) battles, without the known-deck skip (same deck every game)
        rlog = LIVE / f"{a.tag}_replays{chunk_no:02d}.log"
        r = subprocess.run(["uv", "run", "--no-sync", "python", "scripts/replay_runner.py", "-n", str(len(got)), "--no-known"],
                           cwd=ROOT, stdout=rlog.open("w"), stderr=subprocess.STDOUT)
        log(f"chunk {chunk_no}: replay runner exit {r.returncode} ({rlog.name})", sup)
        to_lacie(sup)
        if (LIVE / "STOP_ALL").exists():
            log("STOP_ALL found: no more chunks", sup)
            break
    (LIVE / f"{a.tag}_record.json").write_text(json.dumps({"results": record, "wins": record.count("WIN"),
                                                          "games": len(record)}))
    log(f"done: {record.count('WIN')}-{len(record) - record.count('WIN')} over {len(record)} games", sup)


if __name__ == "__main__":
    main()
