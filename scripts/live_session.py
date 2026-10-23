"""A long bot session (2026-10-21): play in chunks of --chunk games, and after each chunk pull those games' replays
from the Battle Log (it keeps only 25) with replay_runner.py, until --hours are up or data/live/STOP exists.
Replays are processed into answer keys afterwards (ingest_replays.py) so the bot isn't slowed during play.

  caffeinate -i -s uv run python scripts/live_session.py --hours 4
"""
import argparse, subprocess, sys, time
from pathlib import Path

import lancedb

from bot77.detect.yolo import Bot77Detector
from bot77.layout import Layout
from bot77.live.device import Device, recover_adb
from bot77.live import ui
from bot77.live.loop import STOP_FILE, Runner

ap = argparse.ArgumentParser()
ap.add_argument("--hours", type=float, default=4.0); ap.add_argument("--chunk", type=int, default=20)
ap.add_argument("--device", default="127.0.0.1:5555"); ap.add_argument("--layout", default="bluestacks_live")
from bot77.models import adopted_path  # the registry's adopted live detector (phase 5)
ap.add_argument("--weights", default=adopted_path("detector_live", "data/models/bot77_detector/v6l/best.pt"))
a = ap.parse_args()
STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
STOP_FILE.unlink(missing_ok=True)
print(ui.banner(a.device, a.layout, getattr(a, "games", 0) or 999, a.hours), flush=True)
cards = lancedb.connect("data/lancedb").open_table("cards").search().where("kind = 'deck_card'").limit(1000).to_arrow().to_pylist()
det, layout, dev = Bot77Detector(Path(a.weights)), Layout.load(a.layout), Device(a.device)
end, total, chunk_no = time.time() + a.hours * 3600, 0, 0
deck, errors = [], 0
while time.time() < end and not STOP_FILE.exists():
    chunk_no += 1
    left_h = (end - time.time()) / 3600
    print(f"== chunk {chunk_no}: up to {a.chunk} games, {left_h:.2f} h left", flush=True)
    r = Runner(dev, layout, cards, det, max_games=a.chunk, max_hours=left_h)
    r.known_deck = deck
    try:
        r.run()
        errors = 0
    except Exception as e:  # noqa: BLE001 — recover and carry on (2026-10-22: ADB wedged by a stray recorder)
        errors += 1
        print(f"== chunk {chunk_no} crashed ({e!r}); recovering ADB ({errors} in a row)", flush=True)
        recover_adb(a.device)
        if errors >= 3:
            print("3 crashes in a row: stopping", flush=True); break
    deck, played = r.known_deck, len(r.games)
    total += played
    print(f"== chunk {chunk_no}: {played} games played ({total} total)", flush=True)
    if played == 0:
        print("no games played in this chunk: stopping (check data/raw/live_ui/unknown/)", flush=True)
        break
    # the replays of this chunk's games are the newest entries of the Battle Log. Relaunch Clash first so it opens on
    # the home screen (after a game it can sit on reward screens the replay runner doesn't know, 2026-10-22)
    from bot77.live.device import ADB
    # reset ADB first: after a chunk of play the replay runner's scrcpy failed to start twice in a row
    # (2026-10-22, "timed out after 60 seconds"); the one pull right after an ADB reset worked
    recover_adb(a.device)
    subprocess.run([ADB, "-s", a.device, "shell", "am", "force-stop", "com.supercell.clashroyale"])
    time.sleep(2)
    subprocess.run([ADB, "-s", a.device, "shell", "monkey", "-p", "com.supercell.clashroyale", "-c",
                    "android.intent.category.LAUNCHER", "1"], capture_output=True)
    time.sleep(45)
    from bot77.live.device import battle_tab
    im = dev.cap()  # this account reopens on the Cards tab: go to Battle, where the replay runner starts
    tab = battle_tab(im) if im is not None else None
    if tab:
        dev.tap(*tab); time.sleep(3)
    # the replay runner's output streams to the terminal (2026-10-22, Cole: "make it output everything to terminal");
    # a failed pull is retried once after relaunching Clash (2026-10-23: the list wasn't reached once)
    # replays play back in real time (~3 min each): the time limit scales with how many (2026-10-23: a flat 40 min cut
    # a 20-replay pull at 13); a retry pulls only the ones not recorded yet (the newest come first: --skip them)
    done_n = 0
    for attempt in (1, 2):
        todo = played - done_n
        if todo <= 0:
            break
        limit_min = max(15, 6 * todo)
        print(f"== pulling {todo} replays from the Battle Log (skipping the newest {done_n}; try {attempt}, "
              f"up to {limit_min} min)", flush=True)
        rr = subprocess.Popen([sys.executable, "-u", "scripts/replay_runner.py", "-n", str(todo), "--skip", str(done_n),
                               "--no-known", "--device", a.device], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        t_pull = time.time()
        for line in rr.stdout:
            print(ui.c(f"   [replays] {line.rstrip()}", "grey"), flush=True)
            if line.startswith("[") and ".mp4:" in line:
                done_n += 1
            if time.time() - t_pull > limit_min * 60:
                rr.kill(); print(f"== replay pull timed out ({limit_min} min)", flush=True); break
        rr.wait()
        print(f"== replays pulled: {done_n} of {played} (exit {rr.returncode})", flush=True)
        if rr.returncode == 0 or done_n >= played:
            break
        recover_adb(a.device)
        subprocess.run([ADB, "-s", a.device, "shell", "am", "force-stop", "com.supercell.clashroyale"])
        time.sleep(2)
        subprocess.run([ADB, "-s", a.device, "shell", "monkey", "-p", "com.supercell.clashroyale", "-c",
                        "android.intent.category.LAUNCHER", "1"], capture_output=True)
        time.sleep(60)
        im = dev.cap()
        tab = battle_tab(im) if im is not None else None
        if tab:
            dev.tap(*tab); time.sleep(3)
    recover_adb(a.device)  # the replay runner's scrcpy can outlive it and wedge ADB for the next chunk
print(f"session done: {total} games", flush=True)
