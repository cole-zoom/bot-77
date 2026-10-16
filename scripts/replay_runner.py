"""Watch the Battle Log replays in BlueStacks one after another and record each one (decisions.md, 2026-10-12).

Replays show BOTH players' hands (cards, Next, elixir), so one recorded replay is full play truth for both sides, and
the Battle Log entry shows both decks. Everything goes through BlueStacks' ADB: taps and recording happen inside
Android, so the Mac's mouse and screen stay free (keep BlueStacks running, don't quit it). The runner only navigates
the replay menus (Battle Log -> Watch -> OK). It never plays a battle and never reads the game; it sees the screen.

Loop: Battle Log (opens at the top) -> find the entry after the last one recorded (scroll slowly if needed) -> save the
entry (both decks) -> start recording -> Watch -> wait for the end screen (VS + OK) -> stop -> OK (back to the log).

  uv run python scripts/replay_runner.py -n 25            # the newest 25 battles
  uv run python scripts/replay_runner.py -n 10 --skip 5   # skip the 5 newest, then record 10
  uv run python scripts/replay_runner.py --source tv -n 30   # TV Royale (opens it from the home screen if needed)

TV Royale: starts at the top league (Ultimate Champion; right arrow until its tab shows) and, when a league's list
runs out, taps the left arrow for the next league down (Cole: lower leagues are fine too). Each replay's log row
records the league (tab 0 = Ultimate Champion, 1 = one down, ...); step A should keep to tab 0.

Output: data/raw/replays/<date>/ (Battle Log) or data/raw/tv_royale/<date>/ (TV Royale): replay_NN.mp4 +
replay_NN_entry.png + runner_log.jsonl. Battles already recorded on an earlier day (same entry, found by its decks)
are skipped, so the runner can be re-run daily. TV Royale is top players' games: pro truth for step A as well.
Needs: BlueStacks Settings > Advanced > Android Debug Bridge on; `brew install scrcpy` (records the Android screen).
Screen templates (game frames, gitignored) live in data/raw/replays/_ui/.
"""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ADB = "/Applications/BlueStacks.app/Contents/MacOS/hd-adb"
UI = ROOT / "data/raw/replays/_ui"
W, H = 1080, 1920  # Android screen
ENTRY_UP, ENTRY_DOWN = 330, 45  # an entry's fingerprint (both decks): from this far above its Watch button to this far above it
VIEW_TOP = 420  # the list scrolls under the Battle Log header above this y
MATCH = 0.8  # normalized correlation for a screen template
SAME_ENTRY = 0.93  # ... and for finding the last recorded entry again
KNOWN = 0.90  # ... and for "recorded before": blurred quarter-size thumbnails (card shine animates), compared with
              # +-24 px vertical slack (a watched entry shows its result and score, which pushes the decks down 8 px:
              # TV 30/44 slipped through at 0.80 without slack). Duplicates 0.97, different games <= 0.82; a friendly
              # rematch with identical decks (0.997) can't be told apart
MIN_REPLAY_S, MAX_REPLAY_S, POLL_S = 40, 600, 3.0
KNOWN_RUN = 4  # this many already-recorded entries in a row: the rest of the list is older, so it is done (Cole,
               # 2026-10-12: don't scroll through a league we already have). Lists are newest first; a few re-listed
               # games can sit near the top, hence 4 rather than 1.


class Device:
    def __init__(self, serial: str):
        self.s = serial
        subprocess.run([ADB, "start-server"], capture_output=True)
        if serial.startswith("127.0.0.1"):
            subprocess.run([ADB, "connect", serial], capture_output=True)

    def cap(self) -> np.ndarray:
        r = subprocess.run([ADB, "-s", self.s, "exec-out", "screencap", "-p"], capture_output=True, timeout=30)
        im = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)
        if im is None or im.shape[:2] != (H, W):
            sys.exit(f"screencap failed ({len(r.stdout)} bytes); is BlueStacks running with ADB on?")
        return im

    def tap(self, x: int, y: int) -> None:
        subprocess.run([ADB, "-s", self.s, "shell", "input", "tap", str(int(x)), str(int(y))], check=True)

    def scroll(self, dy: int = 250) -> None:
        """Slow swipe up. The list moves ~1.4x the swipe (it still flings a little), so keep it short."""
        subprocess.run([ADB, "-s", self.s, "shell", "input", "swipe", "540", "1500", "540", str(1500 - dy), "1500"],
                       check=True)
        time.sleep(1.2)


TPL = {n: cv2.imread(str(UI / f"{n}.png")) for n in ("end_ok", "end_vs", "log_title", "home_menu", "tv_title", "tv_top_tab")}
TAB_BAND = (290, 400, 180, 900)  # y0, y1, x0, x1 of the TV league tab title
SOURCES = {"log": ("log_title", "replays"), "tv": ("tv_title", "tv_royale")}  # list screen template, output folder
TITLE = "log_title"
POS = json.loads((UI / "pos.json").read_text())


def seen(im: np.ndarray, name: str, slack: int = 40) -> bool:
    t, (x, y) = TPL[name], POS[name]
    h, w = t.shape[:2]
    win = im[max(0, y - slack):y + h + slack, max(0, x - slack):x + w + slack]
    return cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED).max() >= MATCH


def at_end(im: np.ndarray) -> bool:
    """The replay's end screen. The OK button alone is enough when it matches tightly: units still fighting behind
    the banners can cover "VS" (2026-10-13: TV 108 scored 0.74 on VS, 0.99 on OK, and the loop sat there 8 hours)."""
    if not seen(im, "end_ok"):
        return False
    t, (x, y) = TPL["end_ok"], POS["end_ok"]
    h, w = t.shape[:2]
    win = im[max(0, y - 40):y + h + 40, max(0, x - 40):x + w + 40]
    return seen(im, "end_vs") or cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED).max() >= 0.97


def watch_buttons(im: np.ndarray) -> list[tuple[int, int]]:
    """Centres of the green Watch buttons, top to bottom."""
    m = cv2.inRange(cv2.cvtColor(im, cv2.COLOR_BGR2HSV), (40, 120, 100), (80, 255, 255))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    out = [(x + w // 2, y + h // 2) for x, y, w, h, a in st[1:] if a > 5000 and 150 < w < 300 and 40 < h < 100]
    return sorted(out, key=lambda p: p[1])


def fingerprint(im: np.ndarray, by: int) -> np.ndarray | None:
    if by - ENTRY_UP < VIEW_TOP:
        return None
    return cv2.cvtColor(im[by - ENTRY_UP:by - ENTRY_DOWN, 40:W - 40], cv2.COLOR_BGR2GRAY)


def find_entry(im: np.ndarray, fp: np.ndarray) -> int | None:
    """Watch-button y of the entry with this fingerprint, if it's on screen."""
    r = cv2.matchTemplate(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), fp, cv2.TM_CCOEFF_NORMED)
    _, v, _, (x, y) = cv2.minMaxLoc(r)
    return y + ENTRY_UP if v >= SAME_ENTRY else None


def open_log(d: Device) -> np.ndarray:
    """Get to the list (Battle Log or TV Royale) from wherever we are (end screen, home). TV Royale can take a minute
    to load after OK (a spinner over the end screen), and a replay may still be playing when a run starts, so in TV
    mode an unknown screen is waited out (up to MAX_REPLAY_S), never left: Back would quit to the main menu."""
    t0, last_ok = time.time(), 0.0
    for _ in range(6 if TITLE == "log_title" else 10**6):
        im = d.cap()
        if seen(im, TITLE):
            return im
        if at_end(im):
            if time.time() - last_ok > 15:  # tap once, then let it load
                x, y = POS["end_ok"]; d.tap(x + 110, y + 50); last_ok = time.time()
        elif TITLE == "tv_title" and seen(im, "home_menu"):  # home screen: menu -> TV Royale (third item)
            x, y = POS["home_menu"]; d.tap(x + 50, y + 50); time.sleep(1.5)
            d.tap(645, 422); time.sleep(4)
        elif TITLE == "tv_title":
            if time.time() - t0 > MAX_REPLAY_S:
                sys.exit("TV Royale isn't open; open it in BlueStacks and run again")
            time.sleep(3)
            continue
        elif seen(im, "home_menu"):
            x, y = POS["home_menu"]; d.tap(x + 50, y + 50); time.sleep(1.5)
            d.tap(645, 129)  # "Battle Log", first item of the menu
        else:
            subprocess.run([ADB, "-s", d.s, "shell", "input", "keyevent", "4"])  # back
        time.sleep(2.0)
        if TITLE == "tv_title" and time.time() - t0 > MAX_REPLAY_S:
            break
    sys.exit("couldn't reach the list; open it in BlueStacks and run again")


def after(d: Device, im: np.ndarray, fp: np.ndarray) -> tuple[np.ndarray, int] | None:
    """Screen + Watch-button y of the entry right below the one with fingerprint `fp`, scrolling as needed."""
    for _ in range(40):
        y0 = find_entry(im, fp)
        if y0 is not None:
            below = [by for _, by in watch_buttons(im) if by > y0 + 100 and by - ENTRY_UP >= VIEW_TOP]
            if below:
                return im, below[0]
        d.scroll(); im2 = d.cap()
        if np.abs(im2.astype(np.int16) - im.astype(np.int16)).mean() < 1.0:
            return None  # the list didn't move: bottom of the Battle Log
        im = im2
    return None


def thumb(fp: np.ndarray) -> np.ndarray:
    """Blurred, small fingerprint: robust to the cards' shine animation."""
    return cv2.resize(cv2.GaussianBlur(fp, (0, 0), 3), (250, 72), interpolation=cv2.INTER_AREA)


def known_fps(root: Path) -> list[np.ndarray]:
    """Thumbnails of every battle recorded before, from the saved entry crops (they end 60 px below Watch)."""
    out = []
    for p in sorted(root.glob("*/replay_*_entry.png")):
        e = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if e is not None and e.shape[0] >= ENTRY_UP + 60:
            out.append(thumb(e[len(e) - 60 - ENTRY_UP:len(e) - 60 - ENTRY_DOWN, 40:W - 40]))
    return out


def _slack_sim(a: np.ndarray, b: np.ndarray) -> float:
    """a's middle rows searched in b with +-6 thumbnail rows (+-24 px) of vertical slack, both ways round."""
    return max(float(cv2.matchTemplate(b, a[6:-6], cv2.TM_CCOEFF_NORMED).max()),
               float(cv2.matchTemplate(a, b[6:-6], cv2.TM_CCOEFF_NORMED).max()))


def is_known(fp: np.ndarray, known: list[np.ndarray]) -> bool:
    t = thumb(fp)
    return any(_slack_sim(t, k) >= KNOWN for k in known)


def tab_band(im: np.ndarray) -> np.ndarray:
    y0, y1, x0, x1 = TAB_BAND
    return cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.int16)


def switch_tab(d: Device, arrow: str) -> bool:
    """Tap a TV league arrow; True if the league changed."""
    before = tab_band(open_log(d))
    d.tap(*POS[arrow]); time.sleep(2.5)
    return np.abs(tab_band(open_log(d)) - before).mean() > 6


def to_top_tab(d: Device) -> None:
    for _ in range(12):
        if seen(open_log(d), "tv_top_tab", slack=20):
            return
        if not switch_tab(d, "tv_right"):
            break
    sys.exit("couldn't find the Ultimate Champion tab in TV Royale")


def next_button(d: Device, last_fp: np.ndarray | None, skip: int,
                known: list[np.ndarray]) -> tuple[np.ndarray, int] | None:
    """Screen + Watch-button y of the next battle to record (the list reopens at the top after every replay)."""
    im = open_log(d)
    if last_fp is not None:
        nb = after(d, im, last_fp)
    else:
        bs = [by for _, by in watch_buttons(im) if by - ENTRY_UP >= VIEW_TOP]
        if not bs:
            return None
        nb = (im, bs[0])
        for _ in range(skip):
            nb = after(d, nb[0], fingerprint(*nb))
            if nb is None:
                return None
    run = 0
    while nb is not None and is_known(fingerprint(*nb), known):  # recorded on an earlier run
        run += 1
        if run >= KNOWN_RUN:
            return None  # everything below is older and already recorded: this list is done
        nb = after(d, nb[0], fingerprint(*nb))
    return nb


def record(d: Device, out: Path) -> subprocess.Popen:
    return subprocess.Popen(["scrcpy", "-s", d.s, "--no-window", "--no-audio", "--no-control", "--max-fps", "30",
                             "--video-bit-rate", "12M", f"--record={out}"], env={**os.environ, "ADB": ADB},
                            stdout=subprocess.DEVNULL, stderr=open(ROOT / "data/logs/scrcpy_last.log", "w"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=25, help="replays to record (the Battle Log holds 25)")
    ap.add_argument("--skip", type=int, default=0, help="skip this many of the newest battles first")
    ap.add_argument("--device", default="emulator-5554")
    ap.add_argument("--source", choices=SOURCES, default="log", help="log = Battle Log, tv = TV Royale")
    ap.add_argument("--max-tabs", type=int, default=6, help="TV: leagues to go through, from the top")
    a = ap.parse_args()
    global TITLE
    TITLE, folder = SOURCES[a.source]
    d = Device(a.device)
    root = ROOT / "data/raw" / folder
    known = known_fps(root)
    out_dir = root / time.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    first = 1 + max([int(p.stem.split("_")[1]) for p in out_dir.glob("replay_*.mp4")] or [0])
    log = (out_dir / "runner_log.jsonl").open("a")
    last_fp, tab = None, 0
    if a.source == "tv":
        to_top_tab(d)
    for i in range(first, first + a.n):
        nb = next_button(d, last_fp, a.skip, known)
        while nb is None and a.source == "tv" and tab < a.max_tabs - 1:  # this league is done: one league down
            if not switch_tab(d, "tv_left"):
                break
            tab += 1; last_fp = None
            print(f"league {tab} (one down)", flush=True)
            nb = next_button(d, None, 0, known)
        if nb is None:
            print("no more battles in the list"); break
        im, by = nb
        last_fp = fingerprint(im, by)
        known.append(thumb(last_fp))
        cv2.imwrite(str(out_dir / f"replay_{i:02d}_entry.png"), im[max(0, by - 560):by + 60])  # both decks
        mkv = out_dir / f"replay_{i:02d}.mkv"
        # scrcpy occasionally fails to start (2026-10-12 replay 47; 2026-10-13 replay 140 failed 3 tries 2.5 s apart,
        # then started fine by hand): retry with growing waits, stderr kept in data/logs/scrcpy_last.log
        for wait in (0, 5, 10, 20, 30):
            time.sleep(wait)
            rec = record(d, mkv)
            time.sleep(2.5)
            if rec.poll() is None:
                break
            print(f"[{i}] screen recorder didn't start; retrying", flush=True)
        else:
            (out_dir / f"replay_{i:02d}_entry.png").unlink(missing_ok=True)
            sys.exit(f"[{i}] screen recorder won't start; stopping")
        d.tap(900, by)
        t0 = time.time()
        time.sleep(5)
        if seen(d.cap(), TITLE):
            rec.send_signal(signal.SIGINT); rec.wait(30); mkv.unlink(missing_ok=True)
            sys.exit(f"[{i}] Watch didn't start a replay (tapped at y={by}); stopping")
        ended = False
        while time.time() - t0 < MAX_REPLAY_S:
            im = d.cap()
            if time.time() - t0 > MIN_REPLAY_S and at_end(im):
                ended = True
                break
            time.sleep(POLL_S)
        time.sleep(1.5)
        rec.send_signal(signal.SIGINT)
        rec.wait(60)
        mp4 = mkv.with_suffix(".mp4")
        if not mkv.exists():  # recorder died mid-replay: skip this game (its entry crop goes, so a later run retries it)
            (out_dir / f"replay_{i:02d}_entry.png").unlink(missing_ok=True)
            known.pop()
            print(f"[{i}] no recording (screen recorder died); skipped", flush=True)
            continue
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mkv), "-c", "copy", str(mp4)], check=True)
        mkv.unlink()
        dur = time.time() - t0
        log.write(json.dumps({"i": i, "file": mp4.name, "started": time.strftime("%H:%M:%S", time.localtime(t0)),
                              "seconds": round(dur), "end_seen": ended,
                              **({"tab": tab} if a.source == "tv" else {})}) + "\n"); log.flush()
        print(f"[{i}] {mp4.name}: {dur:.0f} s{'' if ended else ' (end screen NOT seen: timed out)'}", flush=True)
        # not ended: the next open_log taps OK if it's an end screen after all, or waits for the list
    print("done:", out_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        subprocess.run(["pkill", "-INT", "-f", "scrcpy -s"])
        sys.exit("stopped")
