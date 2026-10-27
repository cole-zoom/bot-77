"""BlueStacks over ADB (like scripts/replay_runner.py: taps and screenshots happen inside Android, so the Mac's mouse
stays free), screen recognition, and turning a policy Action into taps. A FakeDevice plays a recorded video instead,
so the whole loop can be tested without BlueStacks (scripts/live_dryrun.py)."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

ADB = "/Applications/BlueStacks.app/Contents/MacOS/hd-adb"
REPLAY_UI = Path("data/raw/replays/_ui")  # the end screen (VS + OK) looks the same after a battle as after a replay
LIVE_UI = Path("data/raw/live_ui")  # home screen Battle button (cut from a BlueStacks screenshot)
MATCH = 0.8


def _templates():
    tpl, pos = {}, {}
    for root, names in ((REPLAY_UI, ("end_ok", "end_vs")), (LIVE_UI, ("home_battle", "end_again", "road_ok"))):
        p = json.loads((root / "pos.json").read_text()) if (root / "pos.json").exists() else {}
        for n in names:
            if (root / f"{n}.png").exists():
                tpl[n], pos[n] = cv2.imread(str(root / f"{n}.png")), p.get(n)
        pos.update({k: v for k, v in p.items() if k.endswith("_tap")})
    return tpl, pos


TPL, POS = _templates()


def seen(im: np.ndarray, name: str, slack: int = 60) -> float:
    """Match score of a screen template near where it was cut (0 when unknown)."""
    if name not in TPL or POS.get(name) is None or im.shape[:2] != (1920, 1080):
        return 0.0
    t, (x, y) = TPL[name], POS[name]
    h, w = t.shape[:2]
    win = im[max(0, y - slack):y + h + slack, max(0, x - slack):x + w + slack]
    return float(cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED).max())


# the OK button on the battle end screen when Play Again is shown next to it (fractions of the screen; 2026-10-23: its
# text doesn't OCR, and the end_ok template's spot is the single-OK layout)
END_OK_WITH_AGAIN = (0.655, 0.906)
END_AGAIN = (370, 1736)  # the Play Again button in that layout (pixels, 1080x1920; from the end_again template)


def _bright(im: np.ndarray) -> np.ndarray:
    return cv2.inRange(cv2.cvtColor(im, cv2.COLOR_BGR2HSV), (0, 0, 205), (180, 255, 255))


_WINNER = cv2.imread(str(LIVE_UI / "end_winner.png")) if (LIVE_UI / "end_winner.png").exists() else None
WINNER_TPL = _bright(_WINNER) if _WINNER is not None else None


def winner_y(im: np.ndarray, min_score: float = 0.4) -> float | None:
    """Height (fraction of the screen) of "Winner!" on the battle end screen, or None. Matched on its bright letters
    only: they are yellow, cyan or white depending on who won (2026-10-25; OCR misses the lettering). Checked on 102
    past end screens with known results: all right, weakest match 0.49."""
    if WINNER_TPL is None:
        return None
    im = cv2.resize(im, (1080, round(im.shape[0] * 1080 / im.shape[1])))
    r = cv2.matchTemplate(_bright(im[:, 300:780]), WINNER_TPL, cv2.TM_CCOEFF_NORMED)
    _, score, _, (_, y) = cv2.minMaxLoc(r)
    return y / im.shape[0] if score >= min_score else None


def end_ok_tap(im: np.ndarray) -> tuple[float, float]:
    """Where OK is on the battle end screen: read from the screen if possible, else the Play Again layout's spot."""
    try:
        okb = next(((x, y) for t, x, y in screen_texts(im) if t == "ok"), None)
    except Exception:  # noqa: BLE001
        okb = None
    h, w = im.shape[:2]
    return okb or (END_OK_WITH_AGAIN[0] * w, END_OK_WITH_AGAIN[1] * h)


def screen_kind(im: np.ndarray) -> str:
    """end | home | other (battle is decided from the HUD read: clock + elixir)."""
    if seen(im, "end_ok") >= 0.97 or (seen(im, "end_ok") >= MATCH and seen(im, "end_vs") >= MATCH):
        return "end"
    if seen(im, "end_again") >= MATCH:  # the Play Again + OK buttons (2026-10-25: VS scored 0.72, no text read)
        return "end"
    if seen(im, "end_vs") >= MATCH:  # the battle end screen with Play Again moves OK; confirm by its text
        try:
            if any("winner" in t or "play again" in t for t, _, _ in screen_texts(im)):
                return "end"
        except Exception:  # noqa: BLE001
            pass
    if seen(im, "home_battle") >= MATCH:
        return "home"
    return "other"


class Device:
    def __init__(self, serial: str = "127.0.0.1:5555", stream: bool = False):
        self.s = serial
        subprocess.run([ADB, "start-server"], capture_output=True)
        if serial.startswith("127.0.0.1"):
            subprocess.run([ADB, "connect", serial], capture_output=True)
        self.t0 = time.time()
        # the screen as a live stream (bot77.live.stream: ~6 frames/s) instead of 3.5-5 s screenshots
        self.stream = None
        self.last_frames = -1
        if stream:
            from bot77.live.stream import ScreenStream
            self.stream = ScreenStream(ADB, serial)

    def now(self) -> float:
        return time.time() - self.t0

    def cap(self) -> np.ndarray | None:
        if self.stream is not None:
            t = time.time()
            while self.stream.frame is None and time.time() - t < 10:  # the stream (re)starting
                time.sleep(0.05)
            # a frame not handed out before (up to 3 s): while the stream restarts (each game's recording starts
            # one) the last frame stayed, and the agent played 4 cards on one stale frame (2026-10-27). Menus can
            # be still for longer: after 3 s the last frame is the screen.
            while self.stream.frames == self.last_frames and time.time() - t < 3:
                time.sleep(0.02)
            if self.stream.frame is not None:
                self.last_frames = self.stream.frames
                return self.stream.latest()
        try:
            r = subprocess.run([ADB, "-s", self.s, "exec-out", "screencap", "-p"], capture_output=True, timeout=15)
        except subprocess.TimeoutExpired:  # ADB wedged (2026-10-22: a stray scrcpy held it): the caller recovers
            return None
        im = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)
        return im if im is not None and im.shape[:2] == (1920, 1080) else None

    def tap(self, x: float, y: float) -> None:
        try:
            subprocess.run([ADB, "-s", self.s, "shell", "input", "tap", str(int(x)), str(int(y))], check=True, timeout=10)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass

    def recover(self) -> None:
        """Unwedge ADB: stop stray screen recorders, restart the ADB server, reconnect."""
        recover_adb(self.s)


def recover_adb(serial: str) -> None:
    subprocess.run(["pkill", "-INT", "-f", f"scrcpy -s {serial}"])
    time.sleep(3)
    subprocess.run(["pkill", "-9", "-f", f"scrcpy -s {serial}"])
    subprocess.run(["pkill", "-f", "scrcpy-server"])
    subprocess.run([ADB, "kill-server"], capture_output=True)
    time.sleep(2)
    subprocess.run([ADB, "start-server"], capture_output=True)
    if serial.startswith("127.0.0.1"):
        subprocess.run([ADB, "connect", serial], capture_output=True)


class FakeDevice:
    """Plays a recorded video at real speed as the 'screen'; taps are only logged."""

    def __init__(self, video: Path, start: float = 0.0, speed: float = 1.0):
        self.cap_, self.start, self.speed, self.t0 = cv2.VideoCapture(str(video)), start, speed, time.time()
        self.fps = self.cap_.get(cv2.CAP_PROP_FPS) or 30.0
        self.taps: list[tuple[float, float, float]] = []

    def now(self) -> float:
        return self.start + (time.time() - self.t0) * self.speed

    def cap(self) -> np.ndarray | None:
        self.cap_.set(cv2.CAP_PROP_POS_FRAMES, int(self.now() * self.fps))
        ok, f = self.cap_.read()
        return f if ok else None

    def tap(self, x: float, y: float) -> None:
        self.taps.append((self.now(), x, y))


SAFE_BUTTONS = {"choose", "continue", "ok", "claim", "collect", "open", "next", "close", "skip", "done"}
NEVER = ("buy", "gem", "shop", "purchase", "offer", "pass", "clan", "join", "$", "price", "deal", "sale")


def screen_texts(im: np.ndarray) -> list[tuple[str, float, float]]:
    """(text, x, y) of the words on screen, centres in screen pixels (Apple Vision OCR on a half-size copy, ~0.1 s)."""
    from ocrmac import ocrmac
    from PIL import Image
    h, w = im.shape[:2]
    small = cv2.resize(im, None, fx=0.5, fy=0.5)
    out = []
    for text, _c, (bx, by, bw, bh) in ocrmac.OCR(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)),
                                                 recognition_level="fast").recognize():
        out.append((text.strip().lower(), (bx + bw / 2) * w, (1 - by - bh / 2) * h))  # Vision boxes: origin bottom-left
    return out


NAV = cv2.imread(str(LIVE_UI / "nav_battle.png")) if (LIVE_UI / "nav_battle.png").exists() else None


BOLT_MIN = 0.45


def battle_tab(im: np.ndarray) -> tuple[float, float] | None:
    """The Battle tab's lightning icon in the bottom bar, when another tab is open (2026-10-21: the bot sat on the
    Social tab, where tapping empty space does nothing)."""
    if NAV is None or im.shape[:2] != (1920, 1080):
        return None
    # match the bolt's outline, not its colours: the tab behind it is grey-blue or gold depending on the screen
    edges = lambda img: cv2.GaussianBlur(cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 60, 160), (5, 5), 0)
    bar = edges(im[1650:1920])
    best = None
    for k in (0.8, 0.9, 1.0, 1.1, 1.2):  # the icon is drawn smaller / larger depending on which tab is open
        t = edges(cv2.resize(NAV, None, fx=k, fy=k))
        _, v, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(bar, t, cv2.TM_CCOEFF_NORMED))
        if best is None or v > best[0]:
            best = (v, x + t.shape[1] / 2, 1650 + y + t.shape[0] / 2)
    return (best[1], best[2]) if best[0] >= BOLT_MIN else None


def between_games_tap(im: np.ndarray) -> tuple[float, float, str] | None:
    """Where to tap on a screen between games, or None: a safe whole-word button (never anything about buying,
    gems, offers or clans); the left card on "Choose your reward" (Cole: press Choose, then pick any card)."""
    if seen(im, "road_ok") >= MATCH and POS.get("road_ok_tap"):  # the trophy road: its OK doesn't OCR (2026-10-27:
        x, y = POS["road_ok_tap"]                                  # nudges hit a reward tile for 30 min)
        return x, y, "trophy road: OK"
    texts = screen_texts(im)
    if any(t == "battle log" for t, _, _ in texts):  # left open by the replay runner ("No Clan" on every entry would
        return 1011, 219, "close Battle Log"         # trip the never-tap words): its red X, top right
    if any(any(n in t for n in NEVER) for t, _, _ in texts if len(t) < 30) and \
            not any(t in SAFE_BUTTONS for t, _, _ in texts):
        tab = battle_tab(im)  # a shop / clan screen: never its buttons, but the Battle tab in the bottom bar is safe
        return (tab[0], tab[1], "Battle tab") if tab else None
    for t, x, y in texts:
        if (t in SAFE_BUTTONS or t.startswith("tap to")) and not any(n in t for n in NEVER):
            return x, y, f"button '{t}'"
    if any("choose your reward" in t or "choose a reward" in t for t, _, _ in texts):
        return 330, 540, "reward: left card"
    tab = battle_tab(im)
    if tab:
        return tab[0], tab[1], "Battle tab"
    return None


def slot_centre(layout, k: int, frame_w: int, frame_h: int) -> tuple[float, float]:
    x0, y0, x1, y1 = layout.box(f"slot{k + 1}", frame_w, frame_h)
    return (x0 + x1) / 2, (y0 + y1) / 2


# The bot's units landed 0.55 tiles below where it aimed (median over 1,791 plays of session 1-2, unit's first
# detection in the paired replay vs the intended spot; dx -0.08; Goblin Hut, which never moves, +0.75): docs/simulator.md
# S2b, decisions.md 2026-10-24. Aim that much higher.
AIM_DY = -0.55


def deploy(dev, layout, reader, action, frame_shape) -> tuple[float, float, float, float]:
    """Select the card, then tap the spot (Clash places a selected card where you tap). Returns the taps (px)."""
    h, w = frame_shape[:2]
    sx, sy = slot_centre(layout, action.slot, w, h)
    tile, ox, oy = reader.tile_px(w)
    # never tap past the river for a troop / building (only spells and Miner-type cards go there)
    tx, ty = ox + action.x * tile, oy + (action.y + AIM_DY) * tile
    dev.tap(sx, sy)
    time.sleep(0.08)
    dev.tap(tx, ty)
    return sx, sy, tx, ty
