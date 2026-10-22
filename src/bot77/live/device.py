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
    for root, names in ((REPLAY_UI, ("end_ok", "end_vs")), (LIVE_UI, ("home_battle",))):
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


def screen_kind(im: np.ndarray) -> str:
    """end | home | other (battle is decided from the HUD read: clock + elixir)."""
    if seen(im, "end_ok") >= 0.97 or (seen(im, "end_ok") >= MATCH and seen(im, "end_vs") >= MATCH):
        return "end"
    if seen(im, "home_battle") >= MATCH:
        return "home"
    return "other"


class Device:
    def __init__(self, serial: str = "127.0.0.1:5555"):
        self.s = serial
        subprocess.run([ADB, "start-server"], capture_output=True)
        if serial.startswith("127.0.0.1"):
            subprocess.run([ADB, "connect", serial], capture_output=True)
        self.t0 = time.time()

    def now(self) -> float:
        return time.time() - self.t0

    def cap(self) -> np.ndarray | None:
        r = subprocess.run([ADB, "-s", self.s, "exec-out", "screencap", "-p"], capture_output=True, timeout=30)
        im = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)
        return im if im is not None and im.shape[:2] == (1920, 1080) else None

    def tap(self, x: float, y: float) -> None:
        subprocess.run([ADB, "-s", self.s, "shell", "input", "tap", str(int(x)), str(int(y))], check=True)


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


def battle_tab(im: np.ndarray) -> tuple[float, float] | None:
    """The Battle tab's lightning icon in the bottom bar, when another tab is open (2026-10-21: the bot sat on the
    Social tab, where tapping empty space does nothing)."""
    if NAV is None or im.shape[:2] != (1920, 1080):
        return None
    bar = im[1650:1920]
    r = cv2.matchTemplate(bar, NAV, cv2.TM_CCOEFF_NORMED)
    _, v, _, (x, y) = cv2.minMaxLoc(r)
    return (x + NAV.shape[1] / 2, 1650 + y + NAV.shape[0] / 2) if v >= 0.8 else None


def between_games_tap(im: np.ndarray) -> tuple[float, float, str] | None:
    """Where to tap on a screen between games, or None: a safe whole-word button (never anything about buying,
    gems, offers or clans); the left card on "Choose your reward" (Cole: press Choose, then pick any card)."""
    texts = screen_texts(im)
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


def deploy(dev, layout, reader, action, frame_shape) -> tuple[float, float, float, float]:
    """Select the card, then tap the spot (Clash places a selected card where you tap). Returns the taps (px)."""
    h, w = frame_shape[:2]
    sx, sy = slot_centre(layout, action.slot, w, h)
    tile, ox, oy = reader.tile_px(w)
    # never tap past the river for a troop / building (only spells and Miner-type cards go there)
    tx, ty = ox + action.x * tile, oy + action.y * tile
    dev.tap(sx, sy)
    time.sleep(0.08)
    dev.tap(tx, ty)
    return sx, sy, tx, ty
