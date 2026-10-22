"""Contact sheets for reviewing bot plays (skill review-bot-plays): 4 plays per image, each with the frame at the play
(the bot's tap circled) and 2 s later, labelled with id, card, reason, elixir, hand and threats.

  uv run python scripts/review_sheets.py <plays.json> <out_dir>   (plays = the review page's DATA items)
"""
import json, sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

plays = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
W = 300  # each frame's width in the sheet


def load(uri):
    p = Path(unquote(urlparse(uri).path))
    im = cv2.imread(str(p)) if p.exists() else None
    return cv2.resize(im, (W, round(im.shape[0] * W / im.shape[1]))) if im is not None else None


def tile(d):
    at, after = load(d["at"]), load(d["after"])
    if at is None:
        return None
    at = at.copy()
    cv2.circle(at, (int(d["tap"][0] * W), int(d["tap"][1] * at.shape[0])), 14, (0, 255, 255), 3)
    after = after if after is not None else np.full_like(at, 40)
    pair = np.hstack([at, np.full((at.shape[0], 6, 3), 255, np.uint8), after])
    head = np.full((70, pair.shape[1], 3), 255, np.uint8)
    th = ", ".join(t[0] for t in d["threats"][:4]) or "none"
    lines = [f"{d['id']}  {d['card']}  [{d['reason']}]  elixir {d['elixir']}",
             f"hand: {', '.join(h or '?' for h in d['hand'])}   threats: {th}"]
    for i, l in enumerate(lines):
        cv2.putText(head, l, (6, 26 + 30 * i), 0, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([head, pair])


tiles = [t for t in (tile(d) for d in plays) if t is not None]
for n in range(0, len(tiles), 4):
    group = tiles[n:n + 4]
    h = max(t.shape[0] for t in group)
    group = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in group]
    cv2.imwrite(str(out / f"sheet_{n // 4:02d}.jpg"), np.hstack(group), [cv2.IMWRITE_JPEG_QUALITY, 80])
print(len(tiles), "plays ->", (len(tiles) + 3) // 4, "sheets in", out)
