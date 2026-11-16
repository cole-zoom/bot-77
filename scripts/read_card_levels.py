"""Card levels of both players for every recorded replay, from its Battle Log entry image (2026-11-15).
-> LanceDB `card_levels` (video_id, side, tile, card, level, ...) and a summary.

  uv run --no-sync python scripts/read_card_levels.py

Levels: bot77.readers.levels (digit templates). Cards: a card's tile looks the same in every entry, so tiles are
clustered across all entries and each cluster is named by the decks the hand reading found for the replays it occurs
in (matches.deck_names; the name present in the most of them, >= 80% of the cluster's tiles with a known deck;
surest cluster first, a name taken by another tile of the same entry doesn't count: levels.name_clusters). This
also names tiles of replays not ingested yet (the phone's), as long as the card was seen in an ingested one.
Bottom = the entry's left player (the account that recorded it), top = right.
"""

import argparse
import collections
from pathlib import Path

import cv2
import numpy as np

from bot77.readers.levels import LevelReader, find_tiles, name_clusters
from bot77.sim import store

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--sim", type=float, default=0.90, help="tile similarity to join a cluster")
ap.add_argument("--purity", type=float, default=0.80)
a = ap.parse_args()


def feat(im, box):
    x, y, w, h = box
    t = cv2.resize(im[y:y + h, x:x + w], (100, round(100 * h / w)), interpolation=cv2.INTER_AREA)
    p = cv2.resize(t[int(0.08 * t.shape[0]):int(0.66 * t.shape[0]), 8:92], (24, 24), interpolation=cv2.INTER_AREA)
    p = p.astype(np.float32)
    p = (p - p.mean()) / (p.std() + 1e-6)
    return p.ravel() / np.sqrt(p.size)


decks = {}
for m in store.read("matches", "source IN ('replay_bl', 'replay_bot', 'replay_tv')"):
    vid = m["video_id"].removesuffix("_top")
    decks.setdefault(vid, {})[m["pov_side"]] = list(m["deck_names"] or [])

reader = LevelReader([])
tiles = []  # (video_id, side, tile index, feature, level, digit score)
skipped = 0
for png in sorted((ROOT / "data/raw/replays").glob("*/replay_*_entry.png")):
    vid = f"bl_{png.parent.name}_r{png.stem.split('_')[1]}"
    im = cv2.imread(str(png))
    T = find_tiles(im)
    if len(T) != 16:
        skipped += 1
        continue
    for k, b in enumerate(T):
        side = "bottom" if k % 8 < 4 else "top"
        lv, ds = reader.level(im, b)
        tiles.append((vid, side, k, feat(im, b), lv, ds))
print(f"{len(tiles) // 16} entries read, {skipped} skipped (tiles not found)")

# greedy clustering on the art
F = np.array([t[3] for t in tiles])
cent, members = [], []
for i, f in enumerate(F):
    if cent:
        s = np.array(cent) @ f
        j = int(s.argmax())
        if s[j] >= a.sim:
            members[j].append(i)
            continue
    cent.append(f)
    members.append([i])
names = name_clusters({j: [(tiles[i][0], tiles[i][1]) for i in idx] for j, idx in enumerate(members)},
                      {(v, sd): d for v, ds in decks.items() for sd, d in ds.items()}, a.purity)
lab = {}
for j, idx in enumerate(members):
    for i in idx:
        lab[i] = j
print(f"{len(members)} tile clusters, {len(names)} named; tiles in named clusters "
      f"{sum(len(members[j]) for j in names) / len(tiles):.1%}")

rows = []
for i, (vid, side, k, _, lv, ds) in enumerate(tiles):
    nm = names.get(lab[i])
    deck = decks.get(vid, {}).get(side)
    rows.append({"video_id": vid, "side": side, "tile": k, "card": nm[0] if nm else None, "level": lv,
                 "digit_score": round(float(ds), 3), "cluster": lab[i],
                 "cluster_purity": round(nm[1], 3) if nm else None,
                 "card_in_deck": None if deck is None or nm is None else nm[0] in deck})
store.write("card_levels", rows, mode="overwrite")
lv_ok = sum(r["level"] is not None for r in rows)
print(f"{len(rows)} tiles -> card_levels: level read {lv_ok / len(rows):.1%}, card named "
      f"{sum(r['card'] is not None for r in rows) / len(rows):.1%}, named card not in the hand-read deck "
      f"{sum(r['card_in_deck'] is False for r in rows)}")
print("levels:", sorted(collections.Counter(r["level"] for r in rows if r["level"]).items()))
by = collections.defaultdict(list)
for r in rows:
    if r["level"]:
        by[(r["video_id"][:13], r["side"])].append(r["level"])
for k in sorted(by):
    print(f"  {k[0]} {k[1]:6s} median level {np.median(by[k]):.0f} (range {min(by[k])}-{max(by[k])}, n {len(by[k])})")
