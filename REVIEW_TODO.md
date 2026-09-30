# Your review queue (2026-09-30)

Three click pages — same as before: pick the entity at the top if there are two, click the middle of
the unit, **skip frames where it isn't visible**, then **Export clicks** (lands in ~/Downloads, I'll file it).
Each frame is 1.5 s or 3 s after a play the POV detector logged, so the unit is usually on screen.

| Page | Frames | What |
|---|---|---|
| `data/reviews/click_new.html` | 208 | the 13 new cards, 8 plays each (from your + your friend's games) |
| `data/reviews/click_hero.html` | 106 | 10 hero forms (Berserker, Tombstone + Tomb Queen, Knight, Ice Golem, Goblins, Dark Prince + dismounted, Barb Barrel, Valk, Ice Wiz, Mega Minion) |
| `data/reviews/click_evo.html` | 146 | 18 evo forms (Cannon, Firecracker, Dart Goblin, Goblin Barrel, Ice Spirit, Tesla, Skeleton Barrel, Bats, Royal Hogs, Skeletons, Goblin Drill, and 1–2 each of Baby Dragon, Battle Ram, Snowball, Knight, Royal Ghost, Royal Giant, Witch) |

Tips: two-entity cards are Goblinstein (doctor/monster), Spirit Empress (ground/air), Tombstone hero
(building/Tomb Queen) and Dark Prince hero (mounted/dismounted) — click each visible one. Start with
`click_new` (most valuable), then hero, then evo. ~45–60 min total; they can be done in separate sittings.

## Already done without you
- 13 videos processed overnight (0 errors); 3 OYASSUU videos needed per-video layouts and were rerun.
- **532 exact opponent labels** from pairing your games with your friend's — no review needed.
- Every new card now has 12+ labelled plays of its own.
- Missing still: hero E-Wiz (not in the card API yet), and evo Baby Dragon / Battle Ram / Snowball / Knight /
  Royal Ghost / Royal Giant / Witch have only 1–2 plays each.


---

## Earlier (done)

Two pages, about 20–30 minutes total. Open each in a browser (double-click the file, or paste the
path into the address bar). Everything autosaves in the browser; **Export** when done and drop the
JSON in `~/Downloads` — I'll file it into `data/ground_truth/reviews/`.

## 1. Click the new cards in your games (most important)
**File:** `data/reviews/click_cole_2026-09-28.html` — 80 frames from your recording.

Each frame is taken **2 s after an opponent deploy** in the game you named, so the new unit is
usually on screen:

| Section | Game in your recording | Frames | Click on |
|---|---|---|---|
| Ronin | game starting 3:50 | 16 | `ronin` |
| Spirit Empress | game starting 8:13 | 16 | `spirit-empress` |
| Tombstone (hero) | game starting 15:53 | 13 | `tombstone-hero` and/or `tomb-queen` |
| Rune Giant | game starting 18:52 | 16 | `rune-giant` |
| Goblinstein | game starting 35:20 | 19 | `goblinstein-doctor` and/or `goblinstein-monster` |

- Click once on the **middle of the unit**. Clicking a frame auto-selects that section's card; for
  Tombstone and Goblinstein pick which entity with the buttons at the top first.
- **Skip** frames where the card isn't on screen, is hidden behind something, or is still under the
  deploy clock.
- Two entities in one frame (Doctor + Monster)? Click each once.
- **Export clicks** → `clicks_cole_recording.json`.

## 2. Mining round 2 gallery
**File:** `data/reviews/mined_r2_gallery.html` — 120 candidates found by LanceDB vector search
across all four videos (yours, Alina, Ian), using the 41 slices you've approved so far.

- **Keep** = it really is that card and the unit is mostly visible (background is fine — I cut
  kept ones out with SAM).
- **Reject** anything else. Fix the Doctor / Monster dropdown when it's the other one.
- Ignore the "friendly/enemy" label on the cards (it's the old detector's guess).
- The top ~12 Minion Giants are from your own games 1 and 6 — the search found them without
  being told where.
- **Export** → `slice_review.json`.

## What I did overnight (so you know where things stand)
- **Your recording** (39 min, iPhone Mirroring) → cropped to the phone, given its own layout
  (`cole_iphone_mirroring`: your top HUD sits 80 px lower and the hand bar 52 px higher than Alina's),
  processed: **11 matches**, your Hog EQ Barb Barrel deck read in every one, **18,654 unit crops
  embedded** in LanceDB, 300 opponent deploys (old detector).
- **Synthetic training set:** 12,000 arenas made with KataCR's generator (their MIT sprite cut-outs +
  our 41 approved Minion Giant / Goblinstein slices), validated on 800 of their real hand-labelled frames.
- **Detector v1 training** on the Mac GPU (YOLO11s, 12 epochs, ~6 h). When it finishes it's scored
  automatically on your 52-play opponent benchmark against the old 2024 detector →
  `data/logs/eval_detector_v1.json`. I'll walk you through the result.
- Not committed yet (per your one-big-commit preference) — say the word when you want it.

## After your review
Your clicks give the first slices of Ronin, Spirit Empress, Tombstone hero / Tomb Queen and Rune Giant.
Those become search queries for another mining round, then detector v2 with every new class.

## Detector v1 result (finished this morning)
Scored on your 52 labelled opponent plays (Alina match 1):

| detector | fully right | wrong card | missed |
|---|---|---|---|
| old 2024 detector | 31 / 52 | 19 | 2 |
| our v1 alone | 22 / 52 | 10 | 20 |
| **old + v1 for new cards** | **32 / 52** | **14** | 6 |

v1 learned Goblinstein (2/3 vs 0/3) but misses small units and spells — trained only on synthetic
arenas, small model, 12 epochs. **A decision for you** (details in chat): v2 = more real-footage
training data + a bigger model trained longer, likely on a rented GPU (~$10–30).
