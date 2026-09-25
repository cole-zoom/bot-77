# Play Detector v0 — Spec

Status: draft for review
Source video for v0: Alina — "How to Play Hog EQ in the New Meta" (`uz4VVzGlOjE`, 1080×2340, 59.94 fps, 21:26)
Alina's deck: Hog Rider, Earthquake, Tesla (evo), Firecracker (evo), Skeletons, Electro Spirit, Mighty Miner (champion), The Log.
Common variants: Cannon (evo) instead of Tesla (never both in one deck); Barbarian Barrel instead of The Log.

## 1. Goal

Turn one YouTube video into a clean, per-match **event log of the POV player's plays**, stored in LanceDB, accurate enough that a strong player reviewing it agrees with ≥95% of events.

This is the foundation for everything after it (scenario mining, next-play prediction), so correctness beats coverage.

## 2. Scope

**In scope (v0)** — everything readable from the HUD, which sits in fixed screen positions:

| Signal | Where | Method |
|---|---|---|
| Hand (4 cards) + next card | bottom bar | template match against card art |
| Card state per slot: normal / greyed (unaffordable) / selected (raised) / evolution frame | bottom bar | template match + brightness/offset checks |
| Elixir count (0–10) | left of elixir bar | digit classifier (or OCR) |
| Match timer + overtime flag | top right | OCR |
| Elixir multiplier (none / x2 / x3) | top right, under timer | template match |
| Tower HP (6 towers) | above each tower | OCR |
| Champion / hero ability button: present, cost, available / used / unaffordable | left side, by the hand (a second one on the right when the deck runs two heroes) | template match + brightness check |
| Opponent name, clan, rating | top left | OCR, once per match |
| Match boundaries | whole frame | timer reset to 3:00, victory/defeat screens, opponent name change |

**Out of scope (v0)**
- Opponent plays (needs arena unit detection — v1)
- Unit tracking, positions over time (v1)
- Exact placement tile of POV plays (v0.5: approximate location, see §6.3)
- Ian's split layout (v0.1: reuse everything, add a second calibration)
- Transcript alignment (v1+)

## 3. Detecting a POV play

A play is confirmed when these signals agree inside a short window (~1 s):

1. **Hand change** — a card leaves slot *k*, and the previous "next" card fills slot *k*.
2. **Elixir drop** — elixir falls by that card's cost (from card metadata), accounting for regen during the window.
3. **Next card advances** — a new card shows in the "next" slot.
4. **Name label** — "<Card> lvl.16" text appears at the drop spot (POV plays only). Also gives the approximate placement for free (§6.3).

Confidence scoring:
- hand change + elixir drop + one more → `high`
- hand change + one other → `medium`
- hand change only → `low`, flagged for review

Edge cases to handle explicitly:
- Selecting a card without playing it (slot raised, nothing leaves the hand)
- Cards greyed at low elixir (still the same card; art is desaturated)
- Evolution cycle frames on evo cards
- Mirror, and "mystery" / special-event modes → detect and skip the match in v0
- Elixir leak at 10 (no regen during the window)
- Double / triple elixir regen rates
- Frames with overlays (emotes, "30 seconds left", "Battle ends in…", countdown digits) covering HUD regions

### 3.0 What a play actually looks like (measured, milestone 5)

The spec above assumed the three signals line up. On real footage they happen in sequence:
1. **Drag starts** → the card leaves its slot (slot reads empty). Elixir is unchanged.
2. **Release** → elixir drops by the card's cost, typically 0.2–1.5 s later. The drop animates over ~0.3 s and pauses briefly at whole numbers.
3. **Refill** → the next card advances and slides into the vacated slot.
A cancelled drag is step 1 followed by the same card returning, with no drop.

So `bot77.events.detect_plays` keys on **elixir drops**: the play is the card that left the hand within the last 4 s whose cost matches the regen-corrected drop (±0.4). Confidence: `high` if the next card also advanced and refilled the slot, `medium` if one of those, `low` if no card fits. Extras learned on match 1:
- Two cards released together → one drop equal to the sum of their costs; split by cost pair.
- A card can slide in and be dragged straight back out within one frame; slot smoothing keeps single-frame cards between empties.
- When a whole elixir ticks over, the bar can read ~0.9 high for a frame or two. The elixir series is capped at the regen rate (unless a faster rise holds 0.5 s) so these spikes don't become fake 1-elixir spends.
- A drop with no card leaving the hand is an ability, but only if that champion/hero has a deploy whose ability is unused; otherwise it's kept as `low`.

Match 1 of the source video: 44 high, 15 medium, 2 low plays; all 4 abilities follow the 4 Mighty Miner deploys.

### 3.1 Ability uses

A champion or hero ability use is its own event (`kind = ability`), confirmed by:
1. The ability button changes from available to used
2. Elixir drops by the ability cost

Each ability can be used once per deploy (Boss Bandit: twice). The button only exists while that card is on the field.

## 4. Deck structure

A deck is 8 cards plus a tower troop, with three special slots: **1 evo slot, 1 hero/champion slot, and 1 flex slot** (evo or hero/champion). Each hero or champion in play has an ability button. The first sits on the **right**, just above the hand; a **left** button only appears when a second ability exists (two champions, a champion + a hero, or two heroes). Alina's usual setup (evo Tesla/Cannon, Mighty Miner, evo Firecracker) only ever shows the right button. The readers and schema still take a list of ability buttons so other decks work without a redesign.

Evolutions: the card's evo charges after `evo_cycles` plays; the evo card art in hand shows when it's charged. v0 records which form each play used (`normal` / `evo` / `hero`).

## 5. Card metadata

**Built.** `uv run bot77 cards ingest` → LanceDB table `cards` (127 rows: 123 deck cards + 4 tower troops).

Inputs:
- `data/cards/api_snapshots/<date>.json` — raw official API `/cards` response. `items` are deck cards; `supportItems` are tower troops. New snapshots are added, never overwritten, so each match can be tied to the card pool of its date.
- `data/cards/sources/*.txt` — Cole's evo, hero and champion tables, kept verbatim. `scripts/curate_card_sources.py` parses them into `data/cards/curated.yaml` and cross-checks costs against the API.
- `data/cards/curated.yaml` — facts the API lacks: `evo` (cycles, stat boost, ability), `hero_ability` / `champion_ability` (name, cost, description, uses_per_deploy, cooldown_s), type and elixir overrides. Only gap: Ice Wizard's hero ability.

API quirks handled:
- `maxEvolutionLevel`: 1 = evo, 2 = hero, 3 = both, missing = neither. Champions have none. Checked against the icon URLs for all 123 cards: 0 disagreements.
- `maxLevel` is the number of upgrades, not the cap. Everything caps at 16, so starting level = 17 − maxLevel (common 1, rare 3, epic 6, legendary 9, champion 11). Mirrored cards play at +1 level.
- Card type comes from the id prefix (26 troop, 27 building, 28 spell), with overrides (e.g. Heal Spirit is a troop in the spell range).
- Mirror has no fixed elixir cost (`elixir_variable`).

Columns: identity, type, rarity, elixir, starting level, `has_evo` / `has_hero` / `is_champion`, evo cycles, ability fields, icon URLs, and PNG bytes for every art variant (normal / evo / hero — 186 images). The art doubles as templates for the hand reader.

## 6. Pipeline

```
video ──decode @10fps──► HUD crops ──► per-frame state ──► smoothing ──► play events ──► LanceDB
          (ffmpeg)       (calibration)   (readers)        (temporal)    (§3 logic)
```

### 6.1 Calibration (per creator layout)
A small JSON file with pixel rectangles for every HUD region, normalized to frame size. Made once per creator by hand-marking one frame; checked visually by drawing the boxes on 10 random frames.

Findings from calibrating Alina's layout (`uv run bot77 calibrate --layout alina_portrait --video …` draws the boxes on random frames and writes per-region crop strips):
- Mighty Miner's ability button appears on the **right**, just above the hand. Seen in two states: available (bright blue, pink cost pip) and unavailable (dark blue, grey pip). The left button (mirrored box) only appears with a second hero/champion, so it never shows in this deck.
- Evo cards have a tab above the card with charge pips that fill as the evo charges (confirmed by Cole); a charged evo shows a raised, glowing tab. Champion cards have a pointed silver frame top.
- A played slot is briefly empty (crown placeholder) before the next card slides in.
- Opponent king tower HP only shows once it's damaged; the POV king tower HP region isn't calibrated yet for the same reason.
- "Time left:" becomes "Overtime" in overtime; the multiplier drop shows x2 / x3.
- Occlusions seen: the emote picker covers the whole hand bar; a "1sec" countdown covers part of the next card; units and effects sometimes cover tower HP numbers.

### 6.2 Per-frame readers

**Hand reader (milestone 2, done).** Template-matches the middle of each card's API art against every slot, after local contrast normalisation (handles the greyed-out wipe). Charged evo and hero cards are drawn ~10% larger and matched at that size. A read counts when its score is ≥ 0.45 *and* it beats the next-best card by ≥ 0.20. Per slot it reports card, form (normal / evo / hero), greyed, selected (card raised ~31 px), or empty / unknown. `infer_deck` picks a match's 8 cards; restricting to them cuts read time from ~450 to ~28 ms/frame.

Generalisation check (Princess Gambit, `gAibX9rhMfo`, random 40-card pool, iPad-aspect capture): a second layout (`mohamed_landscape_ipad`) whose hand-bar regions are Alina's mapped by one scale + offset (`ui_scale` 0.557). Templates are built at each layout's `ui_scale`; the tiny next-card slot searches three sizes. Dozens of cards read correctly without a known deck, including hero Bowler. Tests cover both layouts (28 passing).
Each reader returns `(value, confidence)`. Readers are independent, and each is unit-tested against a small set of hand-checked crops.

**Elixir reader (milestone 3, done).** Reads the bar fill: pink segments are whole elixir; a lighter-blue sliver in the next segment is the fraction (e.g. 8.58). Colours are judged relative to each bar, because captures differ (one creator's empty blue is brighter than another's partial blue). 20/20 whole numbers matched the on-screen number; over 30 s of play regen measured 2.8 s/elixir (the ladder rate) and drops matched card costs, animating over ~0.3 s.

**Clock reader (milestone 3, done).** Apple Vision OCR (`ocrmac`, ~15 ms) on the timer and its label, tried at 3×, 1× and 2× upscale (3× suits white text but breaks the red last-30-seconds digits). Multiplier by template (`assets/hud/multiplier_x{2,3}.png`). Over Alina's first 330 s at 1 fps: 0 unreadable frames; x2 at 1:00 left, x3 at 1:00 left in overtime; the clock reset marks the match boundary. Elapsed time is only computed for ladder rules; Princess Gambit's overtime runs longer than 2:00, so other modes keep phase + time left only.

### 6.3 Approximate placement (v0.5)
Within ~1 s after a confirmed play, find where new pixels appear on the POV player's half of the arena (frame difference against the pre-play frame) and snap that to the 18×32 tile grid. Stored with its own confidence; `null` if unclear.

### 6.4 Temporal smoothing
Readers are noisy frame to frame. Take the majority value over a sliding window of ~5 frames before event logic runs.

## 7. Storage (LanceDB)

| Table | One row per | Key columns |
|---|---|---|
| `videos` | video | video_id, creator, title, url, resolution, fps, layout_id |
| `matches` | match | match_id, video_id, t_start, t_end, opponent_name, opponent_rating, result, deck (8 card ids), special_slots, tower_troop, card_snapshot |
| `hud_states` | sampled frame | match_id, t_video, t_match, hand[4], next, elixir, multiplier, tower_hp[6], reader confidences |
| `events` | POV play or ability use | match_id, t_video, t_match, kind (play / ability), card, form (normal / evo / hero), slot, elixir_before, elixir_after, confidence, approx_tile, frame_ref |
| `cards` | card | see §5 |

The deck is inferred from the cards that appear in the hand during the match (all 8 show up within the first few cycles).

## 8. Validation

- **Ground truth:** Cole reviews the event logs for 2–3 matches from the source video, marking each event correct / wrong / missing.
- **Review tool:** a generated HTML page per match: one row per event, with thumbnails of the hand before/after and a link to the timestamp in the video.
- **Metrics:** play precision and recall, card accuracy, timing error (target ≤ 0.5 s), match-boundary accuracy.
- **Pass bar for v0:** ≥ 95% precision and recall on POV plays across reviewed matches.

## 9. Milestones

0. ~~Card metadata ingestion~~ ✅
1. ~~Calibration + box overlay check on the Alina video~~ ✅ (`config/layouts/alina_portrait.json`)
2. ~~Hand + next-card reader (the backbone)~~ ✅
3. ~~Elixir + timer + multiplier readers~~ ✅
4. ~~Match segmentation~~ ✅
5. ~~Play event logic + confidence~~ ✅
6. ~~Write to LanceDB + review page~~ ✅ (`bot77 process`, `bot77 review`)
7. Cole reviews → fix → re-measure
8. Tower HP + opponent info readers
9. Approximate placement (v0.5)

## 10. Tech

- Python 3.12 via `uv`
- `ffmpeg` for decoding; OpenCV for crops and template matching
- Apple Vision OCR via `ocrmac` (fast on Apple Silicon, no model to train) for timer / HP / names
- LanceDB OSS for storage
- `pytest` for reader unit tests against hand-checked crops

## 11. Open questions

1. ~~Name labels on deploy~~ Resolved: the "The Log lvl.16" label at ~2:22 was Alina's own Log (the deck runs Log). Labels appear for the POV player's plays only, which matches Cole's experience. Used as a fourth signal in §3.
2. ~~Evo cycles and ability data~~ Done, except Ice Wizard's hero ability.
3. How to handle matches from special modes, if any appear in the videos?
4. Spirit Empress sits in the spell id range — spell, troop, or both?
