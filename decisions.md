# Decisions log

What Cole decided, contributed, and reviewed while building bot-77, in order. Numbers come from
the scoring runs at the time (`uv run bot77 benchmark`, review exports in `data/ground_truth/`).
Technical detail lives in `docs/play-detector-spec.md` and `docs/opponent-plays-spec.md`.

## 2026-09-24 — The idea and the plan

- **Goal:** train a model to get good at Clash Royale from the large amount of pro gameplay on
  YouTube, and use LanceDB (open source) to mine and label moments as scenarios (e.g. being
  outcycled).
- **Build order:** Cole wanted the analysis engine (C: turn videos into a play-by-play log in
  LanceDB) and wanted "the foundation right before we move on." He first proposed A → B → C;
  since the predictor (A) and the autonomous player (B) both need C's data, the order became
  C → A → B.
- **Compute:** local first (M5 MacBook, 16 GB), rent cloud GPU if needed (budget: a few hundred dollars).
- **Scope: one deck.** Hog 2.8 EQ — Hog, Earthquake, Tesla or Cannon (evo), Firecracker (evo),
  Skeletons, Electro Spirit, Mighty Miner (champion), and Log or Barbarian Barrel. Cole plays
  this deck himself (peaked 1800 Ultimate Champion, ~10k wins), so he can judge the output.
- **Chose the source creators:** Alina (@AlinaCR1) and Ian77, both strong Hog EQ players, and
  chose to stay with YouTube rather than in-game replays because there's far more of it.
- **Asked to spec before building** ("better to get it right the first time").

## 2026-09-24 — Domain knowledge Cole supplied (and corrections he made)

- **Corrected a wrong assumption:** card-name labels ("Hog Rider lvl.16") only appear for your own
  plays, not the opponent's. So opponent plays need sprite recognition.
- **Deck structure:** 1 evo slot, 1 hero/champion slot, 1 flex slot. The first champion/hero
  ability button sits on the right; a left button only appears with a second ability.
  Tesla and Cannon are never in the same deck.
- **Card API quirks:** `maxEvolutionLevel` 1 = evo, 2 = hero, 3 = both (confirmed on all 123
  cards); `maxLevel` counts upgrades, so starting levels are 1/3/6/9/11 by rarity; Heal Spirit's
  id is in the spell range because it used to be the Heal spell; Mirror costs the mirrored card + 1;
  tower troops are `supportItems`.
- **Supplied the evo, hero and champion tables** (cycles, ability costs, descriptions; Boss Bandit
  has 2 ability uses with a 3 s cooldown) — kept verbatim in `data/cards/sources/`.
- **Confirmed** evo charge pips, champion frames, and the "slot empty after a play" signal.
- **Suggested a Princess Gambit tournament video** (random 40-card pool) as a stress test for card
  recognition. It found the hero-form detection worked, and caught a next-card bug at small sizes.
- **Verified uncertain reads** in that video (hero Knight, evo Goblin Cage, evo Bomber).

## 2026-09-24 — Review round 1: Alina, matches 1–2

- Cole reviewed all 124 detected events (118 real plays).
- **Before fixes: 87% precision, 93% recall.** High-confidence events were already 83/83.
- **Findings from Cole's notes** that drove the fixes:
  - After a play, the elixir bar's fractional sliver vanishes and spent segments drain pale; this
    caused fake Mighty Miner abilities and cost mismatches.
  - Emotes cover the hand and elixir bar and were read as plays.
  - When a card disappears from the hand, it was played, even if the cost looks off.
  - A Skeletons + Mighty Miner ability merge, a Log played on a bush, cards dragged mid-slide.
- **Decision:** recalibrate before reviewing more, so the rest of the reviews are on better output.
- **After fixes: 100% precision, 99.2% recall** on those matches.

## 2026-09-25 — Review rounds 2–3: held-out matches

- **Alina matches 3–4** (not tuned on): **99.1% precision, 97.3% recall** on first scoring.
  Cole spotted the pattern: "Mighty Miner ability then Hog" read as a 5-elixir Hog. → Built an
  ability-button reader; then 100% / 100%.
- **Alina match 5** (fully held out): **72/72**. Cole flagged one Tesla + Mighty Miner pair with
  the right cards but swapped order and elixir split.
- **Ian77, matches 1–3** (second creator, split layout, ~3k opponents, jump-cut edits, fully held
  out): **188/188**. Cole's findings: a card selected when the game ended looked like a play; a
  Barbarian Barrel released with a Hog was missed; low-confidence drops at edit cuts.
- **Cole's conclusion:** "any high conf is correct" — backed by 384/384 high-confidence events.
- **Total POV ground truth: 490 events, 100% precision and recall** after the fixes.

## 2026-09-25 — Moving to opponent plays

- **Decided the POV detector was solid** and moved on to detecting the opponent's plays.
- **Confirmed** the "−N elixir" popup is POV-only, and pointed out the **purple diamond** that pops
  up when the opponent plays an evo card.
- **Added the KataCR paper** (Wu et al. 2025, arXiv:2504.04783) and asked that it always be
  cited. Its code and dataset are MIT; the PDF stays local (arXiv's license doesn't cover
  redistribution, and the repo is public).
- Approved the opponent plan: spawn events → card identity with a deck constraint → opponent elixir.

## 2026-09-28 — Opponent review round 1: Alina, match 1

- Asked for review-page changes: a guessed opponent deck that updates as he corrects cards,
  corrected cards first in the picker, and a one-click "Not a play" button.
- Reviewed all 64 detected opponent deploys and added 6 missed plays (52 real plays).
- **Result: 29 right (45%), 18 wrong card, 17 not a play.** It found 89% of real plays but named
  only 55% correctly end to end.
- **Cole's findings:** many "wrong player" errors (Alina's own units tagged as enemy); Goblinstein
  is two entities (Doctor and Monster) but one card; **the 2024 dataset has neither Goblinstein nor
  Minion Giant**; units split behind the king tower or in crowds were missed.
- **Cole's call:** the 2024 detector is missing a lot and is outdated — proposed dropping it.
  (Decision on what replaces it: see below.)

## 2026-09-28 — Replace the 2024 detector (decided)

- **Decision (Cole):** stop relying on KataCR's 2024 detector weights. Build our own detector
  the paper's way [Wu et al. 2025, §3]: current classes (all 123 cards, sub-units, evo and hero
  forms), trained on KataCR's MIT sprite slices for the cards they cover, plus new slices cut
  from our footage for the missing ones, plus generated arenas.
- **Team** comes from our own signals (badge colour and a cross-check against the POV play log),
  not the detector.
- **Labels** come from reviews: Cole's corrections say which card was where, and the POV log says
  exactly when and where our own cards land. No manual box drawing planned.
- **Quick wins first**, before the new model: drop "opponent" deploys that are really POV plays,
  and recognise composite cards (Skeleton Army, Minion Horde, Goblin Gang, Three Musketeers)
  by unit count.
- Benchmark: `data/ground_truth/opponent_truth_uz4VVzGlOjE_m01.json` (52 plays).
  Baseline: 29/52 fully right.

## 2026-09-28 — Building training data for cards the old detector doesn't know

### Quick wins (done)
- **POV cross-check:** an "enemy" deploy of the same card within −1.5…+4 s of one of our own
  logged plays is our unit. It dropped 5 fake opponent deploys (3 Mighty Miner, mostly his ability
  resurfacing; an Earthquake; an Electro Spirit). Fake deploys 12 → 9; deploy precision 81% → 85%;
  no real plays lost.
- **Composite cards by unit count:** Skeleton Army (≥6 skeletons), Minion Horde (≥5 minions),
  Three Musketeers (3), Goblin Gang (goblins + spear goblins).
- **Opponent scorer** (`bot77.review.score_opponent`) against Cole's 52 labelled opponent plays.

### Sprite slices, round 1 — detector boxes + SAM (done)
- Downloaded KataCR's MIT detection dataset: 4,627 RGBA sprite slices over 154 classes (median 22
  per class), 28 empty-arena backgrounds, 7,380 hand-labelled real frames (validation).
- Installed SAM 2.1 tiny. For every opponent play Cole corrected to a card the 2024 detector
  lacks, the tool follows the units born with that deploy and has SAM cut them out once a second.
- **Cole asked whether the level-16 badge belongs in a slice.** No: badges and HP bars are their
  own classes in the paper's generator [Wu et al. 2025, §3.1], and a red badge would bake the
  team into the sprite. The detected badge/bar boxes are now erased from every slice.
- **Cole's gallery review:** Minion Giant 5 of 7 kept (rejected: one missing its collar, one mostly
  background). Goblinstein 0 of 7 kept: the old detector boxed spell effects, tiles, or half the
  unit, so SAM had nothing clean to cut (one Doctor was just a face; one Monster had a hole where
  the deploy clock sat). Cole's rule: keep only if the whole unit is there.

### Sprite slices, round 2 — click-to-cut (done)
- For cards the old detector can't box, Cole clicks the unit and SAM cuts from the click point.
  Page: 6 frames after each corrected Goblinstein play, entity picker (Doctor / Monster).
- **Cole clicked 34 units on 17 frames (17 Monster, 17 Doctor)** →
  `data/ground_truth/reviews/clicks_goblinstein_uz4VVzGlOjE_m01.json`. SAM produced 32 slices;
  deploy clocks, badges and HP bars the detector saw are erased, and only the largest mask piece
  is kept. Awaiting Cole's keep/reject in the gallery.

### Planned next — mining training data with LanceDB vector search (Cole: "make sure to record we did this")
This is the project's original idea — *use LanceDB to mine moments from pro footage* — turned on
the training data itself.

**The problem it solves.** A card the 2024 detector has never seen (Goblinstein, Minion Giant, the
other 11 new cards, 27 evo forms, 17 hero forms) shows up across hours of footage, but the old
detector either misses it or calls it something else (Goblinstein was labelled electro-giant,
lightning, royal-guardian; Minion Giant was labelled electro-dragon or earthquake). Reviewing every
match by hand to find them doesn't scale.

**How it works.**
1. **Candidate crops.** The unit detector runs over every match (4 frames/s; running now on
   Alina matches 2–5 and all 8 Ian matches), and its boxes are stored in the LanceDB table
   `unit_detections`. Every box, whatever class the old detector gave it, is a candidate unit crop.
2. **Embed every crop.** Each crop is turned into an image embedding (a vector that captures what
   the unit looks like), and the vectors are stored in LanceDB next to the crop's match, time,
   position, team and old-detector class.
3. **Queries = Cole-approved slices.** The approved slices (5 Minion Giant, plus whichever
   Goblinstein Doctor/Monster slices Cole keeps) are embedded the same way.
4. **Vector search.** For each card, LanceDB returns the crops nearest to its approved slices,
   across all footage — including crops the old detector mislabelled — filtered with SQL (enemy or
   friendly, match, time range, crop size).
5. **Human in the loop.** The top candidates go to the gallery; Cole keeps or rejects each. Kept
   ones become new slices *and* new queries, so each round finds more varied poses and angles.
6. **Stop** once each new class has enough slices (KataCR's median is 22; target 20–50 per
   entity), then generate synthetic arenas [Wu et al. 2025, §3.1, Alg. 1] and train the new detector.

**Why LanceDB fits.** Vectors, metadata (match, time, team, box) and the crop images live in one
table on disk; search is local and fast on the M5; SQL filters and vector search combine in one
query; and every round's approvals are just rows, so the dataset's history is kept.

**What Cole does:** approve or reject candidates in the gallery — no box drawing.

### LanceDB vector search — built and first results (2026-09-28)
- Cole approved 3 Goblinstein Monster + 5 Doctor slices from his clicks (a leftover "16" badge the
  detector hadn't boxed was stripped automatically, plus crown remnants on three Doctors).
  Approved set: Minion Giant 5, Doctor 5, Monster 3.
- Built `bot77.mine`: every unit box in `unit_detections` is cropped once a second and embedded
  with DINOv2 ViT-S/14; **15,728 crops from Alina's 5 matches** are in the LanceDB table
  `unit_crops` (3 minutes on the M5).
- Search with the 13 approved slices as queries (max cosine similarity, one hit per match-second,
  crops next to the query slices excluded).
- **First result: the top ~20 Minion Giant hits are all Minion Giants, including a batch from
  Alina's match 3 — a match nobody had reviewed — every one mislabelled "electro-dragon" by the
  2024 detector.** The idea works: LanceDB found a card the old model doesn't know, in footage no
  one looked at.
- 90 candidates (30 per class) went to Cole's gallery for approval.
- Ian's 8 matches detected and embedded: **22,270 more crops (37,998 total in `unit_crops`)**.
  Searching his footage found **Goblinstein Monsters and Doctors in Ian's games** (the old detector
  called them baby-dragon, royal-recruit, mighty-miner) — no Minion Giant there. 80 Goblinstein
  candidates from Ian went to a second gallery.
- Cole asked whether recording his own games to cover every card is worth it. Answer: yes, aimed
  at the gaps (the 13 new cards, evo and hero forms) — mining can only find cards that are in the
  footage, and his own plays come with exact labels from the POV detector.

### Mining round 1 results, and Cole's own footage (2026-09-28)
- Cole reviewed the mined candidates: **35 of 170 kept (21%)** — Alina 16/90, Ian 19/80. Cole's
  view: "not the best." After SAM cut the kept ones, a final pass rejected 7 more (SAM spilling
  into neighbours or effects, and gold level badges on Ian's footage).
- **Approved slices now: Minion Giant 14, Goblinstein Monster 15, Goblinstein Doctor 12**
  (from 5 / 3 / 5 before mining).
- **Cole recorded ~40 min of his own ladder games** (iPhone Mirroring on the Mac, 3024×1964 at
  120 fps; the phone screen is a fixed 876×1900 region). He faced **Goblinstein, Minion Giant,
  Rune Giant, Spirit Empress and Ronin** — three of those have no slices at all yet. He plans to
  cherry-pick other cards later.
- The phone region is cropped and scaled to 1080×2340 so Alina's calibrated layout applies
  unchanged (aspect ratios differ by 0.1%).
- Cole gave the times of the new cards in his recording: **Ronin** (game starts 3:50),
  **Spirit Empress** 8:13, **hero Tombstone + Tomb Queen ability** 15:53, **Rune Giant** 18:52,
  **Goblinstein** 35:20. A click-to-cut page with 50 frames around those times was built straight
  from the original recording (cropped exactly like the converted video, so clicks line up).
  First hero-form entities: `tombstone-hero`, `tomb-queen`.

## 2026-09-28/29 overnight — Cole asleep; everything that didn't need him

- **Cole's recording processed.** iPhone Mirroring puts the game HUD at different offsets than
  Alina's native capture (top HUD +80 px, hand bar −52 px after scaling), found by template-matching
  Alina's HUD patches into his frame; new layout `cole_iphone_mirroring`. Result: 11 matches, his
  deck (Hog EQ with Barbarian Barrel) read every match, 18,654 unit crops embedded, 300 opponent
  deploys from the old detector (its inferred decks are noisy — effects read as Earthquake/Poison/Tornado).
- Cole corrected me: the times he gave were **game starts**, not when the cards appeared. The click
  page was rebuilt from each named game's opponent deploys (80 frames, 2 s after each deploy).
- **Mining round 2** (41 approved slices as queries, all four videos, already-reviewed crops skipped):
  120 candidates. **The top Minion Giant hits are from Cole's own games 1 and 6**, where he faced it.
- **Synthetic dataset v1:** KataCR's generator [Wu et al. 2025, §3.1, Alg. 1] run unchanged except for
  3 added classes (minion-giant, goblinstein-doctor, goblinstein-monster, slices rescaled to the
  generator's 568×896 arena). 12,000 images (~74 boxes each), validation = 800 of KataCR's real
  hand-labelled frames.
- **Detector v1 training:** YOLO11s at 896 px on the M5's GPU. Batch 16 needed 16 GB and thrashed
  (914 s/step), so batch 4 (1.8 steps/s, ~28 min/epoch); 12 epochs fit overnight. No faction output —
  team comes from the colour of the level badge the model detects above each unit, then the POV
  cross-check. Evaluated automatically on the 52-play opponent benchmark when done.
- Morning review queue written to `REVIEW_TODO.md`.

### Detector v1 results (2026-09-29, morning)
- Trained 12 epochs in 5.2 h. On 800 of KataCR's real hand-labelled frames: **mAP50 0.70–0.73,
  mAP50-95 0.46** (the paper's larger, size-split models reach AP50 0.83–0.85 [Wu et al. 2025, Tab. 2]).
- On Cole's 52-play opponent benchmark (Alina match 1):

  | detector | plays found | fully right | wrong card | not a play | missed |
  |---|---|---|---|---|---|
  | KataCR 2024 | 96% | 31 (60%) | 19 | 9 | 2 |
  | bot77 v1 alone | 62% | 22 (42%) | 10 | 12 | 20 |
  | **2024 + v1 for the new classes** | 88% | **32 (62%)** | **14** | 10 | 6 |

- Per card, **v1 learned Goblinstein (2/3 right vs 0/3)** but is worse on small units (Skeletons,
  E-Spirit, Barbarian Barrel), on spells, and still 0/3 on Minion Giant. It runs 4.7× faster.
- Why v1 alone is weaker: trained only on synthetic arenas (domain gap to real footage), a small model,
  12 epochs, and one model for all sizes where the paper splits by size because small units suffer.
- Current best: the ensemble (old detector + v1 for classes the old one lacks). Small gain, and the
  inferred opponent deck now includes Goblinstein.
- Cole: **Spirit Empress has two forms (ground and air)** → two entities `spirit-empress-ground` and
  `spirit-empress-air`, both mapped to the card (like Goblinstein's Doctor/Monster); the form is also
  worth logging for gameplay. Hero Tombstone likewise → `tombstone-hero` + `tomb-queen`.
- Cole: the first click page's Ronin frames were better — Ronin was only played early in that game —
  so Ronin got its own page with 26 frames every 3 s from the start of the game.
- **Cole agreed to the "be safe before paying" plan:** a go/no-go checklist before any rented-GPU run
  (≥40 approved slices per new entity, ≥2,000 auto-labelled real frames with a 100-frame spot check,
  a second opponent benchmark held out of training, a local 1-epoch test and a ~$1 cloud dry run).
- Cole gave **exact card times** in his recording (the earlier pages still missed the cards):
  Ronin 5:09, Spirit Empress air 8:42 and ground 9:55, Tomb Queen 17:13, Rune Giant 20:43 ("until he
  crosses the bridge, for a good view"), Goblinstein 36:05 and 2:15. Little Prince + Guardian at 20:06
  skipped: the 2024 dataset already has `little-prince` and `royal-guardian`.
  → `data/reviews/click_cole_v3.html`, 106 frames, 1 per second (22 for Rune Giant). This page replaces
  the two earlier click pages for his recording.
- Cole clicked **104 units** on the exact-times page (Ronin 13, Spirit Empress air 11 / ground 11,
  Tomb Queen 11, Rune Giant 20, Goblinstein Doctor 20 / Monster 18) → SAM cut 102 slices, now in
  Cole's gallery. Spirit Empress cuts are clean; several Ronin cuts are only his hat.
- Cole showed **Goblinstein's Lightning Link ability** (the beam between Doctor and Monster). Planned as
  its own class `goblinstein-ability`, like KataCR's `skeleton-king-skill`, so opponent ability uses
  can be logged. A thin beam won't SAM-cut cleanly; it'll be labelled as a box on real frames instead.

## 2026-09-29 — Mining hits diminishing returns; coverage check → footage list
- Mining round 3 (all 8 new entities, 41+ queries): Cole kept **36 of 156**. Strong for Minion Giant
  and Goblinstein; for Ronin, Spirit Empress, Tomb Queen and Rune Giant only the top few hits were
  real. **Cole: "a lot of them don't actually have more footage, so mining is kind of useless"** —
  correct: those cards barely appear in our footage, and similarity search can only find what's there.
- Approved slices now: Goblinstein Doctor 41 / Monster 34, Minion Giant 22, Rune Giant 13, Spirit
  Empress air 11 / ground 10, Tomb Queen 7, Ronin 6.
- **Coverage check against all 123 cards** (KataCR's 2024 slices + ours): 31 cards at ≥40 slices,
  41 at 20–39, 38 under 20, 13 with none (8 genuinely new: Berserker, Boss Bandit, Goblin Curse, Goblin
  Demolisher, Goblin Machine, Suspicious Bush, Vines, Void; 5 composites). 27 evo forms and all 17 hero
  forms have no slices. Written up as a recording list: `docs/footage-needed.md`.

## 2026-09-29 — Cole + friend dual-POV recordings; farming evo/hero footage from YouTube
- **Cole's idea: he and a friend both play the new cards in friendly battles, both recording.**
  Each POV log labels its own plays; syncing the two recordings on the in-game clock turns each side's
  plays into exact **opponent** labels for the other — opponent ground truth with no review.
  Built `bot77.pairing` (match pairing by constant clock offset, then opponent truth files
  `data/ground_truth/opponent_truth_auto_<match>.json`).
- Recordings: Cole 3:29 PM (14 min) and friend 3:12 / 3:23 PM (10 / 19 min), friend's Mac at
  2940×1912 (phone region 852×1848). Cole's 3:22:59 recording is black after the first seconds
  (mirroring window) — skipped. Friendly battles on a new teal arena skin; Void, Vines, Goblin
  Demolisher, Rune Giant etc. visible.
- **Layout calibration automated** (`bot77.autolayout`): the hand-made method for Cole's iPhone as
  code. HUD templates for unique elements (timer, chat, "Next:", elixir drop) give the UI scale and the
  top / hand offsets; the tower HP bars are found by *colour* (red opponent bar top-left, blue ours
  lower-left) because templates failed on a different arena skin; only in-game frames are used (menus
  fooled it). Agrees with the hand-made layout to within 0–22 px; friend's and Oyassuu's layouts
  verified on calibration sheets.
- **Cole picked 11 YouTube videos** for evo and hero forms, mostly raw footage without facecam:
  OYASSUU (2.6 Hog: evo Cannon, hero Ice Golem, evo Skeletons; 2.1 Log Bait: evo Dart Goblin, evo
  Goblin Barrel; evo Lumberjack; Queen + evo Ice Spirit), Ryley (evo Barrel, hero Knight, evo Princess;
  evo Royal Giant, hero E-Wiz, evo Royal Ghost, and opponent evo Zap, hero Mega Minion; evo Bats, evo
  Tesla, hero Ice Wizard; Little Prince, evo Ice Spirit; hero Goblins, evo Dart Goblin, evo Skeleton
  Barrel; a 5-deck video with evo Furnace, hero Valkyrie, evo Skeletons, evo Baby Dragon, evo Knight,
  evo Witch, evo Snowball; evo Royal Hogs, hero Berserker).
- **Hero Electro Wizard isn't released yet** (a few days out), so it's not in the card API snapshot.
  Decision: add it as its own class now; the card data catches up with the next API snapshot.
- **Bug found:** LanceDB's `table_names()` returns only the first 10 tables by default; with 12 tables
  the pipeline thought `videos` didn't exist and crashed at the save step, after the full frame pass,
  losing that run for all three recordings. Fixed everywhere (`table_names(limit=10_000)`); rerun two at
  a time. The decks read correctly: Cole (Rune Giant, Minion Giant, Goblin Machine, Suspicious Bush,
  Goblinstein/Berserker, Spirit Empress, Boss Bandit, Ronin) and his friend (Vines, Void, Goblin Curse,
  Rune Giant, Berserker, Goblin Demolisher, Goblinstein, Mirror) — **every Tier 1 new card is covered.**
- **Pairing, first attempt was wrong:** each of Cole's games paired with several of his friend's at
  different offsets. The match clock runs at real speed in every game, so any two overlapping matches
  line up at *some* constant offset. **Fix:** the recordings' wall-clock start times (from the file
  names) give the expected offset (Cole 3:29:18 − friend 3:23:01 = 377 s); pairs must sit within 20 s of
  it. Result: 3 pairs at 375–376 s. Independent check: in all three, Cole's opponent reads as his friend
  ("Nathan'S gas") and the friend's opponent as Cole every time. Friend's 3:12 recording and 3 of his
  3:23 games have no partner (Cole's 3:22:59 recording was black).
- **123 exact opponent labels with no review**, covering every Tier 1 card: Berserker 20, Rune Giant 11,
  Goblinstein 11, Goblin Demolisher 11, Ronin 10, Boss Bandit 8, Goblin Machine 8, Vines 7, Goblin Curse
  6, Void 6, Minion Giant 6, Spirit Empress 5, Suspicious Bush 5, Mirror 3, abilities 6.
- Paused here at Cole's request (going home); the YouTube batch (4.6 h of footage, two at a time)
  waits for his go.

### Between sessions: which cards to record next (2026-09-29, afternoon)
- Cole asked whether more recordings would help. Counted POV plays per new card across his, his friend's
  and his ladder recordings against a ~15-play target: done — Berserker 32, Goblin Demolisher 25,
  Goblinstein 24, Rune Giant 22; nearly — Void 14, Vines 13; short — Goblin Curse 11, Ronin 10, Boss Bandit 8,
  Goblin Machine 8, Minion Giant 6, **Suspicious Bush 5, Spirit Empress 5** (both forms). Hero forms: only
  Berserker (9); evo forms: only Firecracker and Tesla. Suggested one deck built from the short cards,
  2–3 plays of each per game, both forms of Spirit Empress, abilities used, ~5–6 games, both recording.

## 2026-09-29 evening — 75 more minutes of dual-POV footage; everything processed in parallel
- Cole and his friend recorded ~38 min each (6:10 / 6:11 PM), aimed at the short cards (Suspicious Bush,
  Spirit Empress both forms, Boss Bandit, Goblin Machine, Ronin, Minion Giant, Goblin Curse, Vines/Void)
  plus **hero Tombstone + Tomb Queen, hero Dark Prince with its ability, and hero Barbarian Barrel**.
  No black frames this time. Expected pairing offset 67 s.
- **Parallel batch** (`scripts/run_all.py`): 13 videos (both new recordings + 11 YouTube), 2 CPU workers
  (convert + clock/hand/elixir pass) and 2 GPU workers (unit detection + DINOv2 crops), ~10 GB peak on the
  16 GB M5. Ryley's layout: auto-calibration picked his pink name text as the red tower bar on his red
  arena, so its arena rows were set by hand from measured bars (opp 445 / own 1477 px).
- **Auto-cut candidates were tried and dropped:** taking "the unit born with the play" from the old
  detector's tracks mislabels units when plays land close together; locating the drop with the on-screen
  name label ("Goblin Machine lvl.11") helps, but yield was 5 of 12 and unverifiable. Review time is the
  scarce resource, so **Cole gets click pages instead** (the method that gave 102 of 104 usable slices):
  `data/reviews/click_new.html`, `click_evo.html`, `click_hero.html`, frames 1.5 s and 3 s after each
  labelled POV play (8 plays per new card, 6 per evo/hero form), built by `scripts/build_review_queue.py`.
- **Cole asked to start everything at once, sized for the M5 / 16 GB**: 2 CPU + 2 GPU workers
  (above that, memory thrashes — the batch-16 training run showed it).
- **Batch results (19:59–23:07, 0 errors):** Cole 6:11 PM recording 10 games / 237 POV plays; friend 6:10 PM
  16 games / 334 plays; OYASSUU 2.6 Hog 7 games / 342; Ryley Log Bait 6 / 308; Ryley hero E-Wiz 4 / 106; plus
  Ryley's other five videos. Cole's recording was 240 fps, so its conversion alone took ~20 min.
- **Three OYASSUU videos found 0 games.** His HUD sits ~40 px lower in those than in the video his layout
  was calibrated on (a different recording setup), so the clock was never read. Each got its own
  auto-derived layout (`oyassuu_<video>`; clock read on 5/5 test frames), and `scripts/rerun_zero_match.py`
  now reruns *any* video that produces 0 games with a per-video layout — lesson: calibrate per video,
  not per creator. The reruns started 23:07; pairing (Cole 6:11:20 − friend 6:10:13 = 67 s) and the
  click pages build automatically when they finish.
- **Reruns done (00:30, 0 errors)** — all three OYASSUU videos found their games with per-video layouts.
- **Pairing, second session:** 9 of Cole's 10 games paired with his friend's at 65–66 s (expected 67), →
  **532 exact opponent labels in total** across both sessions, no review: Vines 38, Mirror 37, Goblinstein 37,
  Tombstone 37, Ronin 35, Goblin Curse 31, Boss Bandit 30, Void 30, Goblin Machine 28, Spirit Empress 27,
  Minion Giant 27, Suspicious Bush 25, Berserker 20, Dark Prince 18, abilities 32, and others.
- **POV plays now available for the new cards** (high confidence, dual-POV videos): every one ≥12 —
  Goblinstein 40, Vines 35, Void 32, Ronin 26, Goblin Curse 21, Boss Bandit 17, Goblin Demolisher 16,
  Berserker 15, Rune Giant 15, Minion Giant 14, Suspicious Bush 14, Goblin Machine 13, Spirit Empress 12.
  Cole's targeted session fixed every short card.
- **Hero forms found:** Berserker 36, Tombstone 33, Knight 30, Ice Golem 29, Goblins 28, Dark Prince 14,
  Barbarian Barrel 6, Valkyrie 5, Ice Wizard 4, Mega Minion 2. **Evo forms:** Cannon 41, Firecracker 33,
  Dart Goblin 24, Goblin Barrel 22, Ice Spirit 16, Tesla 15, Skeleton Barrel 12, Bats 10, Royal Hogs 9,
  Skeletons 8, Goblin Drill 5; only 1–2 plays of Baby Dragon, Battle Ram, Giant Snowball, Knight, Royal
  Ghost, Royal Giant, Witch.
- **Hero E-Wiz wasn't found**: the hand reader identifies cards by their API art, and it isn't in the API yet.
  It'll need its art (or a click-labelled example) once released.
- Review pages built: `click_new.html` (208 frames), `click_evo.html` (146), `click_hero.html` (106).

## 2026-09-30 — Click round on new / hero / evo pages
- Cole clicked **439 units**: new cards 208 (on 195 frames), hero forms 102, evo forms 129. His note: "I think
  you missed a few" — the pages had no usable frames for **Spirit Empress ground, Tomb Queen and dismounted
  Dark Prince** (the 1.5 s / 3 s offsets after the play were too early for ability summons and the form change).
- Cole asked whether to click the POV unit or the opponent's: the POV one (the frame follows its play); the
  opponent's copy of the same card is fine too, since slices are team-agnostic. For hero cards: click the
  hero unit, plus separate summoned units (Tomb Queen, dismounted Dark Prince); not transparent effects.
- SAM cut **401 slices** (new 200, hero 87, evo 114) → three approval galleries. Bug fixed: slices from
  multi-card pages were labelled with the page name ("hero") instead of each click's card.
- Cole's gallery pass on the new-card slices: **74 of 200 kept (37%)**. Weak ones: Rune Giant 2/16,
  Berserker 3/12, Goblin Curse 3/13, Void 3/16, Vines 0/12. Cole asked whether spells with troops inside
  count: no — a troop baked into a spell slice teaches the wrong thing. **Vines always wraps a troop**, so it
  has no clean slice at all. Decision: attached/transparent spell effects (Vines, Void, Goblin Curse,
  Earthquake, Rage, Freeze…) are learned from **boxes on real frames** (overlapping a troop's own box where
  they cover one), using the paired and POV labels, spot-checked by Cole — not from slices.
- Approved slices now: Goblinstein Doctor 47 / Monster 40, Minion Giant 34, Spirit Empress air 17 / ground 10,
  Rune Giant 15, Ronin 12, Suspicious Bush 10, Tomb Queen 7, Goblin Machine 7, Boss Bandit 5, Goblin
  Demolisher 5, Berserker 3, Goblin Curse 3, Void 3. Hero and evo galleries not reviewed yet.
- Approved slices are now committed (`data/slices/approved/`, 9 MB); candidates stay out of git.
- Cole's hero and evo gallery passes ("some were kinda brutal"): **hero 31 / 87 kept, evo 47 / 114**.
  Near-zero for Mega Minion hero 0/4, Valkyrie hero 1/9, Knight hero 2/9, Goblins hero 2/7, Berserker hero
  2/9, Cannon evo 2/9, Firecracker evo 2/10, Ice Spirit evo 1/8. Across all click rounds, SAM from a single
  click keeps ~35–40% — it spills into neighbours and effects on busy frames.
- **Conclusion:** slices alone won't reach 40 per entity. The main source for the paid run becomes
  **auto-labelled real frames** (boxes, from POV plays + paired opponent labels), with slices supplementing
  the generator; Cole spot-checks a sample instead of approving each one.
