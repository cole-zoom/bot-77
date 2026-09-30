# Opponent Plays v1 — Spec

Status: draft
Builds on the POV play detector (`docs/play-detector-spec.md`) and on KataCR [Wu et al. 2025] — see `docs/references.bib`. Method sections below cite the paper by section.

## 1. Goal

For each match, a log of the **opponent's** plays: when, where, which card, and a running estimate of their elixir. Accuracy judged the same way as POV plays: Cole reviews, ground truth goes in `data/ground_truth/`, `bot77 benchmark` scores it.

## 2. What's on screen (and what isn't)

- No opponent hand, elixir bar, name labels or "−N elixir" popups — those are POV-only (confirmed by Cole).
- **Level badges are team-coloured**: opponent units carry red "16" badges and dark-red HP bars, ours are blue. A new red badge = a new opponent unit.
- **Evo popup**: when the opponent plays an evo-capable card, a purple diamond pops up at the deploy (Cole). It marks the card as their evo slot and one cycle closer to charged.
- Spells have no badges; they show as effects (Log rolling, Fireball, Zap flash, EQ ring…).
- Both players have 8-card decks cycled as a queue of 4 in hand [Wu et al. 2025, §2.2], so once 8 distinct opponent cards are seen, the deck is known.

## 3. Pipeline

### Step 1 — Spawn events (identity-agnostic)
Per frame, find red level badges (colour + the badge shape) on the arena; track them (IoU / ByteTrack [Wu et al. 2025, §5.2]). New badges appearing together within ~0.5 s and close together = one opponent deploy: `t, tile (18×32 grid [§2.3]), unit_count`. Purple evo popups attach `evo=True`. Output reviewed by Cole: "was this a real opponent play?"

### Step 2 — Which card
- **Detector**: test KataCR's released YOLOv8 combo (MIT; two size-split detectors, AP50 84.3 on their validation set [§5.2, Tab. 2]) on our footage. If it transfers, use it; otherwise train our own on their MIT generative dataset (sprite slices pasted in layer order onto empty arenas, rare classes oversampled, overlap filtering [§3.1, Alg. 1]).
- **Deck constraint**: classify each spawn among the opponent's cards seen so far plus "new card", narrowing to 8 once the deck is known.
- **New cards / evos / heroes since May 2024**: add sprite slices via the paper's model-assisted labelling + SAM loop [§3], seeded with free labels from our own plays (every POV play gives card + time + rough location, so the spawned blue units are labelled examples of the same sprites).
- Spells: detector classes for spell effects where available; otherwise effect templates.

### Step 3 — Opponent elixir
Start at 5 (ladder), regen by the clock's multiplier, subtract each identified card's cost. Used as a plausibility check (can't go below 0) and as a feature for scenario mining (elixir advantage, outcycling).

## 3.1 Detector coverage (KataCR 2024 classes vs today's cards)

105/123 cards have a detector class. Of the 18 without one, 5 are made of classes it does have and can be recognised by unit count (Goblin Gang = goblins + spear goblins, Minion Horde = 6 minions, Skeleton Army = many skeletons, Three Musketeers = 3 musketeers, Mirror = a repeat). **13 are genuinely new**: Berserker, Boss Bandit, Goblin Curse, Goblin Demolisher, Goblin Machine, Goblinstein, Minion Giant, Ronin, Rune Giant, Spirit Empress, Suspicious Bush, Vines, Void. Also missing: evo classes for 27 evo forms, and hero classes for all 17 hero forms. Found by Cole's first opponent review (Goblinstein, Minion Giant).

Cards with two entities (Goblinstein's Doctor + Monster) get one class per entity, both mapped to the card; they spawn together, so deploy grouping yields one play and the deck counts the card once.

Fine-tuning (milestone 6) is therefore required, not optional. Its labels come from reviews: every corrected opponent play gives card + time + place to cut new sprite slices [Wu et al. 2025, §3].

## 4. Validation
Review page gains opponent rows. Metrics: spawn precision/recall (step 1), card accuracy (step 2), elixir-estimate error at checkpoints where it's inferable.

## 5. Milestones
1. Red-badge spawn detector + review on 2 matches
2. KataCR detector feasibility test on Alina / Ian frames
3. Card identity with deck constraint
4. Spells
5. Opponent elixir estimate
6. Fine-tuning with POV auto-labels + new-card slices

## 6. Attribution
KataCR code and dataset are MIT; keep their license notice with any vendored code or data (`third_party/katacr/LICENSE`). Ideas and algorithms from the paper are implemented with citation to [Wu et al. 2025].
