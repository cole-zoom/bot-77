---
name: review-bot-plays
description: Review the bot's Clash Royale plays (B v0) the way Cole does — OK or Mistake per play, with the card it should have played or "wait", and a short note — from the frames saved at each play. Use when asked to review bot games, fill in a review page, or pre-review plays for Cole to double-check.
---

# Reviewing bot plays

The bot saves, per play, the frame at the play (its tap marked in yellow) and 2 s later
(`data/raw/live_games/<game>/<k>_at.jpg`, `<k>_after.jpg`); every play is a row of the LanceDB table `bot_actions`
(card, slot, tile x / y, reason = `defend` / `idle` / `full elixir` + `(rules)` / `(playbook)`, elixir, hand, threats).
Review pages come from `scripts/build_bot_review.py`. Cole is the expert (1800 Ultimate Champion); his exported reviews
live in `data/ground_truth/reviews/bot_review_*_cole.json`. Read the latest of them before reviewing — they are the
ground truth for these principles, and his notes win over anything below.

## Principles (from Cole's first 60 reviews, 2026-10-22)

**When to play**
1. **Every play needs a reason**: answering an attack that is coming, or building / supporting a push, or spending
   elixir that is about to be wasted (~9+ and nothing better). A play with none of these is a mistake → "wait".
   Cole: "Whats the point? were not even at a full elixir bar?"
2. **Opening (first ~10 s, no enemy plays yet):** wait for the opponent to commit, or cycle a cheap card you don't
   need behind the king tower (e.g. Minions split behind the king so they don't get spelled). With a heavy deck,
   waiting is usually right.
3. **Defenders wait for the attacker to cross the bridge**, then go *on top of / in front of* it. Playing a defender
   at an enemy unit still on its own side is a mistake ("we should have waited for the spear goblins to cross, and
   then placed the knight on top of that").
4. **Not every enemy unit is a threat worth elixir**: small, cheap, slow or ranged units that won't take a tower
   (a couple of Archers) are not worth a 3-elixir answer. "archers here aren't rlly a threat and not worth burning 3
   elixir."

**What to play**
5. **Keep an answer for air.** Don't cycle several air-hitting cards (Musketeer, Archers, Minions, Arrows) in a row
   with nothing to answer — that leaves the deck open to Balloon / Lava. Musketeer is the most valuable defender in
   this deck: don't throw it away unprotected.
6. **Answer the push that is actually coming.** If a Giant / win condition is walking down one lane, the next play
   must deal with it (or be a deliberate counter-push in the other lane); a random play elsewhere is a mistake.
7. **The higher-DPS answer is better against tanks** (Mini P.E.K.K.A over Knight on a Giant).
8. **Spells must hit everything they're aimed at** (Arrows that miss the Spear Goblins are a mistake) and never
   land on nothing ("what are u aiming at lol").

**Giant / pushes**
9. **Giant goes in the back** (behind the king tower) to build a push the opponent can't stop. **Never Giant at the
   bridge.** Don't send it into a defender before that defender has crossed (it just feeds them value).
10. **Support the Giant**: the plays after a Giant should go behind it (Musketeer / Archers / Minions), not elsewhere.
11. **Counter-pushing the opposite lane while defending is fine** ("apply pressure opposite and defend with mini
    pekka").

**Placement**
12. **Mini P.E.K.K.A directly on its target** (e.g. on top of the Musketeer) so it kills fast and survives.
13. **Ranged troops away from the tank** they shoot (Archers / Musketeer further back from a Giant).
14. **Musketeer in front of the king tower** when defending, so both towers protect it.
15. A defender placed where it can't reach the attacker before the tower takes damage is "bad placement".

## How to review

1. List the plays to review (unreviewed ones on a review page, or a game's `bot_actions`).
2. Make sheets: `uv run python scripts/review_sheets.py <plays.json> <out_dir>` — 4 plays per image, each with the
   frame at the play (yellow circle = the bot's tap) and 2 s later, labelled with id, card, reason, elixir, hand,
   threats. Look at every sheet; read the game in order (earlier plays explain later ones).
3. For each play decide, using the principles and Cole's notes:
   - `verdict`: `ok` or `bad`
   - `card`: for `bad`, the card it should have played from the hand shown, or `wait (no play)`
   - `note`: one short line in Cole's voice — the principle broken, and what to do instead
   - `confidence`: `high` / `medium` / `low` (low when the frame doesn't show enough: hidden units, effects)
4. Write the export (same shape as the review page's Export, so the page can show it):
   `{"exported_at": ..., "reviewer": "claude", "plays": [{"id", "game", "k", "verdict", "card", "note", "confidence"}]}`
   to `data/ground_truth/reviews/bot_review_<n>_claude.json` (also keep `bot_card` = the card the bot played, since
   `card` is the card it *should* have played), and rebuild the page with the same flags it was built with plus
   `--prefill <cole export> <claude export>`. Claude's verdicts show with a "Claude · confidence" tag and an Agree
   button; anything Cole touches becomes his (`by: "cole"` in the export).
5. Never overwrite Cole's reviews; when he corrects a Claude verdict, add the principle he used here.
