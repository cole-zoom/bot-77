# Opponent decks: scripted opponents that punish the bot's mistakes (2026-11-04)

Cole: "I wanted to make like the general rules of how to play each deck to counter the bots bad plays ... I wanted it
to play against certain decks, and each deck have its own scenarios, that teach stuff."

One file per standard deck. Each file is the playbook for a **scripted opponent** in the simulator. The opponent plays
that deck the way a good player does and has punish rules for the bot's mistakes. Each file also has the
**scenarios** (start states) that teach the bot something against that deck. The bot always plays our deck: Giant,
Battle Ram, Mini P.E.K.K.A, Mega Minion, Bomber, Goblin Hut, Fireball, Arrows.

Source: one YouTube guide per deck (the newest how-to with real commentary, 2026-11-04). The transcripts stay local in
`data/external/guides/` (gitignored). These files paraphrase them and cite timestamps. Anything marked
**[inferred]** isn't in the video; Cole checks those.

| Deck | File | Source |
|---|---|---|
| 2.6 Hog Cycle | [hog-2.6.md](hog-2.6.md) | Leviathan CR, 2026-10-18 |
| Golem Beatdown (Golem + Hog, no spells) | [golem-beatdown.md](golem-beatdown.md) | Sweaty Clash, 2026-09-30 |
| P.E.K.K.A Bridge Spam | [pekka-bridge-spam.md](pekka-bridge-spam.md) | CheddarCR, 2026-10-27 |
| Miner Control (Miner Rocket) | [miner-control.md](miner-control.md) | Nate, 2023-09 (the newest, Mohamed Light 2026-01, is mostly stream chat) |
| Giant Double Prince | [giant-double-prince.md](giant-double-prince.md) | Sweaty Clash, 2025-09; giodelion, 2025-10 (the newest, Kamikaze CR 2026-10-26, has no commentary) |
| Classic Log Bait | [log-bait.md](log-bait.md) | Ryley, 2026-08-25 |

## The board (engine frame, `third_party/clash-royale-env/include/core/ArenaLayout.h`)

Placements are (x, y) on the engine's 18 x 34 board. The bot is team 0 and defends low y; the scripted opponent is
team 1 and defends high y. River band y 15.5-17.5. Lanes x 3.0 / 14.0, bridges x 2.5 / 14.5, middle x 8.5. Towers:
the bot's princess (3, 6) / (14, 6), king (8.5, 2.5); the opponent's princess (3, 27) / (14, 27), king (8.5, 30.5).
The opponent's bridge spot is about y 18.5, its back y 32-33, its middle defensive spot (8.5, 24.5). The sim calls
our Mini P.E.K.K.A "Mini PEKKA".

## Rules every scripted opponent follows

1. **It plays a legal deck.** It has its 8 cards and a normal 4-card hand with a cycle, and it pays elixir for every
   card. It never gets a card it couldn't have.
2. **It reacts like a person.** It answers 0.5–1.0 s after seeing something, with a seeded random delay.
3. **It can't see the bot's hand.** It reads the board, the towers, its own hand and elixir, and the cards the bot
   has played. Counting the bot's elixir is fair, because good players do it.
4. **Rules trigger on the kind of mistake, not the exact play.** One rule ("the bot spent more than 5 in one lane")
   covers every way of doing it. Rules are checked top to bottom every tick. When none fires, the deck's default plan
   runs.
5. **We score outcomes**: tower damage and the elixir trade. We never score "did it play card X".
6. **Every scenario is checked before training.** In the sim, the bad play has to actually lose, and a sound line has
   to exist. A scenario that fails is fixed, not trained on.
7. **Every rule that fires is logged.** A bot play that no rule punishes and that wins goes on a gap list, and Cole
   adds a rule.

## Template

```
# <Deck name>
Source: <title> · <channel> · <date> · <link>

## Deck
<8 cards with elixir costs, average cost, 4-card cycle cost> · sim: all in the engine / missing: ...

## Game plan
2-4 lines: how this deck wins and what it's scared of.

## Card roles
One line per card: what it's for (win condition, building, spell, cycle, support, tank killer).

## Opening
What it plays at 0:00, what it does if the bot plays first, and what it never opens with.

## Against our deck
How it defends each of our cards (Giant, Battle Ram, Mini P.E.K.K.A, Mega Minion, Bomber, Goblin Hut, Fireball,
Arrows), and which of our cards hurt it.

## Punish rules (checked top to bottom, every tick)
1. IF <observable trigger> → <cards, placement, timing>
2. ...
DEFAULT: what it does when no rule fires (single elixir, double elixir, overtime).

## Scenarios
3-5 start states that teach the bot something against this deck:
- NAME · clock · both elixirs · board · opponent hand + next · what the bot should learn · the bad play the
  opponent punishes

## Open questions for Cole
```
