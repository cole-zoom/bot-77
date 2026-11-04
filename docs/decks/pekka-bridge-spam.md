# P.E.K.K.A Bridge Spam
Source: 20 Wins with Classic PEKKA Bridgespam Guide! 2026 Gameplay+Tips · CheddarCR · 2026-10-27 ·
https://www.youtube.com/watch?v=smkIEE4TAPo

Placements are (x, y) in the engine frame: 18 x 34, cells centred on integers, river y 15.5-17.5. The bot (team 0)
defends low y; this opponent (team 1) defends high y. Lanes x 3.0 / 14.0, bridges x 2.5 / 14.5, middle x 8.5.
Opponent spots: **bridge** (2.5 or 14.5, 18.5) · **high** (lane x, 20.5), support behind a bridge push · **low** (lane
x, 24.5), in front of its princess tower · **middle** (8.5, 24.5), where the existing script puts a defensive card
(`_tempo_opponent` in `src/bot77/rl/situations.py`) · **back** (3.0 or 14.0, 32). The bot's bridge is y 14-15; the
bot's back is y ≤ 8.

## Deck
P.E.K.K.A 7 · Battle Ram 4 · Electro Wizard 4 · Magic Archer 4 · Fireball 4 · Bandit 3 · Royal Ghost 3 · Zap 2 ·
average 3.9 · 4-card cycle (Zap, Bandit, Ghost, Ram) 12 · sim: all in the engine (P.E.K.K.A is `P.E.K.K.A.`).
The video runs Hero E-Wiz (1:04), Evo Royal Ghost (15:32) and Evo Battle Ram (16:05); it says Evo Zap would help in
the bait matchup but he doesn't have it (18:44). It uses Fireball, not Poison (3:43, 14:28). The scripted opponent
plays all eight in normal form.

## Game plan
Make a strong push with P.E.K.K.A in the back during single elixir, get damage, then hold the lead into overtime
(11:50, 12:52). Every time the opponent commits to one lane, punish the other lane with Ram/Bandit/Ghost (3:43, 6:28),
and go in hard when the opponent's key counters or spell are out of cycle (14:28, 15:32). Its only answers to big air
or splash pushes are E-Wiz and Magic Archer, so one bad defensive card loses the game (3:10).

## Card roles
- **P.E.K.K.A**: tank killer and main win condition on a counter-push; at the back in single, at the bridge when the
  bot's ground counters are out of cycle (8:39, 11:50).
- **Battle Ram**: win condition and bridge pressure; forces a response so the opponent can't spell its support (16:05).
- **Bandit**: second win condition, opposite-lane pressure, kills small troops with the dash (2:07, 12:52).
- **Royal Ghost**: cheap defense and a reset/cycle card in the other lane (8:39, 9:43).
- **Electro Wizard**: main air/swarm defense; the stun resets charges and lock-ons; bridge-blocks (6:28, 17:09).
- **Magic Archer**: placed in the middle or high to line up through several targets and the tower (2:38, 11:19).
- **Fireball**: kills supports and finishes towers (3:43, 17:40).
- **Zap**: resets (tower targeting, Sparky/Inferno charge), kills swarms (1:36, 3:10, 8:06).

## Opening
- At 0:00 it waits to ~9 elixir, then cycles a cheap card at the back: Bandit or Ghost (8:39, 13:56). **[inferred]**
  timing.
- Bot plays first: if it's a troop at the bridge, defend it low and build the counter-push (E-Wiz + P.E.K.K.A, 1:04).
  If it's a slow push from the back, go opposite lane (rule 1).
- Never opens P.E.K.K.A or Battle Ram at the bridge into a full-elixir bot. **[inferred]**

## Against our deck
All **[inferred]**; video only shows the same answers against other decks.
- **Giant**: P.E.K.K.A low in the Giant's lane, (3.0 or 14.0, 24.5), dropped when the Giant reaches y ≈ 19.5 so it
  meets it before the tower. Without P.E.K.K.A: tower + Ghost/Bandit on the Giant while E-Wiz handles support. Then counter-push the
  surviving P.E.K.K.A with Ram or Bandit at the bridge in front of it.
- **Battle Ram**: Ghost or Bandit low in the lane; E-Wiz stun resets the charge. Never P.E.K.K.A on a lone Ram.
- **Mini P.E.K.K.A**: E-Wiz stun + Ghost; Magic Archer kites. Keep Mini P.E.K.K.A away from P.E.K.K.A with Zap/E-Wiz.
- **Mega Minion**: E-Wiz or Magic Archer. It's the bot's best card against a P.E.K.K.A push (P.E.K.K.A can't hit air).
- **Bomber**: Bandit dash or Ghost; Zap does not kill it.
- **Goblin Hut**: Spear Goblins pull and chip P.E.K.K.A. Magic Archer through the Hut; Fireball the Hut only with
  goblins or a tower in the radius. Battle Ram goes in the lane where the Hut isn't on its path.
- **Fireball**: never place E-Wiz and Magic Archer within 2.5 tiles of each other unless the bot's Fireball is out of
  cycle (same rule the video uses against Lightning, 8:06).
- **Arrows**: kill nothing in the deck at tournament level; it ignores them.
- Our cards that hurt it most: Mini P.E.K.K.A and Mega Minion (they kill or out-trade P.E.K.K.A), Fireball on stacked
  support.

## Punish rules (checked top to bottom, every tick)
"Out of cycle" = the bot played that card fewer than 4 plays ago (8-card deck). Bot elixir is the opponent's own
count of the bot's spending.
1. IF the bot plays a Giant (or any troop ≥ 5) at its back (y ≤ 8) AND opp elixir ≥ 4 → Battle Ram at the
   **opposite** bridge (2.5 or 14.5, 18.5) within 1 s; Bandit behind it at (same x, 20.5) if opp elixir ≥ 7 and bot
   elixir ≤ 4. Defend the Giant later with P.E.K.K.A low if affordable, else tower + Ghost (3:43, 4:16; numbers
   **[inferred]**).
2. IF the bot spends > 5 in one lane within 4 s AND its estimated elixir ≤ 3 → defend that lane cheaply (P.E.K.K.A
   low on a Giant or Mini P.E.K.K.A; Ghost/Bandit/E-Wiz on the rest), then the moment the bot's units are dead put
   Battle Ram at that lane's bridge in front of the survivors; Bandit in the other lane if elixir ≥ 3 (12:21, 12:52).
3. IF bot elixir ≤ 2 AND no bot troop on the opponent's side AND opp elixir ≥ 4 → Battle Ram at the bridge of the
   weaker bot tower; + Bandit if opp ≥ 7 (2:07, 19:15).
4. IF Mini P.E.K.K.A is out of cycle AND (Mega Minion out of cycle OR on the other side of the arena) AND opp elixir ≥
   7 → P.E.K.K.A at the bridge (x, 18.5) in the lane without the bot's Hut; Ram or Bandit in front of it at the next 4 elixir
   (8:39, 15:32).
5. IF the bot just spent ≥ 6 to stop a P.E.K.K.A push (e.g. Mini P.E.K.K.A + Mega Minion) → Battle Ram + Bandit at
   the opposite bridge as soon as opp elixir ≥ 7 (6:28). **[inferred]** threshold.
6. IF the bot's Fireball is out of cycle AND a P.E.K.K.A push is on the field → add E-Wiz and Magic Archer behind it
   at the same time (15:32); otherwise keep them ≥ 2.5 tiles apart.
7. IF a bot Giant or Battle Ram is walking to the bridge and opp has E-Wiz → E-Wiz high at (that bridge x, 20.5) to block and
   reset (17:09).
8. IF a bot Mini P.E.K.K.A or Mega Minion targets the P.E.K.K.A → Zap/E-Wiz to reset, Ghost or Magic Archer to kill;
   don't let it trade 1-for-1 (3:43).

DEFAULT: Single elixir: at ≥ 9 elixir, P.E.K.K.A at the back in the lane the bot last played its tank away from,
E-Wiz or Magic Archer behind when it reaches the bridge (11:50). Otherwise defend low and cheaply, cycle Ghost/Bandit at
the back to reach P.E.K.K.A or E-Wiz without showing it (13:56, 14:28). Double elixir: after every bot commitment
≥ 5, pressure the other lane with Ram + Bandit/Ghost; don't stack more than two cards behind one push (11:19).
Overtime, ahead: defend and bridge-block only, punish only through rule 3 (16:37). Overtime, behind or level: rules 3
and 5 with Fireball to finish a tower under 700. **[inferred]**

## Scenarios
- **Giant in the back** · 2:40 left · bot 10 / opp 8 · empty board · opp hand Battle Ram, Bandit, P.E.K.K.A,
  Ghost, next E-Wiz · learn: a Giant at the back against bridge spam needs elixir held back for the opposite lane, or
  goes in at the bridge instead · punish: bot Giant + Mega Minion at (3.0, 6) → Ram (14.5, 18.5) + Bandit (14.5, 20.5)
  (rule 1), the bot has 2 elixir and takes ~a tower's third. **[inferred]**
- **Mini P.E.K.K.A on a Bandit** · 2:10 left · bot 6 / opp 7 · opp Bandit crossing at (2.5, 14.5) · opp hand
  P.E.K.K.A, Battle Ram, Zap, Magic Archer, next Ghost · learn: answer cheap troops with cheap cards (Bomber + tower,
  Mega Minion) and keep Mini P.E.K.K.A for the P.E.K.K.A · punish: Mini P.E.K.K.A on the Bandit → P.E.K.K.A at
  (14.5, 18.5) then Ram in front (rule 4).
- **P.E.K.K.A at the back** · 2:30 left · bot 7 / opp 3 · opp P.E.K.K.A at (14.0, 32), walking · opp hand
  E-Wiz, Bandit, Zap, Fireball, next Battle Ram · learn: answer a 7-elixir back play with pressure in the other lane
  (Battle Ram or Giant at (2.5, 14.5)), then defend the P.E.K.K.A with Mini P.E.K.K.A once its E-Wiz is spent or
  zapped · punish: bot Mini P.E.K.K.A in front of P.E.K.K.A + E-Wiz → E-Wiz stun resets it, the push takes the tower
  (rule 8). **[inferred]**
- **Stacked support** · 1:20 left (double) · bot 6 / opp 2 · opp P.E.K.K.A at (14.5, 14.5), E-Wiz (14.0, 18.5)
  and Magic Archer (15.0, 19.5) behind it · opp hand Battle Ram, Bandit, Ghost, Zap · learn: Fireball both supports
  (8 for 4), then Mini P.E.K.K.A + tower on the lone P.E.K.K.A · punish: Arrows or a Fireball on one support → the
  push stays, and the bot's tower falls. **[inferred]**

## Open questions for Cole
- The P.E.K.K.A drop point on our Giant (low vs. letting it walk to the tower first): which do good players use?
- Rule 1: is Ram + Bandit at the opposite bridge right against a back Giant, or does a good player just Ram alone and
  save for P.E.K.K.A?
- 12:21 "go for a Bats" is a mis-caption (probably Bandit or Battle Ram). Does it matter for the opening?
- Should rule 4 also wait for our Goblin Hut to be out of cycle, or is the Hut too weak to stop a bridge P.E.K.K.A?
