# punish-gifts: outcome

## PUNISH (2026-09-19): DEPTH20 + 120 updates, one-ply rules punisher at 0.5 on every board

Owner: "the model we are playing against is not punishing for very stupid moves so model is
not learning to avoid them at all" -> "go ahead with it". `punish_epsilon 0.5`,
`punish_threshold 3`, `punish_boards all`; everything else as DEPTH20. Driver `run_punish.sh`.
Audit and matrix run the bare network (no punisher), so every number below is what the
network learned, not what the punisher did for it.

| dial | DEPTH20 (start) | PUNISH |
|---|---|---|
| free-piece gifts per 100 own plies (vs self) | 9.6 | **6.6** |
| own punish rate on free pieces (vs self) | 0.54 | **0.72** |
| allowed mates per 100 plies (vs self) | 0.87 | 0.83 |
| own punish rate on allowed mates | 0.13 | **0.28** |
| gifts per 100 plies vs SL / SL punishes | 8.3 / 0.49 | 7.4 / 0.61 |
| head to head vs DEPTH20, decisive | - | **14 to 8** for PUNISH |
| prior_score (20 games) | 0.41 | **0.53** (first >= 0.5 since the pool runs) |
| mate-in-1 / 2 / 3 / 4 / endgame (held-out top-1) | 0.804 / 0.723 / 0.683 / 0.580 / 0.751 | 0.772 / 0.698 / 0.656 / 0.561 / 0.733 |
| own-game mate-in-one | 0.352 | **0.257** |
| finishing d2 / d10 / d20 | 0.40 / 0.20 / 0.13 | 0.36 / 0.18 / 0.17 |
| technique Q / R / RR / QR (held-out, level 3) | 0.20 / 0.12 / 0.16 / 0.22 | 0.10 / 0.00 / 0.12 / 0.22 |
| game entropy | 0.288 | 0.303 (up) |
| training draw rate (last update) | 0.70 | 0.48 |
| punish_moves / punish_picks per update | - | ~850 / ~420 (fires 0.5 as designed) |

Reading.
- **The mechanism works.** With the punisher off at evaluation time the network hangs a third
  fewer pieces and takes a hung piece 0.72 of the time instead of 0.54; allowed mates get
  delivered twice as often. It beats its own start 14-8 and is at par with the prior (0.53,
  20 games). Entropy went *up*, not down; training games are far more decisive (draws 0.70 ->
  0.48) because gifts now end games.
- **The cost.** Every puzzle set fell by ~0.03 (consistent across five sets: real), own-game
  mate-in-one fell 0.35 -> 0.26, technique held-out fell. The old shaping lesson in a new
  coat: a hint that pays for captures teaches piece-grabbing over mating.
- **A design flaw that explains the mate numbers.** `punish_actions` returns mates *and* free
  captures as one set and the punisher draws uniformly from it. Whenever a mate and a free
  piece exist together (common in won positions) the punisher takes the piece half the time
  instead of mating, and the mixture log-prob makes that a legitimate sample. The label
  demonstrator already orders its rules (mate first); the punisher must too.
- Caveat: the SL prior "punished" our gifts 0.61 instead of 0.49 without changing. The gifts
  we still make are the ones a one-ply check finds most clearly; some earlier "gifts" were
  poisoned pieces the audit cannot tell apart.

Verdict: hypothesis confirmed (fewer gifts, more punishment, stronger head to head), with a
fixable side effect. Next: mates take priority over captures in the punisher (spec change
within this change), rerun 120 updates from DEPTH20 as PUNISH2, compare on the same table.
