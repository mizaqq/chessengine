# Opponent pool, long run (POOL_LONG), 2026-09-17 23:26 to 2026-09-18 03:22

800 updates from M34 (seed 42), 16 boards (4 puzzle / 2 finishing / 5 mid-game / 5
opening), pool = SL prior + last 4 snapshots every 50 updates on 5 game boards,
guidance 0.25 on curriculum boards. 3.9 h training (17.4 s per update, 16 evals of
about 1 min), then games, finishes, probe, win matrix. Named exception to the
120-update budget (owner: evaluation in the morning).

Predictions: owner none given. Claude: prior_score 0.55-0.60, m1 within a point,
own-game unchanged, conversion unchanged. Actual: prior_score 0.41 (wrong), m1 +4
points (wrong, better), own-game +8 points (wrong, better), conversion unchanged (right).

## Dials

| update | m1 | m2 | m3 | m4 | own-game m1 | prior_score (W-L of 40) |
|---|---|---|---|---|---|---|
| 0 (M34) | 0.741 | 0.634 | 0.546 | 0.442 | 0.22 | 0.49 (12-11 of 60, earlier session) |
| 100 | 0.761 | 0.649 | 0.558 | 0.483 | 0.25 | 0.49 (8-9) |
| 200 | 0.746 | 0.658 | 0.576 | 0.492 | 0.21 | 0.39 (3-12) |
| 400 | 0.753 | 0.668 | 0.611 | 0.511 | 0.30 | 0.44 (4-9) |
| 600 | 0.779 | 0.684 | 0.632 | 0.536 | 0.29 | 0.36 (0-11) |
| 800 | **0.785** | **0.690** | **0.639** | **0.541** | **0.30** | 0.41 (4-11) |

Finishing d2 / d10 / d20 at 800: 0.35 / 0.16 / 0.13 (M34 0.32 / 0.16 / 0.12).
40 sampled games: 22 decisive (M34 15), mate found 22 of 85 (0.26), mean 149 plies.
Probe: Lichess m1 top-1 0.807, confident-wrong 0.13 (M34 0.16); own-game m1 top-1
0.30, <0.05 share 0.55 (M34 0.61); human-mate d0 top-1 0.64 (0.55).

Pool during training (per 100-update window, about 35 pool games): pool_score_prior
0.50, 0.35, 0.50, 0.50, 0.36, 0.38, 0.50, 0.50; against snapshots mostly 0.4-0.6.
The learner never learned to beat the prior on the pool boards either.

## Win matrix (30 games per colour, decisive W-L, rest drawn or capped)

|  | vs SL | vs CLEAN_GUIDE | vs M34 | mean score |
|---|---|---|---|---|
| SL | - | 20-4 | 18-11 | 0.583 |
| CLEAN_GUIDE | 4-20 | - | 7-21 | 0.381 |
| M34 | 11-18 | 21-7 | - | 0.481 |
| POOL_LONG | 5-12 | 21-8 | **18-4** | 0.556 |

## Reading

- **Success criterion missed.** prior_score 0.41 at the end, target 0.50. Against the
  prior POOL_LONG loses 5-12 with 43 of 60 games drawn or capped; M34 lost 11-18.
  The gap in score is the same (0.44 both), the games are quieter.
- **Drift stopped, strength rose within the RL family.** POOL_LONG beats M34 18-4 and
  CLEAN_GUIDE 21-8. Before this run every RL-vs-RL pairing was even; this is the
  first RL checkpoint that clearly beats its own ancestors. The pool did what Bansal
  says it does ("the policy at any time should be able to defeat random older
  versions of itself") for the snapshots; it did not pull the policy back to the prior.
- **Puzzle dials: best of the project**, still rising at 800 with the learning rate
  at its floor (3e-5 from update 200). Own-game mate-in-one 0.22 -> 0.30, decisive
  games 15 -> 22 of 40. Conversion at depth 10-20 unchanged for the ninth arm.
- **Why the prior held.** The prior is one member of five, on five of sixteen boards:
  about one board plays it at any time, about 0.7 games per update, 560 games in the
  run against 5,000+ against snapshots and self. The gradient it contributes is a
  sliver, and pool_score_prior shows no learning against it. Also the prior's
  strength is in openings and middlegames the learner rarely revisits: the drawn
  games are long and the learner's positional play is the only thing being graded.

## Options next (owner decides)

1. Weight the prior: draw it with probability 0.5 instead of 1/5 (or put it on
   dedicated boards). Cheapest test of the "sliver" reading; 120 updates from
   POOL_LONG, prior_score is the dial.
2. Restore the learning rate floor to 1e-4 for the long run (the run trained 600
   updates at 3e-5).
3. Accept par with the prior as not the goal: POOL_LONG is the strongest checkpoint
   on every dial except the prior head-to-head, and self-imitation (next change)
   targets conversion, which is the owner's original complaint.

Files: `POOL_LONG/run.json, games.json, finishes.json, probe.txt`, `run_long.sh`,
`run_long.log`, `SMOKE/run.json`; matrix in `experiments/win-matrix/matrix_pool_long.json`.
