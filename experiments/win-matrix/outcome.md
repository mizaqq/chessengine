# Checkpoint win matrix (step 0 of the model-free plan), 2026-09-17

Question: does a newer checkpoint beat older ones head to head? (Bansal et al. 2017,
Table 1: "the policy at any time should be able to defeat random older versions of
itself".) Six checkpoints in training order, 30 games per colour per pairing, both
sides sampling, draws and the 300-move cap count 0.5. Ran 9.5 min single-process.

Score of the row player (1 = row wins every game):

|             |   SL | PPO_SL | FIN3 | CLEAN | GUIDE | CLEAN_GUIDE | mean |
|-------------|-----:|-------:|-----:|------:|------:|------------:|-----:|
| SL          |    - |   0.59 | 0.67 |  0.62 |  0.66 |        0.62 | 0.632 |
| PPO_SL      | 0.41 |      - | 0.56 |  0.46 |  0.53 |        0.50 | 0.492 |
| FIN3        | 0.33 |   0.44 |    - |  0.53 |  0.49 |        0.51 | 0.460 |
| CLEAN       | 0.38 |   0.54 | 0.47 |     - |  0.49 |        0.50 | 0.476 |
| GUIDE       | 0.34 |   0.47 | 0.51 |  0.51 |     - |        0.51 | 0.468 |
| CLEAN_GUIDE | 0.38 |   0.50 | 0.49 |  0.50 |  0.49 |           - | 0.472 |

Decisive games only (wins row-col, out of 60 per pairing; the rest are draws or capped):

SL beats PPO_SL 15-4, FIN3 20-0, CLEAN 16-2, GUIDE 19-0, CLEAN_GUIDE 18-3.
All RL-vs-RL pairings are within noise of even (largest gap PPO_SL vs CLEAN 5-10).

## Reading

- The supervised prior is the strongest whole-game player we have: 88 wins to 9 over
  the five RL checkpoints combined. Every RL phase, shaped or not, guided or not,
  made the network worse at beating the prior while it got better at every puzzle
  and finishing dial we tracked. This is the whole-game face of the fallen
  human-move-agreement number: the policy drifted away from human openings and
  middlegames toward whatever draws against itself.
- RL checkpoints do not order among themselves: no cycling between them, just a
  plateau at roughly equal strength with 60-75% draws.
- So the self-play weakness is not "newer loses to slightly older" (Bansal's
  imbalance) but "everything after RL loses to the start". Same remedy family:
  the opponent pool must contain the SL prior, so the learner is punished for the
  drift instead of rewarded for drawing against its own copy.

## Consequence for the plan

Opponent pool (step 2) moves ahead of self-imitation (step 1): put SL and the
saved checkpoints in the opponent pool (Bansal: uniform over history was best on one
body, latest-only worst on both) and re-run this matrix as the success test. The
target is a checkpoint whose row beats SL, while the puzzle dials hold.
Self-imitation still addresses conversion and comes next.

Files: `matrix.json` (table and raw counts), `run.sh`, `run.log`; script
`scripts/eval_matrix.py` (`--workers 1` inside the sandbox; the process pool is
blocked there).
