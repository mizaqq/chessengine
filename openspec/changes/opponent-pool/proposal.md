## Why

The checkpoint win matrix (experiments/win-matrix, 2026-09-17) showed that every RL
checkpoint loses head to head to the supervised prior it started from: 88 decisive
games to 9 across five checkpoints, while the same checkpoints beat the prior on
every puzzle and finishing dial. Self-play against an identical copy never charges
for drifting away from the openings and middlegames the prior plays well, because
the only opponent has drifted the same way. The mate-in-3/4 rungs (M34, two seeds)
narrowed the gap but did not close it. The owner chose the opponent pool as the
next model-free step and, on seeing that only 4 of 12 boards play whole games, a
layout with more game boards.

## What Changes

- Some game boards play the learner against a frozen opponent drawn from a pool:
  the supervised prior plus the learner's own last four snapshots taken every 50
  updates (owner: "every 50 updates and we play against a pool of last 4"). The
  learner's colour on such a board is drawn at random when the board resets.
- Only the learner's plies on pool boards enter the PPO batch; the opponent's plies
  are recorded for the environment but dropped from the update. Rewards stay the
  plain terminal win / loss / draw from the learner's side.
- 16 boards: 4 puzzle, 2 finishing, 5 mid-game, 5 opening (owner: "go with 16
  boards"). Half of the ten game boards play the pool, half play the current self.
- Periodic evaluation gains a score against the prior (sampled games, both colours),
  so drift is visible during the run; the win matrix runs at the end.
- The first test run is long (owner: "the next run we test should be a long one so
  the evaluation happens in the morning"), a named exception to the standard budget.

## Capabilities

### New Capabilities
- `opponent-pool`: frozen-opponent sampling on game boards, learner-only batches,
  pool snapshots, and the score-against-prior evaluation.

### Modified Capabilities
- none: puzzle, finishing and start-position behaviour is unchanged; the layout is a
  config change within the existing `start-positions` requirements.

## Alternatives considered

- **Latest self only (today).** Bansal et al. 2017 (ICLR 2018, §4.2, Table 1): "training
  against the most recent opponent leads to imbalance"; delta = 1 (latest only) was
  the worst sampling rule on both bodies. Our matrix shows the failure as drift from
  the prior rather than imbalance between colours, but the mechanism is the same:
  nothing in the training signal opposes the drift.
- **Uniform over the whole history** (Bansal delta = 0, best for Ant). Rejected for
  now by the owner in favour of the prior plus the last four snapshots: fewer
  distinct networks per step and the prior is the one member the matrix says matters.
  If the end matrix shows a two-policy cycle (a checkpoint beats its predecessor but
  not the one before), widen the pool.
- **Neural Fictitious Self-Play** (Heinrich & Silver 2016): best-response network plus
  an average-policy network trained on a reservoir buffer. Addresses exploitability
  in imperfect-information games; larger change, and the matrix showed no cycling
  among RL checkpoints, only drift. Queued behind this change.
- **Self-imitation** (Oh et al. 2018): replay own won games with the (R - V)+ loss.
  Targets conversion, not drift; next after this change.

## How we will know it worked

Dials, every 50 updates and at the end:
- `prior_score`: learner's score against the supervised prior over 20 games per
  colour, both sides sampling (win 1, draw 0.5). Success: at or above 0.50 at the end.
  Baselines: CLEAN_GUIDE 0.35-0.38, M34 0.49 (30 games per colour).
- `lichess_top1`, `lichess_m2_top1`, `lichess_m3_top1`, `lichess_m4_top1`: hold
  within 2 points of M34 (0.741 / 0.634 / 0.546 / 0.442).
- `pool_score` by opponent during training: should rise against the prior and hover
  near 0.5 against recent snapshots.
- End: `scripts/eval_matrix.py` with the prior, M34 and the new checkpoint. Opponent
  overfitting (Bansal §5.5.2) would show as gains against pool members only.

Owner's prediction for `prior_score` at the end of the long run: _____ (not yet given;
the owner skipped the prediction in chat).
Claude's prediction: 0.55-0.60 against the prior; m1 within a point; own-game
mate-in-one unchanged; conversion at depth 10-20 unchanged.

## What to understand

- Why self-play against an identical copy cannot see drift, and why a frozen prior
  in the pool turns drift into lost games.
- Why the opponent's plies must leave the PPO batch (they are another policy's
  actions; the ratio pi/pi_old would be against the wrong distribution) and how the
  learner's bootstrap value on an opponent-to-move board stays correct.
- What a snapshot pool is (the same idea as a target network in DQN, and as
  training a fraud model against last year's patterns too) and how pool size and
  snapshot interval trade diversity against staleness.

## Impact

`src/model/training.py` (rollout routes pool-board plies to the opponent; learner
mask in `StepRecord`; gather drops opponent plies; snapshot hook), new
`src/training/opponent_pool.py`, new `src/eval/matches.py` (shared by
`scripts/eval_matrix.py`), `src/training/metrics.py` (pool results),
`src/entrypoints/train.py` (config resolution), `src/configs/train_default.yaml`
(16 boards, pool keys), `src/core/types.py` (`StepRecord.learner`). Tests for
each. Owner-facing: new config keys, new eval keys, a long run.
