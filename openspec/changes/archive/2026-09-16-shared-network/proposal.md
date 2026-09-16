## Why

Two independently initialised networks, one per colour, each see half the data
and share nothing: a back-rank mate learned by white teaches black nothing, and a
5x colour asymmetry appeared in evaluation games after the mate-in-one curriculum
(learning record 0008). AlphaZero (Silver et al. 2017, methods p.13) uses one
network whose input is "oriented to the perspective of the current player" with
planes for "the player's pieces" and "the opponent's pieces" plus a colour plane,
so every pattern is learned once from both colours' data. Owner chose this on
2026-09-16 as the next change, ahead of PPO/GAE.

## What Changes

- One `ChessPolicyProbs` instance and one optimizer serve both colours.
- Observations are oriented to the mover before they reach the network: for
  black to move, ranks are flipped, white/black piece planes are swapped, and the
  castling planes are swapped; the side-to-move plane is kept as the colour plane.
  White-to-move observations are unchanged. Verified 2026-09-16 that OpenSpiel's
  4,674 move indices are already relative to the mover (a2a3 for white and a7a6
  for black share an index, likewise castling and promotions), so actions need no
  mirroring.
- Rollout, bootstrap, returns and loss keep the per-side bookkeeping (white's
  steps and black's steps), but both sides' samples update the same parameters in
  one backward pass per update.
- Checkpoints: one file `model_<ts>_episodes_<N>.pth`; the loading scripts and
  the notebook accept either the single-file or the legacy white/black pair.
- Metrics unchanged except that entropy and loss are no longer split by model.

## Capabilities

### New Capabilities
- `shared-policy-network`: mover-oriented observations and a single network for
  both colours in self-play training and evaluation.

### Modified Capabilities
- none. `reward-shaping`, `start-positions`, `puzzle-evaluation` requirements are
  unchanged; the puzzle evaluation queries "the model of the side to move", which
  becomes the shared model with an oriented observation.

## Alternatives considered

1. **One network, absolute board plus side-to-move plane.** Minimal code change;
   the trunk is shared but colour-specific patterns are still learned twice
   because a white back-rank mate and a black one look different. AlphaZero
   explicitly orients the board instead (Silver et al. 2017, p.13). Rejected as
   a half step.
2. **Oriented board, one network (chosen).** AlphaZero's representation. Full
   pattern sharing; the asymmetry between colours disappears by construction.
   Cost: an observation transform and its tests.
3. **Keep two networks, fix the asymmetry directly.** Only defensible if the
   seed-1 run shows a systematic colour bug rather than a weaker network; the
   result is recorded in the design when known. Two networks remain data-
   inefficient either way.

## How we will know it worked

Same budget as the curriculum run 2 (500 updates, 12 boards, 15 steps, 6 puzzle
boards, seed 0, lr 3e-4). Compare against `frac05-onemove`:
- held-out puzzle top-1 (0.352) and mating-move probability (0.347);
- in-game mate-in-one rate (18%), decisive rate (40%), white/black split of
  decisive games (2/14 and 4/10 across two seeds);
- wall-clock per update (1.25 s; one backward on 180 samples replaces two on 90);
- opening-board entropy.

Owner's prediction (2026-09-16): an upward trend, with learning roughly twice as fast
because every pattern now sees both colours' data; no exact numbers. Sharpened in
conversation to a testable form: run 2 crossed held-out top-1 0.30 at update 200,
so the shared network should cross 0.30 by about update 100 and finish above
0.352; in-game figures expected to trend up across two seeds. Claude's guess:
held-out top-1 0.40-0.45 (each pattern now sees both colours' attempts), decisive
games split within noise of 50/50, wall-clock within 10% of before.

## What to understand

- Why orienting the board to the mover is a symmetry, and what it means to
  "share" a pattern across colours (data augmentation by construction).
- Which observation planes flip, swap or stay, and why the move index does not.
- How the two-model bookkeeping (per-side returns, bootstrap sign) survives with
  one set of parameters.
- Business analogue: weight sharing across symmetric entities (left/right, buyer/
  seller, home/away) doubles effective data and removes spurious asymmetries.

## Impact

- New `src/model/orientation.py` (observation transform), used in
  `_collect_rollout`, `_compute_bootstrap`, `src/eval/puzzles.py`, `src/viz/play.py`.
- `src/model/training.py`: one model/optimizer; `_backpropagate` merges both
  sides. `src/entrypoints/train.py`: single model, `save_models` -> `save_model`.
- Scripts (`eval_checkpoint_puzzles`, `eval_games`, `run_experiment`), notebook
  loader, README, `knowledge/`.
- Tests: transform (hand-traceable), legal-mask invariance under mirroring,
  training smoke, checkpoint loading of both formats.
