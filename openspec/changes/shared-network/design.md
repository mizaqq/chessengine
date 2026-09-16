## Context

See proposal.md - Why. Facts verified against OpenSpiel on 2026-09-16:
- Observation tensor (20 x 8 x 8): planes 0-11 are piece planes in
  (white, black) pairs per type in `kPieceTypes` order; 12 empty squares; 13
  repetitions (scalar); 14 side to play (player id, white = 1); 15 irreversible
  move counter (scalar); 16-19 castling rights white-left, white-right,
  black-left, black-right. Rank index 0 is white's back rank; the board is not
  flipped for black.
- Actions: `MoveToAction` encodes from the moving player's perspective; a7a6 as
  black has the same index as a2a3 as white; castling 4672/4673 and promotions
  coincide. Legal masks of mirrored positions are therefore identical.
- `parse_move_to_action("e2e4")` accepts UCI and is the tool for tests.
- Current loop keeps per-side tensors (`value_white`, `log_prob_white`, ...) in
  `StepRecord`; returns are computed per model id; bootstrap negates the
  opponent's value.

## Goals / Non-Goals

**Goals:** one network, mover-oriented input, per-side bookkeeping preserved,
legacy checkpoints still evaluable, single-seed comparison against
`frac05-onemove`.

**Non-Goals:** AlphaZero's 8-step history, its 73-plane action head, MCTS,
changes to the loss (PPO is the next change), removing `StepRecord`'s per-side
fields (they still describe who moved).

## Decisions

**D1. Orientation is a pure function on the observation tensor.**
`orient(obs, player)`: for `player == BLACK`, `obs.flip(-2)` (ranks), then
permute planes with a fixed index list `[1,0,3,2,5,4,7,6,9,8,11,10,12,13,14,15,
18,19,16,17]`. Files are not mirrored (queen-side stays the a-file for both
colours, so left/right castling planes keep their meaning). The side-to-move
plane is left as the colour plane, matching AlphaZero's constant colour input.
Alternative: also mirror files; rejected because chess is not left-right
symmetric (castling, initial king/queen placement).

**D2. Actions pass through untouched.** Verified colour-relative encoding.
Guarded by a test comparing legal masks of mirrored positions, so an OpenSpiel
change would be caught.

**D3. One model, per-side bookkeeping kept.** `models = {WHITE: m, BLACK: m}`
in rollout and bootstrap; returns still computed per side; `_backpropagate`
concatenates both sides' filtered tensors and takes one optimizer step. This is
the smallest change to a loop that already treats "the model of the side to
move" as a lookup. Alternative: rewrite `StepRecord` to a single mover-view
record; deferred, the PPO change touches the same code.

**D4. Where orientation happens.** In the three places that call the model:
`_collect_rollout`, `_compute_bootstrap`, `evaluate_mate_in_one`, plus
`play_game`. Not in the environment, so the env still reports the absolute board
that `material` and the notebook FEN logic rely on.

**D5. Checkpoints.** `save_model(model, dir, updates)` -> `model_<ts>_episodes_<N>.pth`.
`load_models(dir)` returns `(white_callable, black_callable, oriented: bool)`:
single file -> same model twice with `oriented=True`; legacy pair -> two models,
`oriented=False`. Evaluation and play apply orientation only when `oriented`.

**D6. Learning rate and batch.** One backward on ~180 samples (both sides)
instead of two on ~90. Keep lr 3e-4; Adam normalises per-parameter so batch
doubling is not a reason to retune. Note for the owner: with a shared network
white's and black's gradients can partially cancel on positions where the two
sides want opposite things; that is the intended averaging, not a bug.

## Risks / Trade-offs

- [Orientation bug silently degrades black's play] -> hand-traceable unit tests
  on the transform plus the legal-mask invariance test; black-to-move puzzle
  accuracy reported separately in the first evaluation.
- [Colour asymmetry cause hidden if it was a bug] -> seed-1 run (2026-09-16):
  white 5 / black 3 decisive, so the 6/24 split was the seed (a weaker white
  network), not a colour bug. Nothing to inspect before merging. Seed 1 also
  showed no in-game transfer (mate rate 4.7%, 25/40 games hit the 300-ply cap)
  despite held-out top-1 0.364, so single-seed in-game numbers are noisy; the
  owner chose to run one training seed anyway (time cost); the caveat stays on
  the result.
- [BatchNorm statistics now mix both colours] -> intended; note in learning
  notes that the small-batch BN caveat (record: notes "Where the time goes")
  still applies.
- [Notebook and scripts break on the new checkpoint name] -> shared loader in
  `src/model/checkpoints.py` used by all of them.

## Open Questions

- Whether to add AlphaZero's history planes later. Independent of this change.
