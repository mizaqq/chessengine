## Context

See proposal.md - Why. Verified 2026-09-16: OpenSpiel's chess game has no `fen`
parameter, but `game.new_initial_state(fen)` builds a state from a FEN and
`rl_environment.Environment.set_state(state)` installs it; legal moves, returns
and terminal detection then work as normal. The sync env resets each
`OpenSpielEnv` directly; the async env resets inside forked worker processes, so
whatever picks the start position must be available in the worker.

## Goals / Non-Goals

**Goals:** start positions from puzzles with a fixed mixing fraction; a fixed
held-out benchmark; no change to returns, shaping or loss; deterministic under a
seed.

**Non-Goals:** adaptive scheduling of the fraction (Florensa et al. adapt to
success rate; later step if the fixed knob works); mate-in-two or longer puzzles;
supervised pre-training; Stockfish evaluation.

## Decisions

**D1. Sampler object shared by both envs.** `StartPositionSampler(puzzles,
fraction, seed)` with `sample() -> Optional[str]`. The sync vector env owns one and
passes the FEN to `OpenSpielEnv.reset(fen)`; the async env gives each worker its own
sampler at construction (seed + env index) so the worker can auto-reset without a
round trip. Alternative: have the parent send a FEN with every step message; rejected
because auto-reset happens inside the worker before the parent knows the game ended.

**D2. Positions stored after the puzzle's first move.** Lichess FENs are the
position before the opponent's move; the solution starts at move two. The prep
script applies move one with python-chess and stores the result, so the training
loop never needs to know Lichess's convention.

**D3. Mating moves recomputed at evaluation.** OpenSpiel actions are not UCI; mapping
would need a per-move string match. Instead evaluation walks the legal actions,
builds each child and checks terminal with a win for the mover. The stored UCI moves
are used only by the prep script to validate rows.

**D4. Fixed `puzzle_fraction`, default 0.0.** What it controls: the share of
episodes that are short and reward-dense versus long and sparse. At 0 today's
training. At 1 the models never see an opening and only learn mating pictures and
defending lost positions; expect in-game play to degrade. Chosen 0.5 by the owner.
Default 0 keeps existing configs and the shaping baseline reproducible.

**D5. `eval_interval`, default 50.** How often the held-out evaluation runs. Very
small values slow training (2,000 forward passes per evaluation); very large ones
lose the learning curve. 50 updates gives 10 points over a 500-update run.

**D6. Evaluation is a callable injected into `run_chess_training`.** Keeps the
training loop free of file paths and lets tests pass a stub.

**D7. Data prep as a script, outputs committed.** The dump is about 250 MB
compressed and CC0. The script downloads it to a gitignored `data/raw/`, filters
`mateIn1`, shuffles with seed 0, writes 20,000 train and 2,000 held-out rows. The
two CSVs (a few MB) are committed so runs are reproducible without the download.
Reading `.zst` needs the `zstandard` package (dev dependency).

**D8 (revision after run 1, 2026-09-16). One-move puzzle episodes, boards not
resets.** Run 1 (`frac05/run.json`) showed flat puzzle accuracy: a missed mate
turned into a 200-ply game, so the run made about 200 attempts in total. Resets
near the reward only help when the episode ends soon after the attempt (the
implicit assumption in Salimans & Chen 2018). Fix: a puzzle board ends its episode
after the mover's move (mate: normal result; else `puzzle_miss`), and puzzle
boards are a fixed subset (`round(puzzle_fraction * num_envs)`) so the share of
plies on puzzles equals the fraction. A miss is a terminal with
`puzzle_miss_reward` (default 0): a penalty would widen the mate/miss gap the
gradient already sees, but would teach the value head that a still-winning
position is a loss; the knob exists so the owner can test that as an experiment.
Alternative "replay the puzzle until solved" deferred: with-replacement sampling
already revisits puzzles, back-to-back replay correlates a batch, and Florensa's
success-band sampling is the principled version if the curve stalls. Puzzle
episodes are counted separately in metrics so opening-game rates stay readable.

## Risks / Trade-offs

- [Puzzle positions are tactical and unbalanced; in-game mate-in-one rate may not
  follow puzzle accuracy] → report both; the gap is itself the lesson on
  distribution shift.
- [At fraction 0.5 the defender model trains a lot on lost positions] → watch
  value-head behaviour on puzzle starts; it should read them as lost.
- [Puzzle episodes are one move long when solved, so many more games end per
  update and the draw penalty becomes rarer relative to wins] → draw/decisive rates
  in the logs mix puzzle and opening games; the final comparison uses
  `puzzle_fraction` 0 in a separate evaluation run.
- [Terminal reward enters as gamma * reward at the mating own step (existing
  quirk)] → unchanged; noted in learning notes.

## Open Questions

- Whether to widen to `mateIn2` before or after PPO. Does not affect this change.
