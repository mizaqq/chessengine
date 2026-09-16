## Context

See proposal.md. Shared network is the default; `StartPositionSampler(puzzles,
seed)` feeds puzzle boards; `evaluate_mate_in_one(white, black, puzzles,
oriented)` gives one set's metrics; `play_game` plays one game at a time (too
slow for thousands of games: ~3 mate positions per game, ~250 plies each).

## Goals / Non-Goals

**Goals:** mixed sources with weights, explicit board count, multi-set
evaluation, a fast harvester, comparability with the shared-network run.
**Non-Goals:** in-loop harvest refresh, mate-in-two, changes to the loss.

## Decisions

**D1. `MixedSampler([(puzzles, weight), ...], seed)`** with the same `sample()`
/ `with_seed()` interface as `StartPositionSampler`, so the vector envs are
untouched. Config: `puzzle_train_files: [{file, weight}, ...]`;
`puzzle_train_file` (single) still accepted.

**D2. `puzzle_boards` overrides the fraction.** 4 of 12 is 0.333, which
`round(fraction * num_envs)` cannot express cleanly. What it controls: share of
plies that are puzzle attempts (4/12) vs real-game plies (8/12).

**D3. Harvester on the vectorized env.** `scripts/harvest_positions.py`: N
boards (default 32), batched sampled forward passes with oriented observations
via the shared checkpoint loader, every ply checked with python-chess for a
mating move; dedup by FEN; seed-shuffled; split into train / held-out (default
90/10); stops after `--positions` unique positions. Output format identical to
the Lichess files so both loaders and the evaluation are reused.

**D4. Lichess held-out stays fixed.** `prepare_puzzles.py --train 50000
--keep-eval data/puzzles/mate_in_1_eval.csv` excludes the existing 2,000 ids and
rewrites only the train file. Same seed, so the first 20,000 rows are the old
train set in the same order.

**D5. Evaluation sets.** `puzzle_eval_files: {lichess: path, selfplay: path}`;
the eval callable loops over sets and prefixes keys. Baseline: evaluate
`experiments/shared-network/seed0` on the new self-play held-out set before the run.

**D6. Refresh cadence = one run.** Harvest from the newest checkpoint before a
run; the next run harvests again. Recorded as the owner's concern: a frozen
harvest would teach "finishing bad games" forever; per-run refresh tracks the
models as they improve.

## Risks / Trade-offs

- [Harvested positions are from a weak player and may be trivial or bizarre] ->
  report self-play held-out accuracy separately; inspect a sample in the notebook.
- [Halving Lichess plies lowers Lichess accuracy] -> expected small drop; it is
  the price of on-distribution data, measured not assumed.
- [4 boards changes two things at once with the data mix] -> owner's decision;
  noted as a confound in the result.
