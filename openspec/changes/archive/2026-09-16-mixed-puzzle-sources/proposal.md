## Why

The shared network finds 43% of held-out Lichess mates but 5% of the mates that
appear in its own games (2026-09-16). Owner's diagnosis: puzzle positions are
curated human-game positions, unlike the positions self-play reaches. Florensa et
al. 2017 generate start states by stepping backwards from the goal, and Salimans
& Chen 2018 reset to states of the agent's own trajectory: in both, the start
distribution is tied to the agent's reachable states. Ours is tied to Lichess.
Owner's counter-concern: harvested positions teach finishing badly played games,
not efficient play. Agreed: puzzles only teach finishing; finishing is what turns
300-ply dead games into outcome rewards that can teach the rest. Decision: mix
both sources, keep Lichess as the aspirational distribution (enlarged to 50k as
the owner proposed), and use 4 puzzle boards.

## What Changes

- Lichess training set enlarged to 50,000 positions; the existing 2,000 held-out
  Lichess set is kept byte-identical (ids excluded from the new train set) so the
  0.352 / 0.427 numbers stay comparable.
- A harvester plays sampled self-play games with a checkpoint on a vectorized env
  and stores every position where the mover has a mate in one, deduplicated by
  FEN, split into a self-play training set and a self-play held-out set. Refresh
  is per run in this version (harvest from the latest checkpoint before each run).
- Puzzle boards draw from several puzzle files with configured weights (here
  Lichess 0.5, self-play 0.5).
- An explicit `puzzle_boards` count (here 4) alongside `puzzle_fraction`.
- Held-out evaluation over several named sets: `lichess` and `selfplay`, keys
  prefixed in the logs.

## Capabilities

### New Capabilities
- none.

### Modified Capabilities
- `start-positions`: puzzle boards can draw from several weighted puzzle files;
  puzzle board count can be given explicitly.
- `puzzle-evaluation`: several named held-out sets, each reported separately.

## Alternatives considered

1. **Lichess only, 50k, 4 boards (owner's first proposal).** Training solve rate
   and held-out accuracy already coincide (0.417 vs 0.427), so the set size is
   not the limit and both measure the same distribution; rejected alone, kept as
   half of the mix.
2. **Self-play harvested only.** Matches Florensa/Salimans-Chen most closely but
   drops the target distribution the owner wants the models to move toward, and
   risks the "learn to finish bad games" failure the owner named. Rejected alone.
3. **Mix (chosen).** Both sources, equal weight, refreshed harvest per run.
4. **In-loop harvest every N updates.** Cleaner tracking of the agent's states;
   deferred until the per-run version shows an effect.

## How we will know it worked

Against `experiments/shared-network/seed0` (held-out Lichess 0.427, in-game
mate rate 5%, 27/40 games at the ply cap), one seed, 500 updates, 4 puzzle boards:
- held-out Lichess top-1 (should not fall much despite half the puzzle plies);
- held-out self-play top-1 (new; baseline measured on the current checkpoint first);
- in-game mate-in-one rate and share of games reaching the ply cap.

Owner's prediction: <to state in conversation before apply>. Claude's guess:
Lichess held-out 0.38-0.42, self-play held-out from a baseline near 0.05 to above
0.30, in-game mate rate 10-15%, fewer games at the cap.

## What to understand

- Why train/held-out agreement rules out "too few puzzles" as the cause.
- Start-state distribution vs the agent's reachable states (Florensa, Salimans &
  Chen) and why a frozen harvest would be the mistake the owner described.
- Finishing as the mechanism that produces outcome rewards for the rest of the game.
- Reading three accuracies at once: Lichess, self-play, in-game.

## Impact

- `scripts/prepare_puzzles.py` (--train 50000, keep held-out ids), new
  `scripts/harvest_positions.py`, `src/envs/start_positions.py` (weighted mixed
  sampler), `src/entrypoints/train.py` (`puzzle_train_files`, `puzzle_boards`,
  `puzzle_eval_files`), `src/model/training.py` unchanged, eval hook returns
  prefixed keys, README, tests.
