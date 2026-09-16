## Why

The scale-0.2 checkpoints find a mate-in-one in 8 of 215 opportunities (3.7%) and
end most games in draws. From the opening a mate is hundreds of random moves away,
so the outcome reward is almost never observed. With sparse rewards the exploration
needed grows exponentially with the number of steps between rewards (Salimans &
Chen 2018); starting episodes one move from the reward removes that barrier. The
owner chose this on 2026-09-16 ahead of PPO/GAE so the curriculum can be judged on
the unchanged A2C loop and later composed with PPO.

## What Changes

- Environments can start a game from a given FEN instead of the opening.
- A start-position sampler: with probability `puzzle_fraction` a reset starts from a
  random training puzzle (a position where the side to move has a mate in one),
  otherwise from the opening. Fixed fraction, chosen 0.5.
- A puzzle data step that turns the Lichess puzzle dump (CC0) into two small CSVs:
  a training set of 20,000 mate-in-one positions and a held-out evaluation set of
  2,000, positions stored after the puzzle's first move so the mover has the mate.
- A puzzle evaluation: top-1 accuracy and mean probability mass on mating moves over
  the held-out set, run every `eval_interval` updates and at the end of training.
- No change to returns, shaping or loss.

## Capabilities

### New Capabilities
- `start-positions`: starting games from supplied positions and mixing puzzle starts
  with opening starts during training.
- `puzzle-evaluation`: measuring a policy on a fixed held-out set of mate-in-one
  positions.

### Modified Capabilities
- none. `reward-shaping` requirements hold unchanged: a puzzle start has a non-zero
  start potential, which shifts every value by a constant and leaves the optimal
  policy unchanged (Ng, Harada & Russell 1999, Theorem 1 holds for any start state).

## Alternatives considered

1. **Reverse curriculum by start states (chosen).** Florensa et al. 2017, "Reverse
   Curriculum Generation for RL" (arXiv 1707.05300): start near the goal, widen the
   start distribution as the agent succeeds; needs only a goal state, no demos or
   shaped reward. Salimans & Chen 2018 (arXiv 1812.03381): resetting to states along
   a demonstration turns exponential exploration into roughly quadratic. Chosen
   because it leaves the optimizer untouched and composes with PPO.
2. **Supervised pre-training on puzzle solutions.** AlphaGo (Silver et al. 2016,
   Nature 529) trained the policy on expert moves before RL. Deferred as the
   comparison arm (learning record 0005): a different signal, so it should be
   measured against the RL curriculum, not replace it.
3. **Benchmark only, no training change.** Needed anyway as the fixed evaluation
   set; alone it does not address the weakness.

## How we will know it worked

Metrics, same budget as the shaping runs (500 updates, 12 envs x 15 steps, seed 0,
shaping scale 0.2):
- held-out puzzle top-1 accuracy and mean mating-move probability, before and after;
- draw rate and decisive-game rate in normal-opening games (`puzzle_fraction` 0 at
  evaluation time);
- mate-in-one rate in sampled games (transfer from puzzles to real play).

Owner's prediction (2026-09-16): fewer draws in normal games. Claude's numeric
guess, to compare against: old checkpoints top-1 about 5%, mating-move probability
about 0.08; after 500 curriculum updates top-1 about 30%, probability about 0.35,
and a smaller improvement in in-game mate-in-one rate than on the puzzle set
(distribution shift).

## What to understand

- Why sparse reward makes exploration cost grow with distance to the reward, and
  how start states change that.
- What `puzzle_fraction` trades off and what happens at 0 and at 1.
- Why a non-zero start potential does not break potential-based shaping.
- Train/test distribution shift: puzzle positions versus self-play positions, and
  why puzzle accuracy should move before the in-game rate.
- Why a fixed held-out set is a better benchmark than sampled games.

## Impact

- `src/envs/open_spiel_env.py`, `open_spiel_vector_env.py`,
  `open_spiel_async_vector_env.py`: reset from FEN, sampler hook.
- New `src/envs/start_positions.py`, `src/eval/puzzles.py`,
  `scripts/prepare_puzzles.py`, `data/puzzles/*.csv`.
- `src/model/training.py`: periodic evaluation hook. `src/entrypoints/train.py`,
  `src/configs/train_default.yaml`: new keys `puzzle_fraction`, `puzzle_train_file`,
  `puzzle_eval_file`, `eval_interval`.
- New dev dependency for reading the `.zst` dump.
