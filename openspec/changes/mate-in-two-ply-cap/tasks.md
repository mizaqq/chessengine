## 1. Data
- [x] 1.1 `Puzzle(mate_in, key_moves)`; loader accepts `mating_moves` or `key_moves`; tests.
- [x] 1.2 `prepare_puzzles.py --depth 2` with forced-mate verification; tests; build the files.

## 2. Environment
- [x] 2.1 `OpenSpielEnv.reset(fen, puzzle_moves, key_moves, max_plies)` per D1; tests
      (mate-in-two solved, wrong first move ends as miss, right first move then miss,
      cap ends game as draw, cap not applied to puzzle boards).
- [x] 2.2 Samplers return `Puzzle`; sync/async vector envs pass it and `max_plies`;
      `info["puzzle_depth"]`; tests.

## 3. Metrics, eval, config
- [x] 3.1 `puzzle_solved_rate_m<depth>`; training loop passes depth; tests.
- [x] 3.2 `evaluate_puzzles` with key-move targets for depth 2; tests.
- [x] 3.3 `max_plies` config + validation; default yaml sources and eval sets; README.

## 4. Experiment
- [ ] 4.1 Arms base, M2, CAP, BOTH (proposal table) to `experiments/mate-in-two-ply-cap/`;
      40-game evaluation each; summary.
- [x] 4.2 Outcome in the proposal; notes; session.

## 5. Comprehension check (moved into `workshop_rl_chess.ipynb`, sections 10-12; owner completes it)
- [ ] 5.1 *Explain back*: why the wrong first move ends the puzzle episode at once,
      and what the network loses by that.
- [ ] 5.2 *Diagnose from symptoms*: read `puzzle_solved_rate_m2` vs `lichess_m2_top1`.
