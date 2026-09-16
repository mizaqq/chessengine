## 1. Data

- [x] 1.1 `prepare_puzzles.py --train 50000 --keep-eval`: exclude held-out ids, rewrite train only; verify eval file unchanged (hash) and train has 50,000 rows disjoint from eval
- [ ] 1.2 `scripts/harvest_positions.py`: vectorized sampled self-play from a checkpoint dir, collect mate-in-one positions, dedup, split; unit tests on the collection/dedup/split functions with a stub; run on `experiments/shared-network/seed0` for 3,000 train / 300 held-out and verify every row mates

## 2. Mixing and configuration

- [x] 2.1 `MixedSampler` with weights; tests for equal weights, zero weight, determinism
- [x] 2.2 Config: `puzzle_train_files` (list of file/weight), `puzzle_boards` override with range check; tests for precedence and errors
- [x] 2.3 Multi-set evaluation: `puzzle_eval_files` mapping, prefixed keys, single-file path unchanged; tests

## 3. Measure

- [ ] 3.1 Baseline: `shared-network/seed0` on the self-play held-out set; record `experiments/mixed-puzzle-sources/baseline.json`
- [ ] 3.2 Run 500 updates, seed 0, 4 puzzle boards, Lichess 0.5 / self-play 0.5 -> `experiments/mixed-puzzle-sources/seed0/`; then 40 games; compare with predictions

## 4. Record

- [ ] 4.1 README, default yaml, notes, knowledge; RESOURCES note on start-state distribution
- [ ] 4.2 Comprehension check with owner (CLAUDE.md "Diagnose from symptoms" on the three accuracies; "Explain back" on why train/held-out agreement rules out set size), code and numbers inline; outcome to `learning/records/`
