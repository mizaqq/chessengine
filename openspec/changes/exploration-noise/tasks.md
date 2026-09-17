## 1. Sampling

- [x] 1.1 `behaviour_probs(p, legal, epsilon, alpha, generator)` in `src/model/training.py`: mixture over legal moves; tests for eps 0 identity, legal-only support, sums to 1, spec log-prob scenario (`tests/model/test_exploration.py`).
- [x] 1.2 `_collect_rollout` takes `explore` (epsilon, alpha, board mask); noisy boards sample from `b`, store `log b(a)`, keep the network's entropy; count off-prior picks (`pi(a) < 0.05`) into `metrics.add_explore(...)`. Test: with eps 1 and alpha large the stored log-prob is about log(1/num_legal) and differs from log pi(a).

## 2. Wiring

- [x] 2.1 `MetricsAggregator.add_explore(count, offprior)` and `explore_offprior_share` in the summary; test in `tests/training/test_metrics_aggregation.py`.
- [x] 2.2 `resolve_exploration(config)` in `src/entrypoints/train.py` (validation, defaults) and the per-update board mask from `envs.is_puzzle_env` / `is_finish_env` (recomputed each update because of `set_layout`); config keys documented in `train_default.yaml` (off by default). Tests in `tests/entrypoints/test_config.py`.
- [x] 2.3 `scripts/probe_finish_values.py --hist`: mate-mass histogram buckets on Lichess m1 (and own-game) positions.

## 2b. Label-guided variant (owner, 2026-09-17: "okay, agreed on it", after NOISE and PUZNOISE flattened the policy)

- [x] 2b.1 Demonstration lines in the env (`set_demo_line`, `demo_actions`, rules-mate fallback), human line carried on finishing starts, `guided_probs`, rollout branch, `guide_epsilon` / `guide_boards` config, `guide_demo_share` metric. Verify: `tests/model/test_guided.py`, config test.
- [ ] 2b.2 Arm GUIDE: 120 updates from CONTROL with `guide_epsilon` 0.25 on curriculum boards; dials as for NOISE plus conversion at depth 10 and 20. Verify: run records and outcome.md.

## 3. Experiment

- [x] 3.1 `experiments/exploration-noise/run.sh`: arms NOISE (eps 0.25, alpha 0.3) and CONTROL from the FIN3 checkpoint, 120 updates each, FIN3 layout; then games, finishes, probe with histogram. Results and reading in `experiments/exploration-noise/outcome.md`, committed.
- [ ] 3.2 RESOURCES entry for Silver et al. 2017 (AlphaZero preprint, read 2026-09-17); NOTES update of the exploration section with the result.

## 4. Comprehension

- [ ] 4.1 Comprehension check with owner (CLAUDE.md "Trace by hand"): given pi(a) = 0.01, noise(a) = 0.20, eps 0.25, advantage +2 and clip 0.2, work out the stored log-prob, the ratio after the network moves pi(a) to 0.02, and whether the clip cuts the gradient; then read the same numbers off `ppo_policy_loss`.
