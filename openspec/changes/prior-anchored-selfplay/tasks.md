## 1. Prior-KL penalty
- [x] 1.1 `update_model(..., prior_model=None, prior_kl_coef=0.0)`: exact masked KL per minibatch, added to the loss when coef > 0, `prior_kl` in the returned stats (null without a prior). Tests in `tests/model/test_ppo_update.py`: the 3-move scenario's KL value (0.6264), step moves probability toward the prior, coef 0 leaves the loss unchanged but reports the KL.
- [x] 1.2 Config `prior_kl_coef` (default 0) threaded through `run_chess_training` -> `update_kwargs`; `resolve_prior_kl` loads the frozen prior from `opponent_prior` when coef > 0 (error if missing); `mean_prior_kl` in the metrics summary; validation test.

## 2. Prioritised opponent sampling
- [x] 2.1 `OpponentPool(sampling="uniform"|"pfsp", power=2, min_weight=0.05)`: EMA win rates per member, weighted `sample`, `record(name, score)`; tests for the three spec scenarios (uniform, prioritised draw probabilities over many draws, EMA update).
- [x] 2.2 Rollout calls `pool.pool.record(name, score)` where `add_pool_result` is called; config keys `opponent_pool_sampling`, `pfsp_power`, `pfsp_min_weight` with validation; per-member weights in the training log (`pool_weight_<name>`).

## 3. Experiment
- [ ] 3.1 Driver `experiments/anchor/run_anchor_arms.sh`: KLPRIOR (prior_kl_coef 0.1) and PFSP (pfsp, opponent_pool_size 12) from slhi2_256, 600 updates, guards as LONG256X; games, finishes, audit, ladder, 200-game matrix vs SLHI2_256 / LONG256X (and DECAY / NOPOOL if finished). Record `experiments/anchor/outcome.md` with the owner's prediction.

## 4. Comprehension
- [x] 4.1 Comprehension check with owner (CLAUDE.md "Trace by hand"): given a 3-move position with the policy and prior probabilities from the spec scenario, compute the prior-KL term and say which move's probability the step raises; then given three pool win rates, compute the PFSP draw probabilities. Check both against the tests.
