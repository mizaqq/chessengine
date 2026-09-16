## 1. Advantage estimation (GAE)

- [x] 1.1 Failing tests in `tests/model/test_gae.py`: lambda 1 equals returns
      (with shaping), lambda 0 residuals, lambda 0.95 blend, alternating players,
      game ending on the opponent's move, new game after done.
- [x] 1.2 Implement `compute_gae_for_model` in `src/model/returns.py` (design D2).

## 2. Rollout storage

- [x] 2.1 New `StepRecord` fields in `src/core/types.py`; update `_make_step` in
      `tests/model/test_returns.py`.
- [x] 2.2 `_collect_rollout` under `torch.no_grad()` filling the new fields; test
      that no stored tensor requires grad and no parameter has `.grad` afterwards.

## 3. Update step

- [x] 3.1 Failing tests in `tests/model/test_ppo_update.py`: ratio one on first
      minibatch (eval-mode model), positive-advantage clip gives no policy gradient,
      negative-advantage clip, single-sample normalisation gives 0 and no NaN,
      epochs x minibatches optimizer steps, value_coef doubling doubles value-head
      gradient, a2c preset equals direct REINFORCE-with-baseline gradient, empty
      batch means no step.
- [x] 3.2 Implement `update_model` in `src/model/training.py`; `ComposedLoss` gains
      `value_coef`.

## 4. Configuration and dispatch

- [x] 4.1 `resolve_algorithm(config)` with presets and validation in
      `src/entrypoints/train.py`; tests in `tests/entrypoints/test_config.py`.
- [x] 4.2 Wire GAE + `update_model` into `run_chess_training` (shared and legacy);
      remove the graph-based path; `tests/integration` and `tests/model` pass.
- [x] 4.3 `clip_fraction` / `approx_kl` in `MetricsAggregator` and logs (null under
      a2c); tests.
- [x] 4.4 `src/configs/train_ppo.yaml`; smoke test with `algorithm: ppo`.

## 5. Verification and docs

- [x] 5.1 Full suite green; 3-update runs of both configs finish, ppo log carries
      the diagnostics.
- [x] 5.2 README: rollout format, algorithm switch, PPO keys, BN deviation.
- [x] 5.3 Local knowledge: training-loop.md re-evaluation design; gotchas
      "separate computation graphs" marked historical.

## 6. Experiment

- [x] 6.1 Run arms A, B, C, C2 (proposal table; C2 = PPO at lr 1e-4, added after the clip-fraction finding) to `experiments/ppo-gae/<arm>/` with
      `scripts/run_experiment.py`; `eval_games.py` on each final checkpoint.
- [x] 6.2 Write the outcome into the proposal ("Outcome" paragraph); compare with
      the baseline numbers. (Learning record waits for the comprehension check.)

## 7. Comprehension check with owner (parked by the owner on 2026-09-16; archived open, carried to the next session's spaced review)

- [ ] 7.1 Plain-language lambda question (scene first), then *trace by hand* the
      lambda 0.95 scenario.
- [ ] 7.2 *Map paper to code*: PPO eq. 7 and eq. 9 terms in `update_model`, and the
      deviations (normalised entropy, per-minibatch normalisation, BN drift).
- [ ] 7.3 *Diagnose from symptoms*: read clip fraction / approx KL curves of arm C.
- [ ] 7.4 Record demonstrated understanding in `learning/records/`.
