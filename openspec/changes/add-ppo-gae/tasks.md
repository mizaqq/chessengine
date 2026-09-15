## 1. Advantage estimation (GAE)

- [ ] 1.1 Write failing tests in `tests/model/test_gae.py` for the five
      advantage-estimation scenarios (lambda 1 equivalence to returns, lambda 0
      residuals, lambda 0.95 blend, alternating players, game ending on opponent
      move) and the value-target requirement; verify they fail with ImportError.
- [ ] 1.2 Implement `compute_gae_for_model` in `src/model/returns.py` per design D2;
      verify `uv run pytest tests/model/test_gae.py tests/model/test_returns.py` passes.

## 2. Rollout storage

- [ ] 2.1 Change `StepRecord` in `src/core/types.py` to hold `obs`, `legal_mask`,
      `action`, `old_log_prob`, `old_value`, `num_legal` alongside `player`,
      `reward_white`, `done`, terminal rewards; update the `_make_step` helpers in
      existing tests; verify the whole suite still collects (`uv run pytest --co`).
- [ ] 2.2 Rewrite `_collect_rollout` to run under `torch.no_grad()` and fill the new
      fields; add a test asserting no stored tensor requires grad and no parameter
      has a `.grad` after collection; verify it passes.

## 3. Update step

- [ ] 3.1 Write failing tests in `tests/model/test_ppo_update.py` for the PPO loss
      scenarios: ratio one on first minibatch, positive-advantage clip (1.5 ratio,
      no gradient), negative-advantage clip (0.5 ratio), single-sample normalisation
      gives 0 and no NaN, epochs x minibatches optimizer-step count (mock optimizer),
      value_coef doubling doubles value-head gradient.
- [ ] 3.2 Implement `update_model` per design D3 in `src/model/training.py`
      (minibatch iterator, ratio, clipped surrogate, normalisation, value_coef,
      normalised entropy, grad clip, clip_fraction and approx_kl accumulation);
      extend `ComposedLoss.compute` with `value_coef`; verify 3.1 tests pass.
- [ ] 3.3 Write a failing gradient-equivalence test: on a fixed rollout, gradients
      from `update_model` with the a2c preset equal a directly computed
      `-(logp*adv).mean() + (v-target)^2.mean() - coef*ent_norm.mean()` gradient
      within 1e-5; then wire the a2c preset and verify it passes.
- [ ] 3.4 Add a test that a model with zero own steps in the rollout gets no
      optimizer step and reports `step_count` 0; verify it passes.

## 4. Configuration and dispatch

- [ ] 4.1 Add `algorithm`, `gae_lambda`, `clip_epsilon`, `ppo_epochs`,
      `ppo_minibatches`, `value_coef`, `normalize_advantage` parsing with presets and
      validation in `src/entrypoints/train.py`; tests in
      `tests/entrypoints/test_config.py` for default a2c, unknown algorithm error
      listing both names, `gae_lambda` out of range, and preset defaults; verify pass.
- [ ] 4.2 Thread the resolved settings through `run_chess_training` and replace the
      two `_backpropagate_for_model` calls with `compute_gae_for_model` +
      `update_model` per model; verify `tests/integration` passes.
- [ ] 4.3 Add `clip_fraction` and `approx_kl` to `MetricsAggregator` and the log
      summary (null under a2c); test both scenarios in
      `tests/training/test_metrics_aggregation.py`; verify pass.
- [ ] 4.4 Add `src/configs/train_ppo.yaml` (algorithm ppo, steps_per_update 64,
      preset values explicit with one-line comments citing the source table) and a
      smoke test that runs 6 updates with it without NaN; verify pass.

## 5. Verification and docs

- [ ] 5.1 Run the full suite `uv run pytest -q` and confirm green; run
      `uv run python -m src.entrypoints.train --max-updates 5` for both configs and
      confirm both finish and the ppo log entries carry the two diagnostics.
- [ ] 5.2 Update README "Key Components" and "Loss Composition" text for the new
      rollout format, algorithm switch and PPO keys; verify by reading the diff.
- [ ] 5.3 Update `knowledge/repos/chessengine/services/training-loop.md` (local) to
      describe the re-evaluation design and retire the "separate computation graphs"
      note as historical; verify the gotchas file still points to it as history.

## 6. Experiment

- [ ] 6.1 Owner fills the prediction block in proposal.md before any run; verify the
      placeholders are replaced.
- [ ] 6.2 Run arms A, B, C (see proposal) with the same seed for 500 updates each,
      saving logs to `experiments/ppo-gae/<arm>.json`; verify three files exist.
- [ ] 6.3 Plot loss, value loss, normalised entropy, draw rate, clip fraction and
      approx KL per arm into `experiments/ppo-gae/summary.png`; compare with the
      owner's predictions and write the discussion into the proposal under
      "How we will know it worked"; verify the section has an "Outcome" paragraph.

## 7. Comprehension check with owner

- [ ] 7.1 *Review the spec delta* (CLAUDE.md): before `/opsx:apply`, the owner reads
      both spec files and states for each requirement whether it matches what they
      expect; flagged gaps are folded into the specs via `/opsx:update`.
- [ ] 7.2 *Trace by hand*: owner computes the lambda 0.95 scenario from the
      advantage-estimation spec on paper, then we run the test and compare.
- [ ] 7.3 *Map paper to code*: owner points at the lines in `update_model` that
      implement each term of PPO eq. 7 and eq. 9 and names the deviation
      (normalised entropy, per-minibatch advantage normalisation).
- [ ] 7.4 *Diagnose from symptoms*: owner explains from config and schedule code why
      the LR rises after 100 updates and decides whether to fix it before arm runs.
- [ ] 7.5 Record demonstrated understanding from 7.1 to 7.4 in `learning/records/`;
      verify at least one record file exists.
