## 1. Environment reports material

- [ ] 1.1 Add failing tests in `tests/envs/` that `EnvStep.material` is 0 after reset
      and 3 after a scripted white knight capture (Fool's-mate-style fixed move
      sequence), for both sync and async envs; verify they fail.
- [ ] 1.2 Replace the per-step difference with `material` in both vector envs and
      `EnvStep`; remove `reward`; update existing env tests that asserted on
      `reward`; verify `uv run pytest tests/envs` passes.

## 2. Shaped returns

- [ ] 2.1 Rewrite `tests/model/test_returns.py`: keep the single-player and
      done-on-own-step structure but express inputs as potentials; add tests for
      every scenario in the reward-shaping spec (capture/recapture, discounted
      potential, scale, mating while ahead, opponent mates with new game, sum to
      zero over a full game, window truncation on opponent's turn, two consecutive
      opponent steps, shaping none); verify they fail.
- [ ] 2.2 Change `StepRecord` (`potential_white` replaces `reward_white`), add
      `final_material`, `shaping_scale` parameters to `compute_returns_for_model`
      and implement design D2; verify `uv run pytest tests/model/test_returns.py`
      passes.
- [ ] 2.3 Update `_collect_rollout` to store the pre-move material and
      `run_chess_training` to pass the final observation's material and the scale;
      verify `uv run pytest tests/integration` passes.

## 3. Configuration

- [ ] 3.1 Parse `shaping` and `shaping_scale` in `src/entrypoints/train.py` with
      validation (accepted values listed in the error; scale >= 0); tests in
      `tests/entrypoints/test_config.py`; add both keys with comments to
      `src/configs/train_default.yaml`; verify tests pass.

## 3b. Readable outcome metric

- [ ] 3.2 Add `mean_terminal_return` (unshaped, white view, averaged over games that
      ended in the window) to `MetricsAggregator` and the log summary, so outcomes
      can be read next to the shaped loss; test in
      `tests/training/test_metrics_aggregation.py`; verify pass.

## 4. Verification and docs

- [ ] 4.1 Run `uv run pytest -q` and confirm green; run
      `uv run python -m src.entrypoints.train --max-updates 5` and confirm it
      finishes without NaN.
- [ ] 4.2 Update README reward description and the "Returns Computation" note in
      `knowledge/repos/chessengine/services/training-loop.md` (local); update
      `openspec/changes/add-ppo-gae/design.md` D2 so the GAE residual uses the
      shaped reward from this walk; verify by reading the diffs.
- [ ] 4.3 Add the AlphaZero reward source to `learning/RESOURCES.md` once fetched,
      or leave it under Gaps; verify the file is consistent.

## 5. Baseline run

- [ ] 5.1 Owner states the prediction in conversation; record it in proposal.md;
      verify the `_pending_` markers are gone.
- [ ] 5.2 Run the default config (potential, scale 1.0), seed 42, 500 updates; save
      logs to `experiments/potential-based-shaping/baseline.json`; verify the file
      exists.
- [ ] 5.3 Compare with the sync arm of `benchmark_results.json`; discuss against the
      prediction in conversation; write an "Outcome" paragraph in proposal.md.

## 6. Comprehension check with owner

- [ ] 6.1 *Trace by hand* (CLAUDE.md): owner walks the "mating while ahead" and
      "opponent mates" scenarios through the D2 pseudocode, stating `R` and
      `phi_next` after each line.
- [ ] 6.2 *Explain back*: owner explains why a winning move can carry a negative
      shaped return and what the value head learns to compensate.
- [ ] 6.3 *Review the diff*: owner answers what breaks if `phi_next` were zeroed
      after the own step instead of before it.
- [ ] 6.4 Record demonstrated understanding in `learning/records/`; verify a file
      exists.
