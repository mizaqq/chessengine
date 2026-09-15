## 1. Tests first

- [ ] 1.1 Update `test_returns_alternating_no_done` and
      `test_returns_black_model_perspective` in `tests/model/test_returns.py` to the
      new expectations (2 and -1) with docstrings explaining the attribution rule;
      verify they fail against the current implementation.
- [ ] 1.2 Add tests for the remaining scenarios: several opponent steps in a row,
      opponent reward at rollout end with bootstrap, opponent captures and mates,
      own mating move not polluted by the new game; verify they fail.

## 2. Implementation

- [ ] 2.1 Add the `pending` accumulator to `compute_returns_for_model` per design D1
      and update its docstring; verify `uv run pytest tests/model/test_returns.py`
      passes, including the unchanged single-player and done-on-own-step tests.
- [ ] 2.2 Run the full suite `uv run pytest -q`; verify green (integration tests
      only check for NaN and shape, so they should be unaffected).

## 3. Baseline run

- [ ] 3.1 Owner fills the prediction block in proposal.md; verify placeholders are
      replaced.
- [ ] 3.2 Run the default config for 500 updates with seed 42 and save logs to
      `experiments/credit-opponent-rewards/baseline.json`; verify the file exists.
- [ ] 3.3 Compare value loss, draw rate and loss curve against
      `benchmark_results.json` (sync arm) and write an "Outcome" paragraph under
      "How we will know it worked" in proposal.md; verify it is present.

## 4. Docs

- [ ] 4.1 Update the "Returns Computation" note in
      `knowledge/repos/chessengine/services/training-loop.md` (local) and the README
      sentence on returns; verify by reading the diff.
- [ ] 4.2 Add the verified Sutton & Barto reference to `learning/RESOURCES.md` and
      update `add-ppo-gae/design.md` D2 to include the `pending` accumulator via
      `/opsx:update add-ppo-gae`; verify both files mention it.

## 5. Comprehension check with owner

- [ ] 5.1 *Trace by hand* (CLAUDE.md): owner walks the "opponent captures and mates"
      scenario through the D1 pseudocode step by step, stating `R` and `pending`
      after each line, before the test is run.
- [ ] 5.2 *Review the diff*: owner answers why `pending` is reset before the
      current reward is added and what would go wrong if the order were swapped.
- [ ] 5.3 Record demonstrated understanding in `learning/records/`; verify a record
      file exists.
