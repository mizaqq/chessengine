## 1. Target-KL early stop
- [ ] 1.1 `update_model(target_kl=None)`: per-minibatch clean-sample KL check before the step; `kl_stop` and `optimizer_steps` in the returned stats. Tests in `tests/model/test_ppo_update.py`: guard fires at the right step, silent when KL small, `target_kl=None` unchanged.
- [ ] 1.2 Config `target_kl` (default 0.03, null = off) threaded through `run_chess_training` -> `update_kwargs`; validation test in `tests/entrypoints/test_config.py`.

## 2. Prior alarm stop
- [ ] 2.1 `run_chess_training(stop_below_prior=None, stop_patience=2)`: consecutive-below counter, state-dict snapshot at passing evals, restore + `alarm_stop` log entry + break. Test with a fake `eval_fn` returning a scripted prior_score sequence.
- [ ] 2.2 Config `stop_below_prior` (default 0.5) / `stop_patience` (default 2); only active when `prior_eval_games > 0`.

## 3. Experiment
- [ ] 3.1 Driver `experiments/width/run_long256c.sh`: LONG256 + 600, lr 5e-5, target_kl 0.03, alarm on; games, finishes, audit, matrix vs LONG256. Record `experiments/width/outcome.md`.

## 4. Comprehension
- [ ] 4.1 Comprehension check with owner (CLAUDE.md "Diagnose from symptoms"): given LONG256B's KL / clip / policy-loss series, say which of the two guards would have fired first and at which update, then check against the logs.
