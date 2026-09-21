## Context
`update_model` runs `epochs x minibatches` optimizer steps and already computes approx KL on
clean samples per minibatch for diagnostics. `run_chess_training` calls `eval_fn` every
`eval_interval`, which includes the prior match (`prior_score`).

## Goals / Non-Goals
Goals: bound the per-update policy move without hand-tuning; make the named alarm halt the
run and keep the last good weights. Non-goals: PPO-penalty, adaptive lr.

## Decisions
- **KL check before the step, on the minibatch being stepped** (Spinning Up checks the epoch
  KL before stepping; per-minibatch is finer and free here). `target_kl` 0.03: our healthy
  runs sit at 0.02-0.04 per minibatch-mean; 0.01 would stop almost every update. At the
  extremes: very small -> one step per update (A2C-like, slow); very large -> off.
- **`mean_kl_stop`** is the dial: near 0 the guard is idle; near 1 every update is cut and the
  learning rate is the real problem.
- **Alarm restores weights** via a deep copy of both models' state dicts taken at each passing
  evaluation (cost: one copy per eval, ~85 MB for 256 filters). The returned/saved model is the
  restored one so `run_experiment` saves a good checkpoint.
- **Patience 2**: one bad 20-game match is inside the noise band (+-0.2); two in a row at 0.35
  was the LONG256B signature.

## Risks / Trade-offs
- A KL stop hides a too-high lr behind fewer steps; that is why `mean_kl_stop` is logged.
- The alarm depends on the 20-game prior match's noise; patience 2 and threshold 0.5 are the
  trade.
