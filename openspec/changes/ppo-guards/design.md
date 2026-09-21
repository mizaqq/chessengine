## Context
`update_model` runs `epochs x minibatches` optimizer steps and already computes approx KL on
clean samples per minibatch for diagnostics. `run_chess_training` calls `eval_fn` every
`eval_interval`, which includes the prior match (`prior_score`).

## Goals / Non-Goals
Goals: bound the per-update policy move without hand-tuning; make the named alarm halt the
run and keep the last good weights. Non-goals: PPO-penalty, adaptive lr.

## Decisions
- **KL check on the whole batch before each pass, stop at 1.5 x target** (Spinning Up's rule;
  a first version checked per minibatch at the target and cut every LONG256C update to 6.5
  of 16 steps because minibatch KLs swing far wider than the mean). Measured on a LONG256
  batch: epoch-mean KL at lr 1e-4 = 0.009 / 0.055 / 0.081 / 0.084, at 5e-5 = 0.003 / 0.015 /
  0.032 / 0.043. `target_kl` 0.02 (owner, 2026-09-21): ~3 passes at 5e-5, 1 at 1e-4. Extremes:
  0.01 (reference default) = one pass at either rate; >= 0.05 = never fires here.
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
