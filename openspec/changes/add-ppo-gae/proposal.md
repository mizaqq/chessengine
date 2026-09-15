## Why

The current update is one REINFORCE-with-baseline gradient step per rollout, using
the full discounted return minus the value as the advantage. That advantage is the
highest-variance choice available (GAE with lambda=1), and nothing limits how far a
single update can move the policy. This is roadmap item 03/1 and the most-cited
remedy in the literature: Generalized Advantage Estimation to trade variance for a
little bias, and the PPO clipped objective to bound each policy step and allow
several optimisation epochs per rollout.

## What Changes

- Add a GAE(gamma, lambda) advantage estimator for the per-model self-play rollout.
  It keeps the existing cross-player done propagation and opponent bootstrap. With
  lambda=1 it reproduces the current discounted-return advantage exactly.
- Make the update algorithm config-selectable: `algorithm: a2c | ppo`. A2C remains
  the default so existing configs and logs stay comparable.
- Add a PPO update: clipped surrogate objective, several epochs of shuffled
  minibatches over the rollout, per-minibatch advantage normalisation, a value loss
  coefficient, and the existing normalised-entropy bonus and gradient clipping.
- Rollout collection stores what an update needs to re-evaluate the policy
  (observation, legal mask, action, old log-probability, old value) instead of
  graph-connected tensors. A2C becomes the degenerate PPO configuration (one epoch,
  one minibatch, no clipping) and its gradient is mathematically identical to today's.
- Log PPO diagnostics: clip fraction and approximate KL divergence between old and
  new policy.
- Add a `train_ppo.yaml` config with sourced defaults.
- Not **BREAKING**: default config behaviour is preserved; `steps_per_update` and
  existing keys keep their meaning.

## Alternatives considered

1. **GAE only, keep A2C** — Schulman et al. 2016, *High-Dimensional Continuous
   Control Using GAE*, arXiv:1506.02438, Section 3. Cheapest change and it isolates
   the variance-reduction effect. Rejected as the whole change because it leaves
   step size uncontrolled, but it is kept as an experiment arm (A2C with lambda<1).
2. **PPO with clipped objective** (chosen) — Schulman et al. 2017, *Proximal Policy
   Optimization Algorithms*, arXiv:1707.06347, eq. 7 and Algorithm 1. Bounds the
   per-sample policy ratio so multiple epochs on the same rollout are safe.
3. **PPO with adaptive KL penalty** — same paper, Section 4. Penalises KL(old, new)
   with a coefficient adjusted toward a target. The paper reports it performs worse
   than clipping in its experiments (Section 6.1), so not chosen.
4. **TRPO** — discussed in the PPO paper Section 2.2 as the method PPO simplifies.
   Requires a conjugate-gradient second-order step and is incompatible with
   shared-parameter noise like dropout. Too heavy for a learning codebase on CPU.
5. **Implementation extras from Huang et al. 2022, *The 37 Implementation Details
   of PPO*, ICLR Blog Track** — value-loss clipping, orthogonal init, Adam eps 1e-5,
   linear LR annealing. Only the details the paper itself specifies plus advantage
   normalisation and the two diagnostics are included; the rest are recorded as
   follow-ups so their effect can be tested one at a time.

## Capabilities

### New Capabilities
- `advantage-estimation`: how per-step advantages and value targets are computed
  from a self-play rollout, including the lambda parameter and its limits.
- `policy-update`: how a rollout is turned into parameter updates, selection between
  A2C and PPO, the PPO objective and its hyperparameters, and the diagnostics logged.

### Modified Capabilities
- none (no main specs exist yet)

## Impact

- `src/model/returns.py`: new GAE function beside the existing returns function.
- `src/core/types.py`: `StepRecord` stores observation, legal mask, action, old
  log-prob and old value (detached) instead of graph-connected per-model tensors.
- `src/model/training.py`: rollout collection under `no_grad`; update step
  dispatches on algorithm; PPO epochs/minibatches; diagnostics into metrics.
- `src/training/metrics.py`: clip fraction and approx KL fields.
- `src/entrypoints/train.py`, `src/configs/`: new keys and a PPO config.
- Tests: new GAE tests, PPO loss tests, A2C gradient-equivalence regression test,
  smoke test with the PPO config. Existing `StepRecord`-based tests updated.
- Roadmap 03/1 marked done at archive time.

## How we will know it worked

Experiment: three arms, same seed, same env count and budget (proposed 12 envs,
500 updates each). Configs differ only in the keys named.

| Arm | algorithm | gae_lambda | Purpose |
|-----|-----------|------------|---------|
| A | a2c | 1.0 | baseline, identical to today (potential shaping, scale 0.2) |
| B | a2c | 0.95 | effect of GAE alone |
| C | ppo | 0.95 | effect of clipping + epochs on top of GAE |

Metrics to watch, per arm, from the logs:
- total, policy and value loss over updates (level and noise)
- normalised entropy (should not collapse faster under PPO than A2C)
- white/black/draw rates in the windowed summary
- PPO only: clip fraction and approx KL per update
- wall-clock per update
- mate-in-one rate on 40 sampled games from the final checkpoints (roadmap/06 §0)

Owner's prediction (fill in before the first run):

> **Loss noise A vs B:** ...
> **Value loss B vs A:** ...
> **Entropy C vs B:** ...
> **Clip fraction in C, early vs late:** ...
> **Wall-clock per update C vs A:** ...
> **Draw rate after 500 updates, any arm you expect to differ:** ...

## What to understand

After this change the owner should be able to explain, in their own words:
- Return vs value vs advantage, and why the policy gradient wants the advantage.
- The TD residual and how GAE sums residuals; what lambda=0 and lambda=1 mean and
  why the best lambda is usually below the best gamma.
- Why REINFORCE cannot reuse a rollout for several gradient steps, and how the
  probability ratio plus clipping makes reuse safe.
- What clip fraction and approximate KL tell you about an update, and what a very
  high or a zero clip fraction means.
- Why A2C is the special case of PPO with one epoch and no clipping.
- Where this applies outside chess: PPO is the standard policy optimiser in
  robotics and in RL from human feedback for language models; the lambda choice is
  the same bias/variance dial as smoothing in any noisy sequential estimate.
