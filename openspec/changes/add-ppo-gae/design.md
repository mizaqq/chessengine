## Context

See proposal.md for motivation. Facts about the current code that shape the design:

- `_collect_rollout` runs both models with gradients enabled and stores
  graph-connected `value_*`, `log_prob_*`, `entropy_*` tensors in `StepRecord`. The
  update then reads those tensors and calls `backward()` once per model. This is why
  the per-model tensor split exists (see `knowledge/synthesis/debugging-gotchas.md`,
  "Shared Computation Graph").
- `compute_returns_for_model` walks the rollout backwards, passes the return through
  opponent steps unchanged, applies `gamma` only on own steps, and cuts the chain on
  `done` by substituting the terminal reward. GAE must mirror this exactly.
- `_compute_bootstrap` returns `V_own(s_T)` if the model is to move at the end of the
  rollout, else `-V_opp(s_T)` (zero-sum assumption).
- The default rollout is 12 envs x 15 steps = 180 transitions, roughly 90 per model.
- The LR schedule halves the LR every `lr_decay_interval` updates with a hard-coded
  floor of 3e-4, while the default `learning_rate` is 1e-4. After 100 updates the LR
  therefore rises to 3e-4. Pre-existing; noted under Risks.

## Goals / Non-Goals

**Goals:**
- One rollout format that both algorithms consume.
- GAE that is provably equal to the current returns at lambda = 1 (tested).
- A2C preset whose gradient equals today's (tested), so the algorithm switch is the
  only variable in the comparison experiment.
- PPO as in Schulman et al. 2017 Algorithm 1 plus per-minibatch advantage
  normalisation and the two diagnostics from Huang et al. 2022.

**Non-Goals:**
- Value-loss clipping, orthogonal initialisation, Adam epsilon, LR annealing,
  early stopping on a KL target. Each is a candidate follow-up change so its effect
  can be measured alone.
- Changing the network, reward shaping, or the LR schedule.
- GPU support.

## Decisions

### D1. Re-evaluate at update time; collect under `no_grad`

`StepRecord` stores `obs`, `legal_mask`, `action`, `old_log_prob`, `old_value`
(plus the existing `player`, `reward_white`, `done`, terminal rewards and legal
counts). The update re-runs the model on the stored observations to get gradients.

Why: PPO needs several gradient steps on the same data, which is impossible with
graph tensors that are freed on the first `backward()`. Re-evaluation also makes the
per-model graph-separation problem disappear, because nothing in the rollout holds
a graph. Collection becomes cheaper (no autograd bookkeeping).

Alternative: keep the current graph-tensor path for A2C and add a separate buffer
for PPO. Rejected: two rollout formats, two update paths, and the A2C path would
keep the subtle graph-sharing constraint. The unified path costs one extra forward
pass per update under A2C.

Key fact for the owner: with theta = theta_old the ratio is exactly 1 and
`grad(ratio * A) = A * grad(log pi)`, so one unclipped full-batch PPO step is the
REINFORCE-with-baseline step. That is why A2C can be a preset of the same code.

### D2. GAE mirrors the returns walk

`compute_gae_for_model(steps, model_id, bootstrap, gamma, lam)` walks backwards
keeping per-env `next_value` and `gae`:

```
for i in reversed(steps):
    next_value = next_value * (1 - done) + terminal_r          # cut chain
    gae        = gae * (1 - done)                              # reset accumulator
    if own step:
        delta = reward + gamma * next_value - old_value
        gae   = delta + gamma * lam * gae
        adv[i] = gae; target[i] = gae + old_value
        next_value = old_value
```

Opponent steps only apply the `done` handling, exactly as `compute_returns_for_model`
does. The function returns lists of `[num_envs]` tensors for advantage and target;
entries at non-own steps are zero and are masked out by the update.

Alternative: rewrite returns in terms of GAE and delete the old function. Rejected
for now: keeping `compute_returns_for_model` gives a free equivalence test. It can
be removed at archive time if the test is kept as a direct numeric check.

### D3. Update step shape

```
def update_model(model, optimizer, batch, cfg) -> dict:
    # batch: own-step tensors (obs, mask, action, old_logp, old_value, adv, target, n_legal)
    for epoch in range(cfg.epochs):
        for mb in shuffled_minibatches(batch, cfg.minibatches):
            probs, value = model(mb.obs, mb.mask)
            dist = Categorical(probs)
            new_logp, entropy = dist.log_prob(mb.action), dist.entropy()
            adv = normalise(mb.adv) if cfg.normalize_advantage else mb.adv
            ratio = exp(new_logp - mb.old_logp)
            policy_loss = -min(ratio*adv, clamp(ratio, 1-eps, 1+eps)*adv).mean()
            value_loss  = (value.squeeze(-1) - mb.target).pow(2).mean()
            ent_norm    = entropy / clamp(log(mb.n_legal), 1e-8)
            loss = policy_loss + cfg.value_coef*value_loss - cfg.entropy_coef*ent_norm.mean()
            step with grad clip
            record clip_fraction = mean(|ratio-1| > eps), approx_kl = mean(old_logp - new_logp)
```

Presets, applied in `run_training_from_config`:

| key | a2c | ppo | source |
|-----|-----|-----|--------|
| ppo_epochs | 1 | 4 | PPO paper Table 3 uses 3 (Atari), Table 5 uses 10 (MuJoCo); 4 is a CPU-friendly middle |
| ppo_minibatches | 1 | 4 | paper Table 3: 128*8 / 32 = 32 minibatches; scaled down for our batch size |
| clip_epsilon | inf (no clip) | 0.2 | paper Table 5; 0.1 in Table 3 |
| normalize_advantage | false | true | Huang et al. detail 7 |
| value_coef | 1.0 | 0.5 | current code; paper eq. 9 c1 = 0.5 |
| gae_lambda | 1.0 | 0.95 | current behaviour; paper Tables 3 and 5 |

`ComposedLoss` keeps composing the three terms; it gains a `value_coef` argument.

### D4. What each hyperparameter controls

- **gae_lambda** (0..1). Weight on longer-horizon residuals. At 0 the advantage is
  the one-step TD error: low variance, biased by every error in V. At 1 it is the
  full return minus V: unbiased given gamma, high variance. GAE paper Section 3:
  lambda "introduces bias only when the value function is inaccurate", so its best
  value is typically below gamma's.
- **clip_epsilon** (>0). Half-width of the trusted ratio band. Tiny values freeze the
  policy (most samples clipped, gradient near zero, clip fraction near 1). Infinite
  value removes the trust region and multiple epochs become unsafe.
- **ppo_epochs** (>=1). Passes over the rollout. More epochs squeeze more learning
  from the same data but push the policy further from the one that collected it;
  approx KL and clip fraction rise. 1 epoch with no clipping is A2C.
- **ppo_minibatches** (>=1). Splits each epoch. More minibatches mean more, noisier
  steps and stronger per-minibatch normalisation noise; with ~90 own samples per
  model, 4 minibatches of ~22 is already small. The PPO config raises
  `steps_per_update` to 64 to give ~384 samples per model.
- **normalize_advantage**. Rescales advantages so the policy step size does not
  depend on reward scale. Extreme: a minibatch of 1 gets advantage 0 and learns
  nothing from the policy term.
- **value_coef**. Balances value fitting against policy improvement in the shared
  ResNet trunk. Very high: trunk features serve the critic, policy stalls. Very low:
  V lags, advantages become noisy, clip fraction rises.

### D5. Diagnostics

`MetricsAggregator.add_ppo_stats(clip_fraction, approx_kl)` accumulates per update;
`episode_summary()` reports means and resets. Under `a2c` the fields are null.
Approx KL uses the simple estimator `mean(old_logp - new_logp)` named in Huang et al.
detail 12; the unbiased `(r-1) - log r` variant is a possible follow-up.

## Risks / Trade-offs

- [Small per-model minibatches make normalisation noisy] → PPO config uses
  `steps_per_update: 64`; the experiment reports wall-clock so the cost is visible.
- [Entropy collapses faster with several epochs] → entropy stays in the loss and in
  the logs; the prediction table asks the owner to forecast it.
- [Re-evaluation doubles forward passes under A2C] → accepted; CPU rollout is
  dominated by environment stepping. Measured in the experiment.
- [Ratio can blow up if an action's new probability underflows] → actions were
  sampled from the masked distribution, so old_logp is finite; ratios are computed
  in log space and clipping bounds the loss. A NaN guard test is included.
- [Pre-existing LR floor bug: LR rises from 1e-4 to 3e-4 after 100 updates] → out of
  scope, but the owner is asked to diagnose it from the config and schedule code as
  a comprehension task before the experiment, and to decide whether to fix it first.
- [Existing tests build `StepRecord` with the old fields] → updated in the same
  change; the returns tests keep passing since the returns function is unchanged.

## Migration Plan

Single commit series on `main` behind the `algorithm` default. Rollback is
`algorithm: a2c`, which is the default. No data or checkpoint format changes.

## Open Questions

- Whether to remove `compute_returns_for_model` after archive, keeping the lambda=1
  equivalence as a numeric test only. Does not affect specs or tasks.
