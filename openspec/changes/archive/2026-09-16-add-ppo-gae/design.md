## Context

Facts about the code (2026-09-16) that shape the design:

- Shared network is the default: `_collect_rollout(..., oriented=True)` runs one
  model on mover-oriented observations and stores graph-connected `value_*`,
  `log_prob_*`, `entropy_*` per side in `StepRecord`; `_gather_side` concatenates
  each side's own steps; `_loss` sums both sides in one backward pass.
- `compute_returns_for_model` walks backwards with potential-based shaping: on own
  steps `R = shaped + gamma * R`, opponent steps pass R through, `done` cuts the
  chain with the terminal reward and zeroes the next potential.
- `_compute_bootstrap` gives `V(s_T)` if the model moves at the end, else `-V_opp`.
- 4 of 12 boards are one-move puzzle episodes (done every ply).
- Profile: forward 0.65 s, backward 1.02 s, env 0.07 s per 15-ply update.
- `ChessPolicyProbs` has BatchNorm (train mode during training).

## Goals / Non-Goals

Goals: one rollout format both algorithms consume; GAE equal to the returns at
lambda 1 (tested); a2c preset whose maths equals today's loss (tested without BN);
PPO per Schulman 2017 Algorithm 1 plus advantage normalisation and diagnostics.

Non-goals: value-loss clipping, orthogonal init, Adam eps, LR annealing, KL early
stopping, network changes, GPU.

## Decisions

### D1. Collect under `no_grad`; re-evaluate at update time

`StepRecord` stores `obs` (as the model saw it, oriented), `legal_mask`, `action`,
`old_log_prob`, `old_value`, `entropy` (collection-time, for logging only),
`num_legal`, `player`, `potential_white`, `done`, `terminal_r_white/black`. All
per-env `[num_envs, ...]` tensors, no graph. PPO needs several gradient steps on
the same data; graph tensors die on the first backward. Side effect: the
"separate computation graphs" constraint (debugging-gotchas) disappears.

### D2. GAE mirrors the returns walk

```
next_value = bootstrap; gae = 0; phi_next = sign * final_material
for i reversed:
    next_value = next_value * (1 - done) + terminal_r
    gae        = gae * (1 - done)
    phi_next   = phi_next * (1 - done)
    if own step:
        shaped = scale * (gamma * phi_next - phi)
        delta  = shaped + gamma * next_value - old_value
        gae    = delta + gamma * lam * gae
        adv[i] = gae; target[i] = gae + old_value
        next_value = old_value; phi_next = phi
```
Opponent steps only apply the done handling. With lam = 1, `gae_i = R_i - V_i`
(telescoping), so `target = R` and the existing returns function gives a free
equivalence test. Terminal reward is discounted once by gamma, as today.

### D3. Update step

```
update_model(model, optimizer, batch, *, epochs, minibatches, clip_epsilon,
             normalize_advantage, value_coef, entropy_coef, grad_clip) -> stats
  for epoch: for mb in shuffled split:
    probs, value = model(mb.obs, mb.mask)
    dist = Categorical(probs); new_logp = dist.log_prob(mb.action); ent = dist.entropy()
    adv = (mb.adv - mean) / (std + 1e-8) if normalize else mb.adv
    ratio = exp(new_logp - mb.old_logp)
    policy_loss = -min(ratio * adv, clamp(ratio, 1-eps, 1+eps) * adv).mean()
    value_loss  = (value - mb.target)^2.mean()
    ent_norm    = ent / clamp(log(mb.num_legal), 1e-8)
    loss = policy_loss + value_coef * value_loss - entropy_coef * ent_norm.mean()
    zero_grad; backward; clip_grad_norm(grad_clip); step
    clip_fraction += mean(|ratio - 1| > eps); approx_kl += mean(old_logp - new_logp)
```
`clip_epsilon = None` means no clipping (a2c preset). Shared mode: the batch is
the concatenation of both sides' own steps. Legacy mode: one call per model.
`ComposedLoss.compute` gains `value_coef` (default 1.0).

Presets resolved in `resolve_algorithm(config)`:

| key | a2c | ppo | source |
|-----|-----|-----|--------|
| ppo_epochs | 1 | 4 | PPO Table 3: 3 (Atari); Table 5: 10 (MuJoCo) |
| ppo_minibatches | 1 | 4 | Table 3: 32 minibatches of 32 from 1024; scaled to 768/4 = 192 |
| clip_epsilon | none | 0.2 | Table 5 (0.1 in Table 3) |
| normalize_advantage | false | true | Huang et al. detail 7 |
| value_coef | 1.0 | 0.5 | today's code; PPO eq. 9 c1 = 0.5 |
| gae_lambda | 1.0 | 0.95 | today's behaviour; Tables 3 and 5 |

Explicit keys override the preset. Validation: `gae_lambda` in [0, 1],
`clip_epsilon` > 0, `ppo_epochs` >= 1, `ppo_minibatches` >= 1, `value_coef` >= 0.

### D4. What each hyperparameter controls

- **gae_lambda**: 0 = one-step TD (low variance, biased by every value error);
  1 = return minus value (no bias from V, high variance). GAE §3: lambda biases
  only when V is inaccurate, so best lambda < best gamma.
- **clip_epsilon**: trusted ratio band. Tiny: policy frozen, clip fraction ~1.
  None: no trust region, multiple epochs unsafe.
- **ppo_epochs**: passes over the rollout. More = more learning per ply, but KL
  and clip fraction rise. 1 epoch, no clip = A2C.
- **ppo_minibatches**: more = more, noisier steps and noisier per-minibatch
  normalisation. 768 / 4 = 192 samples each.
- **normalize_advantage**: step size independent of reward scale. Minibatch of 1
  gets advantage 0.
- **value_coef**: balance of critic vs policy in the shared trunk.

### D5. Diagnostics

`MetricsAggregator.add_ppo_stats(clip_fraction, approx_kl)`; summary reports
means, null when never called (a2c). Approx KL = `mean(old_logp - new_logp)`
(Huang et al. detail 12). Under BatchNorm the first minibatch already shows a
non-zero KL; this is the BN drift named in the proposal.

### D7. BatchNorm: eval mode during training, statistics refreshed per update

Measured on a fresh network with no parameter change (12-board collection vs
192-sample re-evaluation, both train mode): clip fraction 0.52, approx KL 0.10;
with fixed statistics both are exactly 0. Clipping would discard half the samples
before learning started. Decision: `run_chess_training` keeps the network in
eval mode for collection and update, and after each update
`refresh_norm_stats(model, batch.obs, batch.mask)` runs one gradient-free
train-mode pass over the rollout to update the running statistics (momentum 0.1,
so they average roughly the last ten updates). Applies to all algorithms so the
arms share network semantics; the a2c arm therefore differs from pre-change runs
by using one-update-stale statistics. Alternatives: GroupNorm (batch-independent
by construction; changes the checkpoint format, so deferred), plain eval-mode BN
with frozen init statistics (no normalisation at all; rejected).

### D6. Budget

64-ply rollouts for all arms. Estimated per update on this CPU: collection ~3 s
(no_grad forward on 768 samples over 64 steps), measured 2026-09-16: ppo 13 s per update (3-update run); a2c to be measured. Run sequentially in the background.

## Risks / Trade-offs

- [BN batch-statistics drift breaks exact ratio 1] -> documented deviation; tests
  use eval-mode / no-BN model for the maths; drift reported via approx_kl.
- [Entropy collapses faster with 16 steps per rollout] -> entropy stays in loss
  and logs, split game/puzzle.
- [Wall-clock] -> equal plies, wall-clock reported.
- [Ratio overflow] -> computed in log space; actions came from the masked
  distribution so old_logp is finite; NaN guard test.
- [Existing `StepRecord` tests] -> `_make_step` helper updated.

## Migration Plan

Behind `algorithm` default a2c. Rollback = a2c. No checkpoint change.
`scripts/profile_update.py` bucket renamed to `update_model`.
