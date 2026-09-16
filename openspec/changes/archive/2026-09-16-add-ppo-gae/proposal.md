## Why

The update step is one REINFORCE-with-baseline gradient step per rollout on the
shared network: advantage = full 15-ply shaped return minus value (GAE with
lambda = 1, the noisiest estimator), and nothing bounds how far one step moves the
policy. Three runs on 2026-09-16 showed the consequence: opening-board normalised
entropy fell 0.51 -> 0.37 -> 0.24 and games collapsed into repetition and
fifty-move draws (~80-180 plies), so mate chances never arise and the curriculum's
puzzle skill (held-out top-1 0.486) does not transfer in-game (4.5%). This is
roadmap item 03/1 and the standard remedy: Generalized Advantage Estimation to
trade variance for a little bias, and the PPO clipped objective to bound each step
and allow several optimisation epochs per rollout.

## What Changes

- Add a GAE(gamma, lambda) advantage estimator that mirrors the existing shaped
  returns walk (cross-player done propagation, opponent bootstrap, potential-based
  shaping). lambda = 1 reproduces today's advantage exactly.
- Make the update algorithm config-selectable: `algorithm: a2c | ppo`. A2C stays
  the default; the default yaml is unchanged.
- Add a PPO update: clipped surrogate objective, `ppo_epochs` passes of
  `ppo_minibatches` shuffled minibatches, per-minibatch advantage normalisation, a
  value-loss coefficient, the existing normalised-entropy bonus and gradient clip.
- Rollout collection runs under `no_grad` and stores what re-evaluation needs
  (oriented observation, legal mask, action, old log-prob, old value). The update
  re-runs the network. A2C is the degenerate PPO configuration (1 epoch, 1
  minibatch, no clip, lambda 1).
- Under the shared network, both colours' own steps form one batch for the
  epochs; the legacy two-network mode updates each model on its own steps.
- Log PPO diagnostics: clip fraction and approximate KL between old and new policy.
- Add `src/configs/train_ppo.yaml`; `steps_per_update: 64` (768 samples/update).
- Not **BREAKING**: default behaviour preserved; checkpoints unchanged.

## Decisions taken in conversation (2026-09-16)

1. Scope: both switches, three arms (A2C lambda 1, A2C lambda 0.95, PPO lambda 0.95).
2. Rollout length 64 plies for **all** arms so the algorithm is the only variable.
3. Fresh network, seed 42, 4 Lichess puzzle boards, equal plies per arm.
4. Paper hyperparameters: eps 0.2, 4 epochs, 4 minibatches, value coef 0.5,
   advantage normalisation on, lambda 0.95.

Owner's prediction: **skipped at the owner's request** ("for now skip questions
and implement"). The lambda comprehension question was parked; the plain-language
version to return to: lambda is how many moves ahead we look at what actually
happened before letting the value head fill in the rest.

## Alternatives considered

1. **GAE only, keep A2C** (Schulman et al. 2016, arXiv:1506.02438 §3). Isolates the
   variance effect; kept as arm B, not the whole change because step size stays
   uncontrolled.
2. **PPO clipped objective** (chosen; Schulman et al. 2017, arXiv:1707.06347 eq. 7,
   Algorithm 1).
3. **PPO adaptive KL penalty** (same paper §4): reported worse than clipping (§6.1).
4. **TRPO** (§2.2): second-order step, too heavy on CPU for a learning codebase.
5. **Extras from Huang et al. 2022 (37 details)**: only advantage normalisation
   (detail 7) and the two diagnostics (detail 12) are included; value-loss
   clipping, orthogonal init, Adam eps, LR annealing are follow-ups so their effect
   can be measured one at a time.

## BatchNorm handling (found during apply)

`ChessPolicyProbs` uses BatchNorm. Re-evaluating with the statistics of a
192-sample minibatch instead of the 12-board step that sampled the action moved
half the ratios outside the clip band before any learning (measured: clip
fraction 0.52, approx KL 0.10 on a fresh network). Training now keeps the
network in eval mode and refreshes the running statistics once per update from
the rollout (design D7). Ratios start at exactly 1; the maths tests hold on the
real network.

## Capabilities

### New Capabilities
- `advantage-estimation`: per-step advantages and value targets from a self-play
  rollout, lambda and its limits.
- `policy-update`: rollout to parameter updates; A2C vs PPO; PPO objective,
  hyperparameters, diagnostics; shared-network batching.

## Impact

- `src/model/returns.py`: `compute_gae_for_model` beside the existing returns.
- `src/core/types.py`: `StepRecord` holds obs, mask, action, old log-prob, old
  value, entropy (for stats), legal counts, player, potential, done, terminal rewards.
- `src/model/training.py`: collection under `no_grad`; `update_model`; dispatch.
- `src/training/metrics.py`: `clip_fraction`, `approx_kl`.
- `src/entrypoints/train.py`, `src/configs/train_ppo.yaml`: keys, presets, validation.
- Tests: GAE, PPO update, preset equivalence, config, metrics, smoke.

## How we will know it worked

Three arms, seed 42, fresh network, 12 boards (4 Lichess puzzle boards),
`steps_per_update: 64`, 250 updates each (16,000 plies per board, about the plies
the seed0 + cont-lichess path saw). Configs differ only in the keys named.

| Arm | algorithm | gae_lambda | Purpose |
|-----|-----------|------------|---------|
| A | a2c | 1.0 | today's update, 64-ply rollouts |
| B | a2c | 0.95 | GAE alone |
| C | ppo | 0.95 | clipping + 4x4 epochs/minibatches + normalisation on top of GAE, lr 3e-4 |
| C2 | ppo | 0.95 | as C with lr 1e-4: at 3e-4 a fresh update leaves the clip band by minibatch 5 (clip fraction ~0.6), at 1e-4 by minibatch 11 |

Metrics per arm: opening-board and puzzle-board normalised entropy, draw rate and
mean plies on game boards, puzzle solved rate, Lichess and selfplay held-out
top-1/mate-prob, value loss, wall-clock per update; C only: clip fraction,
approx KL. Then `scripts/eval_games.py` on each final checkpoint (in-game mate rate).

Budget note (owner, 2026-09-16, while the arms were running): the standard
budget for future arms is 7,500 plies per board (the old single 500-update run),
i.e. about 120 updates at 64 plies; "then our progress slows a lot". These arms
stay at 250 and are also read at update 120.

Baseline for context: `experiments/mixed-puzzle-sources/cont-lichess` (Lichess
0.486, in-game 4.5%, opening entropy 0.24).


### Outcome (2026-09-16, 4 arms, seed 42, 250 updates x 64 plies)

| Arm | Lichess top-1 @150 / @250 | opening entropy (range 50-250) | value loss | clip / KL | in-game mate (opps) | s/update |
|-----|------|------|------|------|------|------|
| A a2c lam 1 | 0.29 / 0.32 | 0.15-0.55, dipped to 0.15 | 0.85-1.6 | - / 0 | 8/90 = 0.089 | 4.6 |
| B a2c lam 0.95 | 0.32 / 0.32 | 0.03-0.50 (collapsed at 100, recovered) | 0.9-1.3 | - / 0 | 5/71 = 0.070 | 4.6 |
| C ppo lr 3e-4 | 0.41 / 0.44 | 0.21-0.31 | 0.42-0.46 | 0.24-0.36 / 0.09-0.18 | 6/120 = 0.050 | 14.5 |
| C2 ppo lr 1e-4 | 0.41 / 0.42 | 0.51-0.67, never collapsed | 0.46-0.57 | 0.22-0.38 / 0.04-0.09 | 7/101 = 0.069 | 17.4 |

Reference: `experiments/mixed-puzzle-sources/cont-lichess` (a2c, 15-ply
rollouts, 1000 updates = 15,000 plies): Lichess 0.486, in-game 4.5%;
`shared-network/seed0` (a2c, 15-ply, 500 updates = 7,500 plies): 0.427.

Reading:
- **PPO learned the puzzles from ~10 points more of the same plies than A2C with
  64-ply rollouts** (0.44 vs 0.32), and fit the value head twice as well.
- **But the A2C arms are handicapped by the rollout length**: one gradient step
  per 768 samples gives 250 steps in the run, versus 1000 steps in the old
  15-ply runs, and the old fresh 7,500-ply run reached 0.427. A2C's learning is
  bounded by gradient steps, not plies; PPO's 16 steps per rollout (4000 in
  total) is exactly the fix for that. Per ply, PPO ~= old 15-ply A2C on puzzles.
- **Learning rate 1e-4 kept the opening policy alive** (entropy 0.5-0.67 all
  run) at almost no puzzle cost (0.42 vs 0.44) and half the KL. Clip fraction
  sat near 0.24 in both PPO arms regardless of LR; KL, not clip fraction,
  separated them. Both KLs are above the ~0.01-0.02 practitioners target.
- **The game problem is not solved by any arm**: 40-game evaluations still end
  around 200-220 plies with 47-70% draws and 5-9% in-game mate rate (counts of
  5-8 mates, so the arms are indistinguishable there).
- GAE alone (B vs A) did not change puzzle accuracy; its entropy curve went
  through a collapse and recovery, so a single seed says little about it.
- Wall-clock: PPO 3-4x per update. Owner's standard budget going forward:
  7,500 plies per board (~120 updates at 64 plies).

Owner's prediction was skipped, so there is no gap to discuss; the comprehension
tasks (7.x) are open and should use these numbers.

## What to understand

- Return vs value vs advantage, and why the policy gradient wants the advantage.
- The TD residual and how GAE sums residuals; lambda 0 and 1; why the best lambda
  is usually below the best gamma.
- Why REINFORCE cannot reuse a rollout, and how the ratio plus clipping makes
  reuse safe.
- What clip fraction and approximate KL say about an update.
- Why A2C is PPO with one epoch and no clipping.
- Outside chess: PPO is the default optimiser in robotics and RLHF for language
  models; lambda is the same bias/variance dial as smoothing any noisy estimate.
