## Why

LONG256B (2026-09-21) drifted out of PPO's trust band and collapsed: approx KL 0.04 -> 0.06,
clip fraction 0.16 -> 0.26, policy loss positive, held-out mate-in-1 0.80 -> 0.54, 3-48 against
its start. The alarm we had named (prior_score below par at two evaluations) tripped at update
350 and nothing stopped the run. Owner: "go, install all three".

## What Changes

- **Target-KL early stopping** in the PPO update: before each minibatch step the mean approx
  KL of that minibatch (clean samples) is checked; if it exceeds `target_kl` the remaining
  steps of this update are skipped. Reported as `mean_kl_stop` (share of updates cut short)
  and `mean_optimizer_steps`. Default `target_kl: 0.03` (our healthy band 0.02-0.04; Spinning
  Up's default 0.01 with a 1.5x stop). `null` disables.
- **Prior alarm stop**: when the periodic prior match reports `prior_score` below
  `stop_below_prior` (default 0.5) at `stop_patience` (default 2) consecutive evaluations,
  training stops and the models are restored to the last evaluated state that was at or above
  the threshold. Off when there is no prior match. Logged as `alarm_stop: <update>`.
- **Recipe note, not code**: continuations of the 256-filter line use `learning_rate 5e-5`
  (arm LR5: KL 0.019, clip 0.11, puzzles held).

## Capabilities

### New Capabilities
- (none)

### Modified Capabilities
- `policy-update`: target-KL early stopping requirement; training-stop-on-alarm requirement.

## Alternatives considered

- **Lower learning rate only** (LR5). Works for this line but is hand-tuned per network; the
  KL stop makes the step self-limiting (Spinning Up PPO docs, fetched).
- **Adaptive KL penalty** (Schulman et al. 2017 §4, PPO-penalty). Heavier: a Lagrangian
  coefficient; the clipped objective plus early stop is what the reference code ships.
- **Smaller clip epsilon**. Clipping alone "is still possible to end up with a new policy which
  is too far from the old policy" (Spinning Up); it did not save LONG256B at 0.2.
- **No alarm, read the dashboard**. The run cost 55 GPU minutes after the alarm tripped.

## How we will know it worked

Arm LONG256C: LONG256 + 600 updates, lr 5e-5, target_kl 0.03, alarm on. Watch `mean_approx_kl`
(stay <= 0.03), `mean_kl_stop` (how often the guard fires; > 0.5 means the lr is still too
high), `prior_score` (>= 0.5), puzzles (LONG256: 0.802 / 0.728 / 0.666 / 0.573), and the
matrix vs LONG256. Owner's prediction: (not given).

## What to understand

- Why clipping does not bound the policy step and what the KL check adds (Spinning Up).
- What `mean_kl_stop` tells you about the learning rate.
- Why the alarm restores the last good weights instead of saving the final ones.
