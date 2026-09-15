## Why

Per-step material rewards emitted on the opponent's move are dropped from a model's
return today (`compute_returns_for_model` keeps `R` unchanged on non-own steps). A
model is rewarded for its captures and never directly charged for being captured,
so value targets are optimistic and the shaping signal is asymmetric. Rather than
patch the attribution, this change replaces the ad-hoc material-difference reward
with potential-based shaping, which has a policy-invariance guarantee. It lands
before `add-ppo-gae` so PPO is compared against a corrected baseline.

Decided in conversation on 2026-09-15; this document is the record.

## What Changes

- The environment reports the material balance of the position it returns
  (`material`, white view) instead of a per-step material difference. The reward
  channel carries only terminal outcomes (win/loss/draw via `terminal_rewards`).
- A model's shaped reward for its own move at position `s` is
  `shaping_scale * (gamma * Phi(s') - Phi(s))`, where `s'` is the next position where
  that model is to move, `Phi` is material from that model's perspective, and the
  terminal position has `Phi = 0`. Opponent replies are inside the difference.
- New config keys: `shaping: potential | none` (default `potential`) and
  `shaping_scale` (default 1.0). `none` gives outcome-only reward.
- Return tests that pinned the old skipping behaviour are replaced by tests for the
  shaped reward, its terminal repayment, window truncation, and the `none` switch.
- **BREAKING** for logs: returns and value targets change magnitude and sign
  pattern; runs before and after are not comparable.

## Alternatives considered

1. **Credit opponent-step material changes to the preceding own step.** Same as
   potential shaping with `gamma` dropped from the potential term. Simple, but a
   heuristic without the invariance guarantee. Framing: Sutton & Barto, *RL: An
   Introduction*, Section 1.5 tic-tac-toe (opponent is part of the environment).
   Superseded by option 2.
2. **Potential-based shaping** (chosen). Ng, Harada & Russell 1999, *Policy
   invariance under reward transformations*, ICML. Theorem 1: `F = gamma*Phi(s') -
   Phi(s)` is necessary and sufficient for the shaped MDP to share optimal policies
   with the original. Corollary 2: `V'(s) = V(s) - Phi(s)`. Remark 1: near-optimal
   policies are preserved. Remark 2: with only potential-based reward every policy
   is optimal, so the outcome reward is the sole source of preference.
3. **Outcome-only reward.** The AlphaZero recipe (Silver et al. 2017/2018; not yet
   fetched, listed as a gap). Cleanest objective, but sparse: a random policy
   almost never wins in chess, so learning on a CPU budget may not start. Available
   here as `shaping: none` so it can be tested rather than assumed.
4. **Shaping that fades over training.** Roadmap 04/1. Becomes a schedule on
   `shaping_scale`; deferred so one variable changes at a time.

## Capabilities

### New Capabilities
- `reward-shaping`: how per-move rewards are formed from material potentials and
  terminal outcomes for each model in a self-play rollout, and how the shaping is
  configured.

### Modified Capabilities
- none (no main specs yet)

## Impact

- `src/envs/open_spiel_vector_env.py`, `src/envs/open_spiel_async_vector_env.py`:
  emit `material` of the returned observation; drop the per-step difference.
- `src/core/types.py`: `EnvStep.material`; `StepRecord.potential_white` (potential
  of the pre-move observation, white view) replaces `reward_white`.
- `src/model/returns.py`: the backward walk carries the next own potential and
  forms the shaped reward; `shaping` and `shaping_scale` parameters.
- `src/model/training.py`, `src/entrypoints/train.py`, `src/configs/`: pass the
  final observation's material for window truncation; new config keys.
- Tests in `tests/envs`, `tests/model/test_returns.py`, integration smoke.
- `add-ppo-gae` design D2 updated: the GAE residual uses the shaped reward from the
  same walk.

## How we will know it worked

Unit tests pin the formula and its boundaries. Then one baseline run of the default
config (shaping potential, scale 1.0), seed 42, 500 updates, logs saved to
`experiments/potential-based-shaping/baseline.json`, compared with the sync arm of
`benchmark_results.json` on value loss level, draw rate and loss trend.

Owner's prediction (stated in conversation before the run):

> **Value loss vs old run:** could go either way; the level of loss and return is
> not the test, model behaviour is.
> **Draw rate vs old run:** similar, still nearly all draws. The previous runs drew
> because the model never learned to checkmate, and this change does not by itself
> teach checkmating.
> **Anything else expected to move:** nothing specific.

**Outcome (2026-09-15, 500 updates, seed 42, sync, 12 envs):** loss level moved
(first-50 mean 14.8 -> 19.1, last-50 21.5 -> 25.1) as the owner predicted was
possible and irrelevant; draw share essentially unchanged (86% -> 87%, 246/287 ->
291/336 games), matching the prediction that nothing here teaches checkmating.
Games ended somewhat more often (287 -> 336) which is consistent with both sides
now being charged for material losses. `mean_terminal_return` hovers around the
draw value (-0.5) per window. Rerun with the same seed reproduced the numbers
exactly, so the loop is deterministic. Checkpoints saved under
`experiments/potential-based-shaping/` (gitignored). Baseline for `add-ppo-gae`.

**Follow-up (scale 0.2, same recipe):** value-vs-material correlation on the seeded
game flipped from +0.46/+0.39 to -0.28/-0.18 (owner predicted the sign), slope
-0.02 vs theoretical -0.2; 368 games, W/B/D 34/55/279 (24% decisive vs 13% at
scale 1); final loss ~1.0 vs ~25. Both seeded games ended in real checkmate.
Logs in `experiments/potential-based-shaping/scale02/run.json`.

## What to understand

- Why a reward for the transition between two of my decisions must include the
  opponent's reply, and why the opponent is "environment" from my seat.
- Potential-based shaping as a loan: paid out as material changes, repaid at the
  terminal state, summing to zero over any complete game, so the outcome is the
  only net reward.
- `V'(s) = V(s) - Phi(s)`: why the value head in the shaped game reads lower in
  materially winning positions, and why a winning move can carry a negative shaped
  reward at the end.
- What the shaping does and does not buy: faster credit assignment, no change to
  what is optimal; and what `shaping_scale` trades off with a small network.
- Business analogue: intermediate KPIs (clicks, engagement) as potentials that
  must net to zero against the real outcome, or the agent optimises the proxy.
