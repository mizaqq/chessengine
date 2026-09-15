## Why

Per-step material rewards emitted on the opponent's move are currently dropped from
a model's return. A model is rewarded for its own captures but never directly
penalised for being captured; the loss only reaches it through the value estimate
of the next position and the final result. This biases value targets upward and
makes the shaping signal asymmetric. It must be fixed before `add-ppo-gae`, so that
the PPO comparison runs against a corrected baseline.

## What Changes

- A model's reward for its own step becomes the material change of its move plus
  the material change of every opponent step up to its next own step, all from that
  model's perspective. Discounting stays per own decision, as today.
- Rewards from steps after a game end are never credited to steps before it; the
  terminal reward still replaces the future value at `done`.
- Existing return tests that pinned the old skipping behaviour are updated; new
  tests pin the new attribution and the boundary cases.
- No config, model, or environment change. Not **BREAKING** for config, but returns
  and therefore training dynamics change for every run after this.

## Alternatives considered

1. **Credit opponent-step rewards to the preceding own step** (chosen). The
   opponent is part of the environment from one player's viewpoint; the transition
   from my decision to my next decision includes their reply and its reward. Source
   to be confirmed: Sutton & Barto, *Reinforcement Learning: An Introduction*,
   Chapter 1 tic-tac-toe example (see `learning/RESOURCES.md` once verified).
2. **Keep skipping, rely on the value head** (current). The next-position value does
   encode the loss, but only after the value head has learned it, and the return
   targets used to train that head omit the loss, so the signal is circular and
   optimistic. Rejected.
3. **Drop material shaping entirely and use only terminal rewards.** Removes the
   asymmetry by removing the signal. Sparse reward in chess is exactly what shaping
   is there to mitigate; this is a separate roadmap question (04/1 reward decay).
   Not chosen here.
4. **Apply a discount step on opponent moves too.** Would halve the effective
   horizon per own decision and change the meaning of `gamma`. Rejected; the
   opponent's reply is part of one transition, not a separate decision.

## Capabilities

### New Capabilities
- `return-attribution`: which rewards are credited to which step of which model
  when computing discounted returns from a self-play rollout.

### Modified Capabilities
- none

## Impact

- `src/model/returns.py`: `compute_returns_for_model` accumulates pending
  opponent-step rewards.
- `tests/model/test_returns.py`: two existing expectations change (alternating case,
  black-perspective case); new boundary tests added.
- `add-ppo-gae` design D2 must mirror the same accumulation; noted there.
- Baseline logs collected before this change are not comparable to logs after it.

## How we will know it worked

Unit tests pin the attribution. Then one short baseline run of the default config,
same seed as the last benchmark, 500 updates, logs saved to
`experiments/credit-opponent-rewards/baseline.json`. Metrics to watch against the
previous benchmark run: value loss level, draw rate, and whether the loss curve
shifts downward once losses are counted.

Owner's prediction (fill in before the run):

> **Value loss vs old run:** ...
> **Draw rate vs old run:** ...
> **Anything else you expect to move:** ...

## What to understand

- Why the opponent counts as environment from one player's viewpoint.
- The difference between a reward reaching the policy through the return and
  reaching it only through the value estimate, and why the second is slower and
  circular.
- How reward attribution errors show up in practice: optimistic value heads,
  policies that chase gains and ignore losses. Business analogue: a pricing agent
  credited for sales but not for churn caused between its decisions.
