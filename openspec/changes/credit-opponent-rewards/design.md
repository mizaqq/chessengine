## Context

See proposal.md. `compute_returns_for_model` walks the rollout backwards with one
running value `R` per environment. On a `done` step it replaces `R` by the terminal
reward; on an own step it sets `R = reward + gamma * R`; on an opponent step it
leaves `R` untouched, which is where the opponent's reward is lost.

## Goals / Non-Goals

**Goals:** credit every reward in a game to exactly one own step of each model;
keep the walk O(steps) with tensor ops only; keep `gamma` per own decision.

**Non-Goals:** changing the environment reward, the terminal rewards, or anything in
the update step. GAE lands in `add-ppo-gae` and mirrors this walk.

## Decisions

### D1. A per-environment `pending` reward accumulator in the backward walk

```
R = bootstrap; pending = 0
for step in reversed(steps):
    done_f = step.done
    R = R * (1 - done_f) + terminal_r            # cut chain (unchanged)
    pending = pending * (1 - done_f)             # nothing from the next game leaks back
    r = step.reward_white (or its negation for black)
    is_mine = step.player == model_id
    new_R = (r + pending) + gamma * R
    R = where(is_mine, new_R, R)
    pending = where(is_mine, 0, pending + r)
    returns[i] = R
```

Why: one extra tensor and two `where` calls, same shape as the existing code, so the
returns tests stay readable and the GAE version can copy the pattern.

Alternative: pre-fold opponent rewards into the preceding own step during rollout
collection. Rejected: it would spread the attribution rule across collection and
returns, and it breaks when the rollout window ends on an opponent step, because the
preceding own step is already stored.

Ordering subtlety worth tracing: `pending` is reset by `done` *before* the current
step's reward is added, so a mating move's reward is still credited, while anything
from the new game is dropped.

### D2. Update the two tests that pinned skipping

`test_returns_alternating_no_done` and `test_returns_black_model_perspective`
encode the old behaviour. Their expected values change (4 → 2, 3 → -1). The
docstrings should say why, so the history stays legible.

## Risks / Trade-offs

- [Return magnitudes shift for every run] → accepted and intended; logs before and
  after are not comparable, noted in the proposal.
- [Value head may need to re-learn from scratch] → checkpoints from before this
  change are not reused for the baseline run.
- [Async env desynchronisation produces several opponent steps in a row] → covered
  by the "several opponent steps" scenario.
