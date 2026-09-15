## Context

See proposal.md. Today the environment emits `reward = material(after) -
material(before)` per step, `StepRecord` stores it as `reward_white`, and the
backward returns walk adds it only on own steps. Terminal outcomes arrive separately
through `info["game_results"]` and are converted to `terminal_r_white/black`.

## Goals / Non-Goals

**Goals:** one formula for the per-move reward with a cited invariance guarantee;
opponent replies handled by construction; outcome-only mode as a switch; tests that
can be traced by hand.

**Non-Goals:** changing terminal reward values, piece values, gamma, the network, or
the update step. GAE arrives in `add-ppo-gae` and reuses this walk.

## Decisions

### D1. Environment reports a potential, not a difference

`EnvStep.material` is the white-view material of the returned observation (the
position the next mover sees). The per-step difference is removed; `EnvStep.reward`
is dropped from the environments (terminal rewards already travel via `info`).

Why: a potential is a property of a state, which is what the theorem needs. A
difference is a property of a transition and has to be re-attributed. Both vector
envs already compute the material; they just stop subtracting.

Alternative: keep `reward` and compute potentials from stored observations in the
training loop. Rejected: duplicates the piece-value logic in a second place.

### D2. Per-model backward walk carries the next own potential

`StepRecord.potential_white` holds the material of the pre-move observation for
that step (white view). The walk receives `final_material`, the white-view material
of the observation returned after the last step.

```
R        = bootstrap                          # [num_envs]
phi_next = sign * final_material              # sign = +1 white, -1 black
for i in reversed(steps):
    done_f   = step.done
    R        = R * (1 - done_f) + terminal_r  # cut chain at game end
    phi_next = phi_next * (1 - done_f)        # terminal potential is 0
    if own step (per env, via where):
        phi = sign * step.potential_white
        F   = scale * (gamma * phi_next - phi)   # 0 when shaping == none
        R   = F + gamma * R
        phi_next = phi
    returns[i] = R
```

Ordering matters: `phi_next` is zeroed by `done` before the own step uses it, so a
mating move sees a terminal potential of 0; potentials from the new game that
follows never reach the old game because the reset happens at the `done` step
during the backward walk.

Alternative: shape per environment step for both players and discount per step.
Rejected: it changes what `gamma` means (per ply instead of per own decision) and
the telescoping no longer lines up with the per-model bootstrap.

### D3. `shaping: none` is `scale = 0`

Implemented as a resolved scale of 0 in config parsing, so the walk has one code
path. `shaping_scale` must be >= 0; a negative scale would invert the potential and
is almost certainly a mistake.

### D4. What the knobs do

- **shaping_scale** (>= 0). Multiplies the material loan. 0 is outcome-only reward:
  the theoretically cleanest objective and the sparsest signal. 1 matches today's
  magnitudes, where a queen is worth 4.5 game outcomes. Very large values make the
  value head spend its capacity modelling material rather than winning chances,
  and the policy gradient becomes dominated by material swings within the window.
- **gamma** (unchanged) appears inside the shaping term. Setting gamma below 1
  makes holding material slightly costly over time (the `gamma * Phi(s')` is smaller
  than `Phi(s)` for equal material), which is the discounted-MDP version of the
  invariance result and is expected.

## Risks / Trade-offs

- [Shaped return for a winning move can be negative, e.g. -7 in the mating-while-
  ahead scenario] → expected under `V' = V - Phi`; the advantage is measured against
  a value head that also carries `-Phi`. Covered as a comprehension question.
- [Value head must re-learn; old checkpoints meaningless] → baseline run starts
  from scratch; documented in the proposal.
- [Desynchronised async envs produce several opponent steps in a row] → the walk
  only acts on own steps, so intermediate opponent steps need no special handling;
  a test with two consecutive opponent steps is included.
- [Env tests asserting on `reward`] → updated to assert on `material`.

## Migration Plan

Single change on `main`. Rollback is `git revert`; there is no way to reproduce the
old skipping behaviour through config, by design.
