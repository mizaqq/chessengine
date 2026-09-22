## Context
`update_model` (src/model/training.py) forms the PPO loss per minibatch from stored observations,
legal masks, actions, old log-probs and advantages. The frozen prior is already loaded for the
prior match (`resolve_prior_match`) and the pool (`resolve_opponent_pool`) from `opponent_prior`.
`OpponentPool.sample` (src/training/opponent_pool.py) draws uniformly; `metrics.add_pool_result`
already receives every finished pool game's learner score and member name.

## Goals / Non-Goals
Goals: bound drift from the prior inside the loss; make the pool spend its games on the members
the learner currently loses to. Non-goals: exploiter agents, adaptive coefficient, annealing,
any search or engine-labelled target (owner constraint 2026-09-22).

## Decisions
- **KL direction** `KL(pi_theta || pi_prior)`, i.e. `sum_a pi_theta(a) (log pi_theta(a) - log
  pi_prior(a))` over legal moves. This is Ziegler's `log(pi/rho)` under `pi` and the form whose
  gradient discourages mass where the prior has none. Computed exactly (both distributions are
  small masked categoricals), not by a sample estimate. Extremes of `prior_kl_coef`: 0 = today's
  recipe; very large = the policy cannot leave the prior (imitation with extra steps). 0.1 for the
  first arm: at a drift of 0.3 nats it contributes 0.03 to a loss whose policy term is ~0.01-0.05.
- **Prior forward pass** without gradient on each minibatch's observations (256x10 net, ~1/3 of a
  training step's cost on MPS); the prior lives on the learner's device.
- **Which samples**: all learner plies in the minibatch, noisy boards included (the KL is about the
  policy at a state, not about the behaviour that reached it).
- **`mean_prior_kl` always logged** when a prior exists, so the baseline arms show the drift the
  guards permit; this is the number a future adaptive target would be set from.
- **PFSP weights** `(1 - w)^2 + 0.05`: AlphaStar's f_hard with p = 2 (their released pseudocode's
  "squared" weighting); the floor keeps every member's win rate fresh (f_hard(1) = 0 would never
  re-probe a beaten snapshot). Extremes: power 0 = uniform; large power = only the single hardest
  member; floor 0 = beaten members never return; floor large = uniform.
- **Win-rate estimate** EMA 0.1 of learner scores, initial 0.5, updated where `add_pool_result`
  is called (the pool receives the same event). A window of ~10 games; with 5 pool boards and
  ~2 games per board per update that follows a shift within a few updates.
- **Pool size 12 for the PFSP arm** so no snapshot of a 600-update run at every 50 is dropped
  (12 frozen 256x10 copies, ~0.6 GB unified memory). Under uniform sampling this would dilute the
  prior to 1/13 of games; under PFSP it costs nothing because beaten snapshots get the floor.
- **One mechanism per arm.** KLPRIOR keeps the uniform 4-snapshot pool; PFSP keeps coef 0.

## Risks / Trade-offs
- A constant prior-KL term caps how far the policy can improve past the prior (Kickstarting's
  plateau). Accepted: the problem today is falling below it.
- PFSP concentrates games on the prior when the learner is losing to it, which is also when the
  alarm is about to fire; the arm tells whether that pressure prevents the fall or merely delays it.
- EMA win rates on few games are noisy; the floor and power 2 (not higher) limit the swing.
