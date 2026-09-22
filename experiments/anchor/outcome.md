# Prior-anchored self-play arms (change prior-anchored-selfplay) — 2026-09-22

Question: the policy improves for 200-600 guarded PPO updates from any prior, then loses to
the prior; and LONG256X's head-to-head gain over its prior did not show on the Stockfish
ladder. Does anchoring to the prior (loss term) or aiming the pool at the opponents the learner
loses to (PFSP) hold the gains? Constraint (owner): search-free everywhere.

Arms, both from slhi2_256, 600 updates, guards as LONG256X (target_kl 0.02, alarm 0.5 / 2 on
40 games), lr 1e-4 flat, pool 5 boards:
- KLPRIOR: `prior_kl_coef 0.1`, uniform pool of 4 snapshots.
- PFSP: `opponent_pool_sampling pfsp` (power 2, floor 0.05), `opponent_pool_size 12`, coef 0
  (drift still logged as mean_prior_kl).
Read alongside DECAY and NOPOOL (experiments/width/outcome.md), same budget and prior.

Predictions (before the run):
- Owner: (not given)
- Claude: KLPRIOR never trips the alarm and lands inside the prior's ladder interval
  (932, 773-1024); PFSP trips later than LONG256X's 300 but still trips. mean_prior_kl in the
  unanchored arms grows past 0.3 nats before the alarm.

Results: (pending — driver `run_anchor_arms.sh`, log `run_anchor_arms.log`)
