## Why

After 480 technique updates (arms CURSIL3, CURSIL_LONG) the pairs climbed their level
ladder but the single pieces stalled: queen v king at level 1 winning a third of its
clean games for 300 updates, rook v king at level 0. Two suspects remain once the
queen-only arm (QONLY) has ruled sample rate in or out: the win's credit reaching the
approach moves, and a level ladder whose rungs are too far apart. The owner chose both
levers for one arm ("okay so 1 and 2 is next arm").

## What Changes

- **Credit reaching the approach moves.** The arm runs with `gae_lambda` 1.0 instead of
  the PPO preset 0.95, so the advantage of every own move is the full discounted
  outcome minus the value estimate rather than a lambda-shrunken share; ten moves before
  the mate the credit is gamma^10 instead of (gamma*lambda)^10 (0.90 vs 0.54). No code
  change: lambda is already a validated config key (advantage-estimation spec).
- **Good-starts curriculum (Florensa et al. 2017, Algorithm 1 and §4.1).** The four fixed
  levels are replaced by feature buckets over generated positions: weak king's distance
  to the nearest edge (0-3), king-to-king distance (2, 3, 4+), nearest strong piece's
  distance to the weak king (1-2, 3-4, 5+). Each (set, bucket) keeps a windowed clean
  success rate. Starts are drawn from buckets whose rate lies in [R_min, R_max] =
  [0.1, 0.9] ("good starts": reached sometimes but not always) or is still unknown;
  graduated buckets (rate > R_max) keep a replay share so they are not forgotten (the
  paper's N_old); too-hard buckets (rate < R_min) keep a small probe share so they can
  become good starts. Level 3 / unconstrained held-out evaluation is unchanged.
- **Curriculum state travels with the checkpoint**: the bucket statistics are saved
  next to the model and reloaded by `init_from` (CURSIL_LONG restarted every set at
  level 0 because they were not).

## Capabilities

### Modified Capabilities
- `endgame-technique`: the level curriculum requirement is replaced by the good-starts
  bucket curriculum; a new requirement for persisted curriculum state.

### New Capabilities
- none.

## Alternatives considered

- **Keep the level ladder, lower the threshold to 0.5.** Cheaper, but the ladder's rungs
  stay fixed and coarse; Florensa's band adapts per start region and drops solved starts
  automatically. Rejected in favour of the paper's rule.
- **Distance reward shaping** (potential on the weak king's distance to the edge, Ng
  1999). Florensa Appendix B.1 found distance shaping "does not actually improve
  training" and hurt the key task; deferred, with that caution recorded.
- **Tablebase-defined distance to mate** as the bucket feature: rejected by the owner
  (perfect environment model). The features here are board geometry only.
- **Search** (AlphaZero): ruled out by the owner for this project phase.

## How we will know it worked

Arm GSL: 120 updates from the strongest technique checkpoint (CURSIL_LONG, or QONLY if it
wins the queen comparison), default layout, `gae_lambda: 1.0`, good-starts curriculum.
- `technique_rate_Q / R / RR / QR` held-out at unconstrained placement: from 0.06 / 0.06 /
  0.18 / 0.16 (CURSIL_LONG end).
- New dials: `technique_good_buckets_<set>` (in band), `technique_graduated_<set>`
  (above R_max), `technique_clean_success_rate_<set>` on drawn starts.
- Guards: m1-m4, endgame-mate, prior_score, game entropy; PPO clip fraction and approx KL
  (lambda 1 raises advantage variance).

Owner's prediction: _____ (not given).
Claude's prediction: with lambda 1 the queen's clean success on its good starts climbs
past 0.5 within 120 updates and at least three buckets graduate; held-out queen reaches
0.10-0.20; rook still near zero held-out; clip fraction rises but stays under 0.3.

## What to understand

- What lambda does to credit assignment: the trade between variance and bias, and why
  a ten-move plan with a terminal-only reward wants lambda near 1.
- Florensa's rule in one sentence: train from where you sometimes win, and why "sometimes"
  is the whole point (always = nothing to learn, never = no signal).
- Why graduated starts must stay in the mix (forgetting) and too-hard ones must be probed
  (the band moves only if hard buckets are sometimes seen).

## Impact

`src/envs/technique.py` (bucket features, bucketed generation, `GoodStartsCurriculum`
replacing the level ladder inside `TechniqueSampler`), `src/training/metrics.py` (bucket
dials), `src/entrypoints/train.py` and `src/configs/train_default.yaml` (keys
`technique_curriculum: good_starts | levels | off`, `technique_r_min`, `technique_r_max`,
`technique_bucket_window`, `technique_replay_share`, `technique_probe_share`; the arm sets
`gae_lambda: 1.0`), checkpoint save/load of `curriculum.json`. Tests for each.
