## Context

`TechniqueSampler` draws a set, a level in [0, current], and generates a position by
level; `report_label(label, success, clean)` advances levels on clean episodes only.
`FinishStart.depth` carries the level. Checkpoints are the model file only. Lambda is
a validated algorithm key (`gae_lambda`), preset 0.95 for PPO.

## Goals / Non-Goals

**Goals:** bucketed good-starts sampling with the paper's band and replay; curriculum
state with the checkpoint; dials; one arm at lambda 1.0.

**Non-Goals:** Brownian generation of new starts (our generator can target any bucket
directly, so the paper's SampleNearby is unnecessary); tablebase features; shaping.

## Decisions

- **Buckets**: 4 x 3 x 3 = 36 per set. Features are computed from the board, so the
  same function classifies generated and (if ever needed) harvested positions.
  `FinishStart.depth` carries a bucket id (0-35) in good-starts mode; the level ladder
  stays available as `technique_curriculum: levels`.
- **Generation by bucket**: choose the weak king square from the edge-distance class,
  the strong king from squares at the king-distance class (rejecting adjacency), the
  pieces so that the nearest is in the piece-distance class; then the usual rejections
  (legal, no check, no mate in one). Retry limit per bucket; an impossible bucket (none
  found in 200 tries) is marked infeasible and never drawn again.
- **Success estimate from training episodes**, as the paper does ("we use the
  trajectories collected by train_pol to estimate R"), clean episodes only; window 20 per
  bucket.
- **Weights**: unknown 1, in band 1, graduated `technique_replay_share` 0.2 (the paper
  appends N_old = 100 old starts to N_new = 200, a 1:2 ratio; we use 0.2 of weight),
  too hard `technique_probe_share` 0.1 (paper reaches hard starts by Brownian expansion
  from good ones; a probe share is the bucket analogue).
- **Persistence**: `curriculum.json` next to the model file with the per-bucket deques;
  `load_models` returns its path when present; `TechniqueSampler.load_state`.
- **Lambda**: arm sets `gae_lambda: 1.0`; default stays 0.95 until the arm reads.

## Hyperparameters

- `technique_r_min` / `technique_r_max` (0.1 / 0.9, paper A.1): the band. Narrow = only
  mid-difficulty starts, faster per start, more forgetting; wide = nearly uniform.
- `technique_bucket_window` (20): results per bucket before it is "known". Small =
  noisy rates, buckets flip in and out of the band; large = slow to notice progress.
- `technique_replay_share` (0.2): 0 = forget graduated starts (Florensa cites Held et al.
  on catastrophic forgetting); 1 = no curriculum pressure.
- `technique_probe_share` (0.1): 0 = hard buckets never enter; large = back to uniform.
- `gae_lambda` 1.0 vs 0.95: variance up, bias down; watch clip fraction and KL.

## Risks / Trade-offs

- Lambda 1 with advantage normalisation may over-weight rare wins on long boards;
  guarded by the PPO clip and the diagnostics.
- 36 buckets x 4 sets need many clean episodes to become known; the unknown weight of
  1 keeps them drawn, so early training is close to uniform, which is the right default.
- Two changes in one arm again; the owner chose both. If it works, an ablation with
  lambda 0.95 tells them apart.
