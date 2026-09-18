## 1. Technique curriculum

- [x] 1.1 `generate_position(material, rng, level)` with the four level constraints; `TechniqueSampler` levels per set, `report_label`, `current_level`, `levels()`; `FinishStart.depth` carries the level for technique starts; env calls `report_label` when the start has a label. Tests: level-0 start scenario, advance scenario, held-out stays level 3, reproducible.
- [x] 1.2 Metrics `technique_level_<set>` from the sampler each update (like `finish_depth`); config keys and resolver; test.

## 2. Self-imitation

- [x] 2.1 `src/training/sil.py`: `EpisodeAccumulator` (per-board learner plies, returns on done), `ReplayBuffer` (ring, priorities, proportional sampling with bias weights), `sil_loss(model, batch, value_weight)`. Tests: three-ply-win returns scenario, opponent plies excluded, only-better-than-expected-plies-train scenario, buffer ring drops oldest.
- [x] 2.2 Training loop: feed the accumulator from the rollout's `StepRecord`s and terminal rewards; after the PPO update run `sil_updates` minibatches; diagnostics into metrics; `sil_updates: 0` leaves the update sequence identical (test with a fixed seed).
- [x] 2.3 Config keys and resolver (`resolve_sil`), defaults per design; tests.

## 3. Run and record

- [x] 3.1a Arm CURSIL: 120 updates from POOL_LONG; recorded in `experiments/endgame-technique/outcome.md` (curriculum raced on teacher-made wins; held-out unchanged).
- [x] 3.1b Clean-episode advancement (rollout tracks demonstrator picks per board, reports `clean`), 0.7 over 40, `sil_positive_share`; arm CURSIL2 from POOL_LONG, same budget; outcome.
- [ ] 3.2 RESOURCES: SIL appendix hyperparameters; NOTES sections (self-imitation, start-state curricula); learning record if demonstrated.

## 4. Comprehension check with owner

- [x] 4.1 Trace by hand (CLAUDE.md): given a 3-ply technique episode with rewards and current values pasted, compute R per ply, the SIL priorities, and which plies get policy gradient; then run the test from 2.1 and compare.
