## 1. Good-starts curriculum

- [x] 1.1 `bucket_of(board, strong)`, `generate_position_in_bucket(material, bucket, rng)`; `GoodStartsCurriculum(sets, r_min, r_max, window, replay_share, probe_share)` with `weights(label)`, `draw(label)`, `report(label, bucket, success)`, `state()` / `load_state()`; `TechniqueSampler(mode=good_starts|levels|off)`. Tests: band-selects scenario, geometry-honoured scenario, infeasible bucket retired, teacher-made wins ignored.
- [x] 1.2 Dials `technique_known_buckets_<set>`, `technique_good_buckets_<set>`, `technique_graduated_<set>`; config keys and resolver (`technique_curriculum: good_starts` default). Tests.
- [x] 1.3 Curriculum state saved as `curriculum.json` with the checkpoint and reloaded by `init_from`. Test: resume scenario.

## 2. Run and record

- [x] 2.1b Follow-ups (owner: "okay agreed on both"): arm LAMBDA (levels + lambda 1.0, queen-only, from CURSIL_LONG; isolates lambda) and arm GSL2 (good starts with warm start + lambda 1.0); outcome.
- [x] 2.1 Arm GSL: 120 updates from the strongest technique checkpoint, `gae_lambda: 1.0`, good-starts on; games; compare with CURSIL_LONG / QONLY in `experiments/endgame-technique/outcome.md`.
- [ ] 2.2 RESOURCES (Florensa Algorithm 1, A.1 values, B.1 caution on distance shaping); NOTES; learning record if demonstrated.

## 3. Comprehension check with owner

- [ ] 3.1 Map paper to code (CLAUDE.md): with Florensa's Algorithm 1 and `select(starts, rews, R_min, R_max)` beside `GoodStartsCurriculum.weights`, point at what implements the band, what stands in for `starts_old`, and name the deviation (probe share instead of Brownian expansion).
