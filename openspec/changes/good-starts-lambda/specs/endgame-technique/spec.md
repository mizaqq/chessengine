## REMOVED Requirements

### Requirement: Technique level curriculum
**Reason**: replaced by the good-starts bucket curriculum (Florensa et al. 2017 §4.1); the
fixed four-rung ladder stalled the single pieces at one rung for 300 updates.
**Migration**: `technique_curriculum: levels` keeps the old behaviour for comparison arms.

## ADDED Requirements

### Requirement: Good-starts curriculum
When the curriculum mode is `good_starts`, every generated start SHALL be assigned a
bucket from board geometry: the weak king's distance to the nearest edge (0, 1, 2 or 3),
the distance between the kings (2, 3, or 4 and more) and the nearest strong piece's
distance to the weak king (1-2, 3-4, or 5 and more). For each material set and bucket
the sampler SHALL keep the clean success rate over the last `technique_bucket_window`
clean episodes. A bucket with fewer results than the window SHALL count as unknown.
Starts SHALL be drawn from buckets in proportion to a weight: 1 for unknown buckets and
for buckets whose rate lies within [`technique_r_min`, `technique_r_max`];
`technique_replay_share` for buckets above `technique_r_max`; `technique_probe_share`
for buckets below `technique_r_min`. The drawn bucket's geometry SHALL be honoured by
the generator. Held-out evaluation SHALL keep unconstrained placement.

#### Scenario: band selects the learnable starts
- **WHEN** set `Q` has bucket A at rate 0.95, bucket B at rate 0.5, bucket C at rate 0.02
  and bucket D unknown, with replay share 0.2 and probe share 0.1
- **THEN** the draw weights are A 0.2, B 1, C 0.1, D 1

#### Scenario: geometry honoured
- **WHEN** the drawn bucket is (edge distance 0, king distance 2, piece distance 1-2)
- **THEN** the generated position has the weak king on an edge square, the strong king
  exactly two squares away and the nearest strong piece within two squares of the weak king

#### Scenario: teacher-made wins do not count
- **WHEN** a `Q` board is won after the demonstrator fired in that episode
- **THEN** no bucket's window is updated

### Requirement: Curriculum state persists with the checkpoint
Saving a checkpoint SHALL also write the curriculum state (per set and bucket, the
recent clean results) next to it, and `init_from` SHALL reload that state when present so
a continued run resumes where the curriculum stood. A checkpoint without the state file
SHALL start the curriculum fresh.

#### Scenario: resume
- **WHEN** a run ends with bucket B of `Q` at rate 0.5 over a full window and a new run
  starts from that checkpoint
- **THEN** the new run's first summary reports the same bucket rate before any new result

### Requirement: Curriculum dials
The summary SHALL report per set `technique_good_buckets_<set>` (buckets in the band),
`technique_graduated_<set>` (buckets above the band) and `technique_known_buckets_<set>`.

#### Scenario: counts
- **WHEN** set `Q` has 36 buckets of which 6 are known, 4 in the band and 2 above it
- **THEN** the summary shows known 6, good 4, graduated 2
