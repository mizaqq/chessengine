## ADDED Requirements

### Requirement: Target-KL early stopping
When `target_kl` is set, before each minibatch optimizer step the update SHALL compute the
mean approximate KL `mean(old_log_prob - new_log_prob)` over the minibatch's clean samples
(samples whose stored log-probability is the network's, not a behaviour mixture's). If it
exceeds `target_kl`, no further optimizer steps SHALL be taken in this update. The update
summary SHALL report `mean_kl_stop` (1 for an update cut short, else 0, averaged) and
`mean_optimizer_steps`. `target_kl` SHALL be a positive number or null (off; default 0.03).

#### Scenario: guard fires
- **WHEN** ppo_epochs 4, ppo_minibatches 4, and the third minibatch's KL is 0.05 with target_kl 0.03
- **THEN** exactly 2 optimizer steps are taken and kl_stop is 1

#### Scenario: guard silent
- **WHEN** every minibatch's KL stays below target_kl
- **THEN** all 16 steps are taken and kl_stop is 0

### Requirement: Training stop on prior alarm
When a prior match is evaluated and `stop_below_prior` is set, training SHALL stop once
`prior_score` has been below `stop_below_prior` at `stop_patience` consecutive evaluations.
On stopping, the models SHALL be restored to their weights at the most recent evaluation whose
`prior_score` was at or above the threshold (or the initial weights if none), and the log
SHALL record `alarm_stop` with the update number. Defaults: 0.5 and 2.

#### Scenario: alarm trips
- **WHEN** prior_score is 0.6, 0.35, 0.38 at evaluations 50, 150, 250
- **THEN** training stops after update 250 with the weights from update 50

#### Scenario: single dip
- **WHEN** prior_score is 0.6, 0.35, 0.55
- **THEN** training continues
