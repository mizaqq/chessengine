## Purpose

Defines how a collected rollout is turned into parameter updates for each player's
model, which algorithm is used, and what is logged so the update can be judged.

## ADDED Requirements

### Requirement: Algorithm is selected by config
The system SHALL read `algorithm` from the training config, accepting `a2c`
(default) or `ppo`, and SHALL reject any other value with an error listing the
accepted values.

#### Scenario: default is a2c
- **WHEN** the config has no `algorithm` key
- **THEN** training runs with the `a2c` behaviour

#### Scenario: unknown algorithm
- **WHEN** the config sets `algorithm: sac`
- **THEN** training refuses to start and the error lists `a2c` and `ppo`

### Requirement: Rollout stores what re-evaluation needs
For every step and environment, the rollout SHALL retain the observation, legal
action mask, sampled action, the acting model's log-probability of that action and
its value estimate at collection time, none of them connected to a computation
graph. Collection SHALL NOT perform gradient computation.

#### Scenario: stored values are detached
- **WHEN** a rollout of any length is collected
- **THEN** no stored tensor requires grad and the acting model's parameters have no
  accumulated gradient afterwards

### Requirement: A2C preset reproduces current gradient
Under `a2c`, the update SHALL perform one gradient step per model per rollout using
the whole rollout, with policy loss `-(log_prob * advantage).mean()`, value loss
`value_coef * (value - target)^2.mean()`, and the normalised-entropy bonus. The
gradient SHALL equal that of the previous implementation for the same parameters,
rollout and advantages.

#### Scenario: gradient equivalence
- **WHEN** the a2c update runs on a fixed rollout with a fixed seed
- **THEN** every parameter gradient matches a directly computed REINFORCE-with-
  baseline gradient on the same data within 1e-5

#### Scenario: model with no steps in the rollout
- **WHEN** a rollout contains no steps for one model
- **THEN** that model's parameters are unchanged and its reported step count is 0

### Requirement: PPO clipped surrogate objective
Under `ppo`, for each model the update SHALL run `ppo_epochs` passes over that
model's own steps, each pass split into `ppo_minibatches` shuffled minibatches. For
each minibatch it SHALL recompute log-probabilities and values with the current
parameters, form the ratio `exp(new_log_prob - old_log_prob)`, and use the policy
loss `-mean(min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A))` where `eps` is
`clip_epsilon`.

#### Scenario: first minibatch has ratio one
- **WHEN** the first minibatch of the first epoch is processed
- **THEN** every ratio is 1 within 1e-6, the policy loss equals `-mean(A)`, and the
  clip fraction is 0

#### Scenario: positive advantage with ratio above the clip range
- **WHEN** a sample has advantage +2, old log-prob log(0.2), new log-prob log(0.3)
  and `clip_epsilon` 0.2
- **THEN** the ratio is 1.5, the clipped term 1.2 * 2 = 2.4 is smaller than the
  unclipped 3.0, the sample contributes -2.4 to the sum and contributes no gradient
  to the policy parameters

#### Scenario: negative advantage with ratio below the clip range
- **WHEN** a sample has advantage -1, ratio 0.5 and `clip_epsilon` 0.2
- **THEN** the unclipped term is -0.5 and the clipped term is -0.8; the minimum -0.8
  is used and the sample contributes no policy gradient

#### Scenario: epochs times minibatches gradient steps
- **WHEN** a model has 180 own steps, `ppo_epochs` is 4 and `ppo_minibatches` is 4
- **THEN** the model receives 16 optimizer steps with minibatches of 45 samples

### Requirement: Advantage normalisation per minibatch
Under `ppo`, when `normalize_advantage` is true (default), advantages within each
minibatch SHALL be shifted to mean 0 and scaled to standard deviation 1 (with a 1e-8
guard) before the policy loss. Value targets SHALL NOT be normalised.

#### Scenario: single-sample minibatch
- **WHEN** a minibatch contains one sample
- **THEN** its normalised advantage is 0 and no NaN is produced

### Requirement: PPO value loss and entropy
Under `ppo`, the total loss per minibatch SHALL be
`policy_loss + value_coef * value_loss - entropy_coef * normalized_entropy` with
value loss the mean squared error between recomputed values and value targets, and
gradients clipped to `grad_clip` before each optimizer step.

#### Scenario: value coefficient scales value gradient
- **WHEN** `value_coef` is doubled and everything else is fixed
- **THEN** the gradient on value-head-only parameters doubles

### Requirement: PPO diagnostics are logged
Under `ppo`, the system SHALL record for each update the mean clip fraction (share
of samples whose ratio left [1 - eps, 1 + eps]) and the mean approximate KL
`mean(old_log_prob - new_log_prob)` across all minibatches of both models, and
include them in the windowed log summary.

#### Scenario: diagnostics present in logs
- **WHEN** training runs under `ppo` for at least one log interval
- **THEN** each log entry contains `clip_fraction` and `approx_kl` numeric fields

#### Scenario: diagnostics absent under a2c
- **WHEN** training runs under `a2c`
- **THEN** log entries do not contain PPO diagnostic fields, or contain them as null

### Requirement: PPO hyperparameters are configurable with validation
The system SHALL read `clip_epsilon` (default 0.2), `ppo_epochs` (default 4),
`ppo_minibatches` (default 4), `value_coef` (default 0.5 under ppo, 1.0 under
a2c), `normalize_advantage` (default true) and `gae_lambda` (default 0.95 under
ppo). It SHALL reject `clip_epsilon` <= 0, `ppo_epochs` < 1, or `ppo_minibatches` < 1.

#### Scenario: minibatches larger than samples
- **WHEN** a model has fewer own steps than `ppo_minibatches`
- **THEN** the update uses as many minibatches as there are samples, each of size 1,
  and does not fail
