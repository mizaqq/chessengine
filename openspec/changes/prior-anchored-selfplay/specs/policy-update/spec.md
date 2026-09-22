## ADDED Requirements

### Requirement: Prior-KL penalty
When `prior_kl_coef` is greater than 0 and a frozen prior network is supplied, every minibatch
step of the policy update SHALL add `prior_kl_coef * mean_i KL(pi_theta(.|s_i) || pi_prior(.|s_i))`
to the loss, where both distributions are the masked softmax over the legal moves of `s_i` and
the mean runs over the minibatch. The prior SHALL receive no gradient. The update summary SHALL
report `mean_prior_kl` (the unweighted mean KL over the update's minibatches) whenever a prior is
supplied, including when `prior_kl_coef` is 0. `prior_kl_coef` SHALL be a non-negative number
(default 0).

#### Scenario: penalty pulls toward the prior
- **WHEN** a position has three legal moves, the policy gives them (0.7, 0.2, 0.1), the prior gives (0.2, 0.7, 0.1), `prior_kl_coef` is 0.1 and every advantage is 0
- **THEN** the minibatch loss includes 0.1 x 0.7 ln(0.7/0.2) + 0.2 ln(0.2/0.7) + 0.1 ln(0.1/0.1) = 0.1 x 0.6264 = 0.0626 (plus the unchanged value and entropy terms), and one optimizer step raises the policy's probability of the second move

#### Scenario: coefficient zero
- **WHEN** `prior_kl_coef` is 0 and a prior is supplied
- **THEN** the loss is unchanged and `mean_prior_kl` is still reported

#### Scenario: no prior
- **WHEN** no prior network is supplied
- **THEN** the update is unchanged and `mean_prior_kl` is null
