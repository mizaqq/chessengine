# 0016 — Regularising toward a reference policy caps improvement; it is a bet, not a free lunch

Date: 2026-09-22. Change: prior-anchored-selfplay. Type: explain back / challenge.

## What the owner said

Told that the new PPO term is `prior_kl_coef * KL(pi_theta || pi_prior)` against the frozen
supervised network, the owner objected: "we dont need those to agree, but the learner should be
better than prior."

## Why this is demonstrated understanding

The objection names the exact trade-off of every reference-policy regulariser (RLHF's beta-KL,
AlphaStar's supervised-KL, Kickstarting's distillation term): minimising the KL pulls toward
the reference, so a constant coefficient puts a ceiling on how far the learner can improve past
it. Kickstarting reports the plateau directly (constant lambda 41.2 vs annealed 59-61). The term
is justified only when unanchored drift costs more than that ceiling, which is what the
Stockfish ladder suggested (LONG256X 92-62 over its prior head to head, no better than the
prior against Stockfish).

Resolution the owner accepted: the gradient is a tug of war per position; where self-play
has real evidence the reward gradient outweighs 0.1 x KL and the learner departs from the
prior, where it has only noise the KL term holds it in place. The PFSP arm is the alternative
with no ceiling: anchoring through opponents rather than through the loss.

## Where else this shape appears

Fine-tuning a language model with RL (reward minus beta KL to the pretrained model); any
"stay close to the incumbent process" constraint on an optimiser in production, where the
incumbent is known-good and the optimiser's signal is noisy.
