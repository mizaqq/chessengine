## Why

The Stockfish ladder (experiments/elo/outcome.md, 2026-09-22) showed that LONG256X beats its
supervised prior 92-62 head to head yet scores no better against an outside opponent (+1 =1 -38
vs +4 =1 -35 at UCI_Elo 1320). Self-play gains were partly opponent-specific, and every RL
continuation from any prior rises for 200-600 updates and then loses to the prior once game
entropy nears 0.2. The current guards bound the step from the *previous* policy (target-KL) and
detect the fall (alarm); nothing bounds drift from the *prior* and nothing keeps the learner
training against the opponents it has started losing to. Owner (2026-09-22): the model must stay
search-free everywhere (no MCTS, no engine-labelled targets), so the remaining levers are the RL
ones the AlphaStar paper used for the same symptom.

## What Changes

- **Prior-KL term in the PPO loss** (`prior_kl_coef`, default 0 = off): each minibatch adds
  `prior_kl_coef * mean KL(pi_theta || pi_prior)` over legal moves, where `pi_prior` is the frozen
  network in `opponent_prior` run on the same observations. Logged every update as
  `mean_prior_kl` (also when the coefficient is 0, as a drift dial).
- **Prioritised opponent sampling** (`opponent_pool_sampling: uniform | pfsp`, default uniform):
  pool boards draw the opponent with probability proportional to `(1 - w)^p + floor`, where `w`
  is the learner's smoothed win rate against that member (AlphaStar's f_hard), `p =
  pfsp_power` (default 2) and `floor = pfsp_min_weight` (default 0.05). Members the learner
  beats every time are rarely drawn; members it loses to, including the prior, dominate.
- **Experiment** `experiments/anchor/run_anchor_arms.sh`: arms KLPRIOR (coef 0.1) and PFSP
  (pfsp, pool size 12 so every snapshot of a 600-run stays available), both from slhi2_256, 600
  updates, guarded as LONG256X and the recipe arms; 200-game matrix vs SLHI2_256 / LONG256X and
  the Stockfish ladder.

## Capabilities

### New Capabilities
- (none)

### Modified Capabilities
- `policy-update`: prior-KL penalty requirement and `mean_prior_kl` diagnostic.
- `opponent-pool`: pool boards may draw opponents by prioritised fictitious self-play weights.

## Alternatives considered

- **Adaptive coefficient toward a target prior-KL** (Ziegler et al. 2019, log-space proportional
  controller on beta). Self-tuning but adds a target we have no measurement for yet; the run logs
  `mean_prior_kl` so a target can be chosen next time.
- **Annealed coefficient 1 -> 0** (Kickstarting, Schmitt et al. 2018: constant lambda plateaus at
  41.2, linear anneal reaches 59-61 on DMLab-30). The anneal exists to let the student surpass the
  teacher; our failure is the opposite (falling below it), so a constant term is the direct test.
- **L2 toward the prior's weights** (Dohare et al. 2024 fixed a PPO rise-then-collapse with L2).
  Parameter-space proxy for the same idea; cheaper, less targeted. Kept as a later arm.
- **Full league with exploiters** (Vinyals et al. 2019: +153/+284 Elo from exploiters). Too heavy
  for one laptop; PFSP over main agents is the component with the anti-forgetting evidence
  (FSP 1143 < pFSP 1273 < SP 1519 < pFSP+SP 1540).
- **Learning-rate decay / prior-only pool** (arms DECAY, NOPOOL, running 2026-09-22). Their result
  is read alongside these arms; Andrychowicz et al. 2021 rate lr decay "of secondary importance".

## How we will know it worked

Per arm (600 updates, guarded): `alarm_stop` update (LONG256X: 300; absent = held), `prior_score`
trajectory, `mean_prior_kl` (KLPRIOR should hold it flat; the others show how far the recipe
drifts), game entropy, puzzles, then the 200-game matrix vs SLHI2_256 and LONG256X and the
Stockfish 1320 ladder (SLHI2_256 +4 =1 -35; LONG256X +1 =1 -38). Success = an arm that both
beats LONG256X in the matrix and matches or exceeds the prior on the ladder.
Owner's prediction: (to be filled in chat before the run).
Claude's prediction: KLPRIOR does not trip the alarm by 600 and lands within the prior's ladder
interval; PFSP trips later than 300 but still trips.

## What to understand

- Why a trust region on the previous policy cannot stop slow drift from the prior, and what a
  KL term to a frozen reference adds (Ziegler 2019; Andrychowicz 2021 §5 on redundancy).
- The AlphaStar human-data ablation: init only 1020 vs + supervised KL 1400.
- Fictitious self-play vs prioritised FSP: why uniform sampling over recent snapshots forgets,
  and what `(1 - w)^p` does at w = 0, 0.5, 1.
- Why "beats the prior head to head" and "stronger" are different claims (Balduzzi 2019).
