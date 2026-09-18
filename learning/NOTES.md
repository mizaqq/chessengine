# RL notes, by topic

Cumulative reference of what has been learned in this project, one topic per
section. Each session that touches a topic appends to it or corrects it. Sources
are in `RESOURCES.md`; demonstrated understanding is in `records/`.

## Return, value, advantage

- **Return** is what actually happened after a move: discounted sum of later
  rewards, with a value-head guess (the bootstrap) standing in for whatever lies
  beyond the rollout window.
- **Value** is what the model expected at the position where the move was chosen.
- **Advantage** = return − value. The policy gradient pushes a move up in
  proportion to its advantage. The absolute return level is irrelevant; only
  "better or worse than expected" moves the policy.
- Consequence: any constant offset shared by return and value cancels. The loss
  level is therefore not a measure of learning progress.

## Two-player games: the opponent is environment

- From one player's seat the opponent is part of the world. A transition runs from
  my decision to my next decision and includes the opponent's reply
  (Sutton & Barto, Section 1.5).
- Rewards that arrive between two of my decisions belong to the earlier one.
  Dropping them (as the old code did on opponent steps) makes the signal
  asymmetric: paid for captures, never charged for being captured.
- Discount once per own decision, not per ply.

## Reward shaping (potential-based)

- Dense hand-made rewards (material) speed up credit assignment but can change what
  is optimal. Ng, Harada & Russell (1999) prove that the only safe form is
  `F = gamma * Phi(next) − Phi(now)` for some potential Phi over states.
- Intuition: a loan. Material gains are paid out during the game and repaid at the
  terminal state (Phi(terminal) = 0), so shaping sums to zero over any complete
  game and only the outcome is net reward.
- If the *only* reward is potential-based, every policy is optimal (Remark 2): the
  shaping teaches nothing by itself; the outcome reward does all the teaching.
- In this repo: Phi = material from the mover's view, piece values 9/5/3/3/1,
  `shaping_scale` multiplies the loan, `shaping: none` is outcome-only.
- Material only matters while the game continues. A rook lost on the mating move
  is never charged: the terminal potential is zero and there is no future for the
  material to stand in for.

## Reading a shaped value head

- Under potential shaping the value head learns `V' = V − Phi` (Corollary 2). At a
  materially winning position it outputs a *negative* number.
- Before judging whether the model thinks it is winning, add material back:
  `value + material`.
- Empirically (500 updates): at scale 1.0 the head had not learned the offset
  (value moved *with* material, corr +0.46); at scale 0.2 the sign flipped
  (−0.28) but the slope was a tenth of theory. The head learns the loan slowly, and
  a smaller loan is less to learn.
- Fair comparison of two value heads needs the same fixed positions, not games
  each model played itself.

**Update 2026-09-17 (finishing curriculum).** Probed directly: the SL+PPO value
head predicted -2.6 (below the loss reward) for a side that could mate next move
in its own games, +1.3 on curated puzzles. Outcomes span -2..+2, so the number
came from the material term: the head had become a material counter. Two arms of
the finishing curriculum were flat with shaping on; the same arm with shaping off
moved the value on those positions to +0.1, broke the mate-in-one plateau
(0.57 -> 0.64) and doubled decisive games. Rule of thumb: dense shaping is a
bootstrap for a blank network; once a prior (here 300k human positions) supplies
the density, turn it off and let the outcome be the only reward (AlphaZero's
setting). See records 0013 and experiments/finishing-curriculum/outcome.md.

## Terminal states and bootstrapping

- On `done`, the future value is replaced by the terminal reward and the running
  potential is reset to zero, before the current move is processed. Order matters:
  reversed, material from the *next* game would leak into a checkmate.
- The bootstrap is the value head's guess for the last observed position. It is
  arbitrary in a hand trace and must be wiped by any earlier `done`.
- Quirk kept: the terminal reward enters as `gamma * terminal` at the ending move
  (1% at gamma 0.99).

## Variance and GAE (implemented Sep 2026)

- `return − value` is GAE with lambda = 1: unbiased given gamma, high variance,
  and every move in the window leans on one value-head guess at the horizon.
- The one-step surprise `r + gamma V(next) − V(now)` is lambda = 0: low variance,
  biased by every error in V. GAE sums surprises with decay lambda.
- Lambda adds bias only when V is wrong; gamma adds bias always. So lambda can sit
  around 0.95 while gamma stays 0.99 (Schulman et al. 2016, Section 3).
- GAE leans on the value head, so the shaped-head lesson above matters for it.
- Implemented as `compute_gae_for_model`: the same backward walk as the returns,
  accumulating surprises; lambda 1 reproduces return minus value exactly.
- Single-seed result (2026-09-16, arm B vs A): lambda 0.95 alone changed neither
  puzzle accuracy (0.32 both) nor the game outcome; its entropy curve collapsed
  and recovered. Not a conclusion, a reason for a second seed.

## Step size and PPO (implemented Sep 2026)

- Plain policy gradient buys one gradient step per rollout: after the step the
  policy has changed and the samples are stale.
- PPO looks at the ratio new/old probability of each taken move and clips the
  objective once the ratio leaves [1 − eps, 1 + eps] in the helpful direction, so
  several epochs over the same rollout are safe (Schulman et al. 2017, eq. 7).
- A2C is the special case: one epoch, one minibatch, no clip; at ratio 1 the
  gradients are identical.
- Diagnostics to log: clip fraction, approximate KL.
- Reuse is the point: with 64-ply rollouts A2C took 250 gradient steps in a run,
  PPO 4000, from the same moves. A2C's learning is bounded by gradient steps,
  which is why the old 15-ply A2C (1000 steps) beat 64-ply A2C (250 steps) on
  puzzles, 0.43 vs 0.32, and PPO matched the old A2C per ply (0.44).
- The ratio must depend only on parameters. BatchNorm in train mode made the
  re-evaluated probability depend on batch composition: clip fraction 0.52 on
  a fresh network with no parameter change. Fix: eval-mode BatchNorm during
  training, running statistics refreshed from the rollout once per update.
- Clip fraction is a learning-rate reading. At 3e-4 a fresh update left the band
  after 5 of 16 minibatches; at 1e-4 after 11; at 3e-5 never. Over a run both
  PPO arms sat near clip fraction 0.24, but KL halved at 1e-4 (0.05 vs 0.12) and
  the opening policy kept entropy 0.5-0.67 instead of 0.2-0.3, at almost no
  puzzle cost (0.42 vs 0.44). Practitioner target KL ~0.01-0.02 is still lower.
- None of it fixed the games: 200-plus-ply draws and 5-9% in-game mate rate in
  all four arms. The update rule was not the bottleneck for that symptom.

## Sampling versus greedy play

- A policy trained by sampling is calibrated for sampling. Its argmax is not a
  faithful summary of it.
- Greedy self-play with a weak policy loops into threefold repetition; sampled
  play explores into positions the policy handles well.
- Evaluation should sample at low temperature, not argmax.

## What the models actually know (Sep 2026)

- They know a few mating pictures with high confidence (queen mates, back-rank
  rook mates: top-1 with p 0.5–0.99) but miss ~96% of available mates-in-one.
- The draw rate is therefore a recognition problem, not a randomness problem.
- Mate-in-one rate adopted as the first evaluation metric (baseline 8/215).
- Candidate remedy: endgame / mate curriculum (start games near the goal), with
  supervised puzzle pre-training as the comparison, not the starting point.

## Start-state curriculum (Sep 2026)

- Sparse reward: exploration cost grows with the number of steps between reward
  and start. Resetting near the reward (Florensa 2017; Salimans & Chen 2018)
  collapses that distance, but only if a failed attempt also ends the episode.
  In chess a missed mate does not end anything, so puzzle episodes are cut after
  the mover's move (miss = terminal, reward 0, configurable penalty).
- Mix by board, not by reset: a one-ply episode next to a 200-ply game gives
  ~1% of plies to puzzles at "fraction 0.5". Fixed puzzle boards give the share
  the knob promises.
- Attempts per update is the quantity to watch, not updates.
- Held-out set vs training solve rate = the train/test gap; equal means the skill
  transfers across positions, not memorised.
- Puzzle accuracy (35%) leads the in-game rate (18%): distribution shift between
  tactical puzzle positions and self-play positions.
- Under potential shaping a mid-game start charges the start potential and never
  repays it; values shift, policy does not.

**Finishing boards (2026-09-17).** Reverse curriculum from human checkmates
(Florensa et al. 2017): start k plies before the mate, winner to move, cap
k + 20, depth rises with the success rate. 6 finishing boards for 300 updates did
not lift conversion from 10-20 plies out (held-out 0.05 -> 0.04-0.06) in any of
three arms; with shaping off the near-mate values and puzzle accuracy improved
instead. Lessons: (1) one ply before a 1500-rated human's mate the model found the
mate 38% of the time against 58% on curated puzzles -- the curriculum's own
positions are a third distribution; (2) an advance threshold must be set from the
measured starting rate, not assumed (0.6 was unreachable, the adaptive part never
ran); (3) more boards did not help because the ceiling was not sample count.

## Pretraining on human moves, then RL (Sep 2026)

- AlphaGo's first stage: predict the human move by cross-entropy, then start RL
  from those weights ("weights rho are initialized to the same values").
- Ours: 299k positions from Lichess 2013-12 (both Elo >= 1500, 8 positions per
  game so one game's shared outcome does not dominate), 3 epochs, held-out
  human-move top-1 0.34. Alone it reaches mating positions four times as often as
  the RL network (139 vs 31 chances in 40 games) but takes almost none (0.17 on
  mate-in-one puzzles).
- PPO 600 updates from those weights: mate-in-one 0.57 and mate-in-two 0.51
  held-out, against 0.44 / 0.35 for the same RL from scratch, with no plateau on
  the two-step reward: a decent prior turns a 1-in-900 search into a common
  event. PPO dials calm (KL 0.01-0.04).
- The cost: human-move agreement fell 0.34 -> 0.17 during RL. The outcome
  objective narrows the beam (AlphaGo p. 486: the SL policy searched better
  because humans "select a diverse beam"). Watch this number when RL follows
  pretraining; it is the same tension as SFT -> RLHF in language models.
- Mid-game start boards (4 of 8 game boards from human positions at plies
  20-60) ran alongside; their separate effect was not isolated.

## Sparse rewards plateau, then knee (Sep 2026)

- Mate-in-two boards pay only when the first move is the verified forcing move
  *and* the mate follows against the network's own defence: two right choices
  out of ~30 each, so a fresh policy succeeds about once in 900 attempts.
- Observed 2026-09-16: solved rate ~1% for 200 updates (held-out first-move
  accuracy flat at 0.08), then 0.12 at 300 and 0.23 at 600 (held-out 0.35).
  Mate-in-one caught up with the single-rung baseline despite half the attempts.
- Lesson: a curriculum rung that looks dead at a short budget may be one knee
  away. Either budget for the knee or give the first decision its own signal
  (partial credit for the forcing move). Salimans & Chen's quadratic-vs-
  exponential argument is exactly this: each extra decision before the reward
  multiplies the search.
- Ply cap at 120 (AlphaZero terminated over-long games as draws): 2.3x more
  finished games per update; training draws 0.98 (the cap ends most games);
  evaluation games shorter (186 plies) with fewer mate chances (31 vs ~100).

## Exploration ceiling: confident and wrong (Sep 2026)

Mate-in-one accuracy sat at 0.57 for 900 PPO updates from the pretrained
checkpoint. The histogram of probability mass on the mating move over 1,000
puzzles was bimodal: 52% above 0.8, 35% below 0.05. Policy gradient learns from
what it plays; on the confident-wrong third the mate is sampled less than once in
twenty and each hit is one clipped step, so those positions never get fixed. The
same network trained with cross-entropy on the mating moves reached 0.90 held-out
in 12 epochs, so capacity was not the limit. Remedies, in order: exploration noise
on curriculum boards (AlphaZero's root Dirichlet noise exists for exactly this;
PPO's ratio then uses the behaviour log-prob), search as the teacher (finds the
mate whatever the prior says), supervised targets where labels exist. Business
analogue: a recommender that never shows an item cannot learn that users want it.

**What actually worked (2026-09-17, evening).** Random Dirichlet noise in the
behaviour policy flattened the network on every board, even confined to one-move
puzzle boards (m1 0.653 -> 0.628): with normalised advantages and a clipped ratio,
random picks get reinforced about as often as punished. Label-guided behaviour,
the same mixture but with the demonstrator being the puzzle key move, the human
line on finishing boards, or a rules mate, did the opposite: 420 guided updates
took m1 0.653 -> 0.734, m2 0.571 -> 0.630, own-game mate positions 0.12 -> 0.23,
the confident-wrong share 30% -> 19%, the depth curriculum 6 -> 18, with entropy
unchanged. The label decides what gets tried; the outcome still decides what is
learned. Also settled the same evening: outcome-only from the pure SL prior beats
the shaped run at every checkpoint (CLEAN arm), so the base is unpolluted from
now on. See experiments/exploration-noise/outcome.md.

## Distribution shift in start states (Sep 2026)

- Same network: 0.43 on Lichess mate-in-one, 0.07 on mate-in-one positions from
  its own games. Own-game positions are unlike human ones (kings in the centre at
  move 11, 17 pieces on average).
- Training on harvested own positions raised own-position accuracy (0.07 -> 0.14),
  lowered Lichess (0.43 vs 0.49 for the control) and did not change in-game
  mate finding. A harvest is stale as soon as the network changes; the in-loop
  buffer was the principled version but was dropped when the A/B came back flat.
- Mate in one needs no planning but the network learns it as pictures; rare
  geometries stay hard. One-ply search makes it exact (AlphaZero's simulations).
- Continuing from a checkpoint (`init_from`) is what makes A/B arms comparable.

## One network, oriented input (Sep 2026)

- AlphaZero: board oriented to the current player, "my pieces" / "their pieces"
  planes, colour plane kept. OpenSpiel gives absolute planes but mover-relative
  move ids, so only the observation is transformed (`orient_black`: flip ranks,
  swap colour pairs, swap castling planes; files untouched).
- Effect on one seed: held-out mate-in-one 0.352 -> 0.427, reached 0.30 at ~60
  updates instead of 200; colour split of decisive games 6/24 -> 3/3. No wall-
  time gain: one backward on both sides' samples costs the same as two.
- Per-side bookkeeping (returns, bootstrap sign, terminal rewards) is unchanged;
  only the parameters are shared.

## Where the time goes (Sep 2026)

- Per update (12 boards x 15 steps, sync): forward 0.65 s, backward 1.02 s,
  environment 0.07 s. The network dominates; environment language is irrelevant.
- Sync vs async differ by ~5%.
- Two separate colour networks halve the data each pattern sees; batch norm on
  batches of ~6 is noisy and evaluation uses running stats (train/eval regime
  mismatch); the LR floor above the start makes the rate jump up at update 100.

## Experiment hygiene

- State the prediction before the run; compare afterwards; record the gap.
- One variable per experiment; the algorithm switch (A2C ↔ PPO) is only clean if
  the baseline is otherwise identical.
- Fixed seeds make runs reproducible (verified: identical numbers on rerun), and
  also make "rerun" silently replay the same game.
- Loss level is not the test. Behaviour is: outcomes, mate rate, value-head
  sanity.
- Split every counter by source (puzzle vs game) and by side (white vs black)
  before reading it; a 5x asymmetry hid inside an aggregate.
- "Small sample" is a hypothesis with a test: the binomial odds, then a second
  seed.

## Business analogues

- Reward attribution gaps: an agent credited for sales but not for churn caused
  between its decisions.
- Potential shaping: intermediate KPIs (clicks, engagement) must net to zero
  against the real outcome or the agent optimises the proxy.
- Argmax vs sampling: a recommender evaluated only on its top pick can look worse
  or better than the policy actually deployed.

## Self-imitation (Oh et al. 2018) — 2026-09-18

Replay of the learner's own finished episodes with Monte Carlo returns; only plies whose
return beats the current value estimate get gradient (the (R - V)+ clip), drawn in
proportion to that gap. No importance ratio. The mechanism switches itself off per ply
as the value head catches up (record 0015). In arms CURSIL-CURSIL3 it was neutral on the
puzzle dials and did not collapse entropy; its contribution to technique is confounded
with the curriculum (owner chose both at once).

## Start-state curricula without a ruler — 2026-09-18

Generated technique positions graded by a heuristic (weak king cornered / on the edge /
near the edge / anywhere) stand in for a distance-to-goal ruler (Florensa 2017). Two
failure modes met in one day: counting teacher-made wins toward advancement raced the
levels ahead of the skill (CURSIL); flagging "teacher acted" by the played move froze
the levels, because every win ends with the demonstrator's move by definition (CURSIL2).
The fix was to sample the guided mixture by component so "the teacher fired" is an
observed coin, not an inference from the action (CURSIL3).

## GAE lambda and long plans — 2026-09-18

With a terminal-only reward and plans of ten moves, lambda 0.95 shrinks the credit
reaching the setup moves to (gamma*lambda)^10 = 0.54 against gamma^10 = 0.90 at lambda
1. Three queen-only arms and the 600-update LONG2 run at lambda 1 raised decisive games,
in-game mate rate and own-game mate-in-one with no change to the PPO clip band or KL, so
lambda 1 is now the default. The bias-variance trade the GAE paper describes was, for
this problem, on the wrong side at 0.95: the value head's bias on long conversions was
worse than the Monte Carlo variance.

## Consolidation run LONG2 — 2026-09-18

600 updates on the full layout (pool, SIL, level ladder on four technique sets, lambda 1,
LR floor 1e-4) produced the first RL checkpoint at par with the supervised prior in the
win matrix (8-7) while beating every earlier RL checkpoint by a wide margin (36-4 over
M34), and the best value on all puzzle sets. Finishing conversion from 10-20 plies in
human middlegames did not move; that dial has now resisted fourteen arms.
