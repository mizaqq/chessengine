# RL Resources

Curated, high-trust sources for this project. Every RL design decision should cite
an entry here. Add a one-line annotation to each link: what it covers and when to
reach for it. Prune anything that turns out shallow or wrong.

## Knowledge

<!-- Papers, official docs, reference implementations. Annotate every entry. -->

- [Paper: "Proximal Policy Optimization Algorithms" — Schulman, Wolski, Dhariwal, Radford, Klimov (2017)](https://arxiv.org/abs/1707.06347)
  Defines the clipped surrogate objective (eq. 7), the combined loss with value and
  entropy terms (eq. 9, c1=0.5, c2=0.01), Algorithm 1, and the truncated GAE used in
  practice (eq. 11-12). Hyperparameters: Table 3 (Atari: T=128, 3 epochs, minibatch
  32, eps=0.1, gamma=0.99, lambda=0.95) and Table 5 (MuJoCo: T=2048, 10 epochs,
  minibatch 64, eps=0.2). Use for: why clipping, what epsilon/epochs/minibatches do.
- [Paper: "High-Dimensional Continuous Control Using Generalized Advantage Estimation" — Schulman, Moritz, Levine, Jordan, Abbeel (2016)](https://arxiv.org/abs/1506.02438)
  Section 3: TD residual delta_t = r_t + gamma V(s_{t+1}) - V(s_t) (eq. 10), GAE as
  sum of (gamma*lambda)^l delta_{t+l} (eq. 16), lambda=0 is one-step TD (eq. 17),
  lambda=1 is Monte Carlo returns minus baseline (eq. 18). Key sentence: gamma
  introduces bias regardless of value accuracy; lambda introduces bias only when V is
  inaccurate, so best lambda is usually lower than best gamma. Use for: choosing and
  explaining lambda, bias/variance intuition.
- [Blog: "The 37 Implementation Details of Proximal Policy Optimization" — Huang, Dossa, Raffin, Kanervisto, Wang (ICLR Blog Track 2022)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
  The 13 core details: vectorized rollouts, GAE with bootstrapping and
  returns = advantages + values, shuffled minibatches, per-minibatch advantage
  normalization, clipped value loss (optional), global grad-norm clip 0.5, Adam
  eps 1e-5, linear LR annealing, debug variables (clip fraction, approx KL). Also
  invalid-action masking via -inf logits. Use for: implementation checklist and
  what to log.

- [Book: *Reinforcement Learning: An Introduction* — Sutton & Barto, Section 1.5 tic-tac-toe example](http://incompleteideas.net/book/the-book-2nd.html)
  The learning player has "no model of its opponent of any kind"; the opponent's
  reply is simply part of how the board changes between the player's own moves, and
  the TD backup V(s) <- V(s) + a[V(s') - V(s)] runs between states where the player
  is to move. Use for: why opponent moves are environment from one player's view,
  and why rewards between two own moves belong to the earlier one. (Verified via the
  archive.org full text; the incompleteideas.net site had a certificate error.)

- [Paper: "Policy invariance under reward transformations: Theory and application to reward shaping" — Ng, Harada & Russell (ICML 1999)](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
  Theorem 1: F(s,a,s') = gamma*Phi(s') - Phi(s) is necessary and sufficient for the
  shaped MDP to keep the original optimal policies. Corollary 2: Q' = Q - Phi,
  V' = V - Phi. Remark 1: near-optimal policies preserved. Remark 2: a purely
  potential-based reward makes every policy optimal. Use for: material shaping
  design, why shaping nets to zero over a game, reading the shaped value head.

- [Paper: "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" — Silver et al. (2017)](https://arxiv.org/abs/1712.01815)
  Abstract: superhuman play "given no domain knowledge except the game rules",
  tabula rasa self-play, no handcrafted evaluation. Grounds `shaping: none` as the
  outcome-only reference point. The exact reward formulation (game outcome only) is
  in the paper body; verify there before quoting numbers.

- [Paper: "Mastering Chess and Shogi by Self-Play..." — Silver et al. (2017), methods p.13](https://arxiv.org/abs/1712.01815)
  "The board is oriented to the perspective of the current player. The M feature
  planes are composed of binary feature planes indicating the presence of the
  player's pieces ... and a second set of planes indicating the presence of the
  opponent's pieces", plus constant planes for colour, move count, castling,
  repetitions, no-progress count. Use for: one network for both colours;
  `src/model/orientation.py`. OpenSpiel's move ids are already mover-relative
  (verified 2026-09-16), its observation planes are absolute.

- [Paper: "Mastering the game of Go without human knowledge" — Silver et al. (Nature 550, 2017), unformatted manuscript](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf)
  Read 2026-09-16 (pp. 3-6, 24-27). Search: each edge stores {N, W, Q, P};
  select a = argmax Q + U with U = c_puct * P * sqrt(sum_b N(s,b)) / (1 + N(s,a));
  leaf expanded and evaluated by the network once, edges initialised N=W=Q=0,
  P=p_a; backup N += 1, W += v, Q = W/N; play pi(a) proportional to N^(1/tau),
  tau = 1 for the first 30 moves then -> 0; root prior noise P = (1-eps) p +
  eps eta, eta ~ Dir(0.03), eps = 0.25; subtree reused after the played move.
  Training: loss l = (z - v)^2 - pi^T log p + c||theta||^2 (eq. 1), c = 1e-4,
  cross-entropy and MSE weighted equally because rewards are unit scaled;
  1,600 simulations per move; resignation when root and best child value below
  a threshold tuned to <5% false positives. MCTS is "a powerful policy
  improvement operator" and self-play with search "a powerful policy evaluation
  operator" (p. 3). Raw network without search reached 3,055 Elo vs 5,185 with
  search (p. 12). Use for: the search-in-the-loop change; AlphaZero (above)
  inherits these parameters except where it says otherwise (800 simulations,
  Dir alpha 0.3 for chess, lr 0.2 dropped 3x, max game length -> draw).

- [Paper: "Mastering the game of Go with deep neural networks and tree search" — Silver et al. (Nature 529, 2016)](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)
  Read 2026-09-16 (pp. 484-485, Methods "Policy network: classification",
  "Policy network: reinforcement learning", "Value network: regression"). SL
  policy: 13 layers, trained on 30M positions from 160k KGS games (6-9 dan) to
  predict the human move by stochastic gradient ascent on log p(a|s); 57.0%
  held-out accuracy (44.4% prior state of the art); minibatch 16, step 0.003
  halved every 80M steps, 340M steps. RL stage: "weights rho are initialized to
  the same values" as the SL network, REINFORCE with a value baseline against a
  pool of earlier policies ("randomizing from a pool of opponents ... stabilizes
  training by preventing overfitting to the current policy"); the RL policy won
  >80% of games against the SL policy and 85% against Pachi with no search.
  Value network: regressing on all positions of KGS games overfit ("successive
  positions are strongly correlated ... regression target is shared for the
  entire game"; test MSE 0.37 vs 0.19 train), fixed by one position per
  self-play game. Caution (p. 486): the SL policy worked better *inside search*
  than the stronger RL policy, "presumably because humans select a diverse beam
  of promising moves". Use for: supervised pretraining on Lichess games before
  PPO (change `human-pretraining`), sampling few positions per game, and the
  opponent-pool idea for later.
- [Data: Lichess standard rated games database (CC0)](https://database.lichess.org/)
  Monthly PGN .zst files; counts at `standard/counts.txt`. 2013-06: 293,459
  games, 33 MB; 2013-12: 578,262 games, 92 MB; 2014-06: 961,868 games, 182 MB.
  Headers carry WhiteElo/BlackElo/Result. Use for: human-move pretraining and
  mid-game start positions.

- [Paper: "Reverse Curriculum Generation for Reinforcement Learning" — Florensa et al. (CoRL 2017)](https://arxiv.org/abs/1707.05300)
  Abstract: for goal-oriented sparse-reward tasks, start the agent near the goal and
  widen the start-state distribution as it succeeds; needs only one goal state, no
  demonstrations or shaped reward; the curriculum adapts to performance. Use for:
  why puzzle start positions help, and the later adaptive `puzzle_fraction`.

- [Paper: "Learning Montezuma's Revenge from a Single Demonstration" — Salimans & Chen (2018)](https://arxiv.org/abs/1812.03381)
  Abstract: resetting episodes to states along a demonstration turns exploration cost
  that scales exponentially in the number of steps between rewards into roughly
  quadratic. Use for: the argument that a mate hundreds of moves from the opening
  is unreachable by random play, one move away it is not.

- [Data: Lichess puzzle database (CC0)](https://database.lichess.org/#puzzles)
  6.1M rated puzzles as CSV with FEN, UCI moves, rating and themes. FEN is the
  position *before* the opponent's move; the solution starts at move two. Theme
  `mateIn1` gave 898,211 puzzles on 2026-09-16; we use 20,000 train / 2,000
  held-out, converted by `scripts/prepare_puzzles.py`.

- [Paper: "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (MuZero) — Schrittwieser et al. (2020)](https://arxiv.org/abs/1911.08265)
  Abstract read 2026-09-17: learns a model that predicts only "the reward, the
  action-selection policy, and the value function" and plans with tree search
  inside that model, "without any knowledge of their underlying dynamics";
  matched AlphaZero on chess/shogi/Go. Use for: the search-with-learned-model
  option when a perfect simulator is not available.

- [Paper: "Muesli: Combining Improvements in Policy Optimization" — Hessel et al. (2021)](https://arxiv.org/abs/2104.06159)
  Abstract read 2026-09-17: regularised policy optimisation with model learning
  as an auxiliary objective; matches MuZero on Atari "without using deep search";
  "acts directly with a policy network". Use for: the model-free-at-play middle
  road between PPO and MuZero.

- [Paper: "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games" (NFSP) — Heinrich & Silver (2016)](https://arxiv.org/abs/1603.01121)
  Abstract read 2026-09-17: plain self-play RL "diverged" in Leduc poker while
  NFSP (best-response network + average-policy network) approached a Nash
  equilibrium; no game model needed. Use for: roadmap 05 opponent sampling.

- [Paper: "Mastering Diverse Domains through World Models" (DreamerV3) — Hafner et al. (2023)](https://arxiv.org/abs/2301.04104)
  Abstract read 2026-09-17: "learns a model of the environment and improves its
  behavior by imagining future scenarios"; one configuration across 150+ tasks.
  Use for: the learned-world-model family, contrast with MuZero (imagined
  rollouts for policy gradient vs tree search).

- [Paper: "Reverse Curriculum Generation for Reinforcement Learning" — Florensa et al. (2017)](https://arxiv.org/abs/1707.05300)
  Abstract read 2026-09-17: the agent "is trained in reverse, gradually learning
  to reach the goal from a set of start states increasingly far from the goal";
  the start distribution "adapts to the agent's performance"; needs only "a
  single state in which the task is achieved". Use for: the finishing boards
  (start k plies before a human checkmate, depth rises with the success rate);
  the puzzle boards are the k = 0 special case.
  Read 2026-09-18 (pp. 3-6, A.1, B.1): Algorithm 1 keeps a list of "good starts"
  S_i^0 = {s0 : R_min < R(pi_i, s0) < R_max}, success estimated from the training
  trajectories themselves; new starts by Brownian rollouts from good starts
  (SampleNearby, T_B = 50); N_old = 100 old good starts replayed with N_new = 200
  each iteration ("the replay buffer is an important feature to avoid catastrophic
  forgetting"); R_min = 0.1, R_max = 0.9, gamma 0.998. B.1: adding a distance-to-goal
  shaping reward "does not actually improve training" and hurt the key task. Use for:
  the good-starts bucket curriculum on technique boards (change `good-starts-lambda`).

- [Paper: "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (AlphaZero preprint) — Silver et al. (2017)](https://arxiv.org/abs/1712.01815)
  Read 2026-09-17 (pp. 3-4, 13-15): loss (z - v)^2 - pi^T log p + c||theta||^2
  (eq. 1); outcome -1/0/+1 only; "the noise that is added to the prior policy to
  ensure exploration ... is scaled in proportion to the typical number of legal
  moves" (p.4); Methods p.14: "Dirichlet noise Dir(alpha) was added to the prior
  probabilities in the root node ... alpha = {0.3, 0.15, 0.03} for chess, shogi
  and Go"; evaluation "selects moves greedily" (p.15); 800 simulations per move;
  chess input 119 planes (8-step history), policy 8x8x73. Use for: exploration
  noise (alpha 0.3 for chess) and the later search change.

- [Paper: "Self-Imitation Learning" — Oh, Guo, Singh & Lee (ICML 2018)](https://arxiv.org/abs/1806.05635)
  Read 2026-09-17 (pp. 1-8): replay buffer of (s, a, R) with Monte-Carlo return R;
  loss L_policy = -log pi(a|s) (R - V(s))_+ and L_value = 1/2 ||(R - V(s))_+||^2
  (eq. 2-3): only state-actions whose past return beat the current value estimate
  get gradient; sampled with priority (R - V)_+; no importance ratio, so it works
  from any past behaviour. M = 4 SIL updates per A2C update (Atari), 10 per PPO
  iteration (MuJoCo). Helped most on sparse/delayed-reward tasks; caveat (5.4):
  "sometimes gets stuck at a sub-optimal policy" when exploitation is excessive,
  fixed by fewer SIL updates or a smaller weight later. Appendix A read 2026-09-18:
  PPO+SIL (MuJoCo) M = 10 SIL updates per batch, SIL batch 512, loss weight 0.1,
  value weight 0.01 or 0.05, buffer 50,000, prioritisation exponent 0.6 or 1.0, bias
  correction 0.1; A2C+SIL (Atari) M = 4, batch 512, loss weight 1, value weight 0.01,
  buffer 1e5. Use for: replaying our own won games on finishing boards (the
  demonstrator becomes the past self); change `self-imitation-curriculum`.

- [Paper: "Emergent Complexity via Multi-Agent Competition" — Bansal et al. (ICLR 2018)](https://arxiv.org/abs/1710.03748)
  Read 2026-09-17 (pp. 4-8): "training agents against the most recent opponent
  leads to imbalance ... the other agent is unable to recover" (4.2); sample the
  opponent from Uniform(delta*v, v) over checkpoint history; delta = 0 (whole
  history) best for Ant, delta = 0.5 for Humanoid, delta = 1 (latest only) worst
  in both (Table 1). Evaluation: win-rate matrix between agents trained with each
  delta. Also 5.5.2: long self-play "over-fitting to the behavior of the opponent",
  cured by an ensemble pool. Use for: opponent pool from saved checkpoints, and
  the checkpoint-vs-checkpoint win matrix as the "cycling" probe.

- [Paper: "Open-ended Learning in Symmetric Zero-sum Games" — Balduzzi et al. (ICML 2019)](https://arxiv.org/abs/1901.08106)
  Abstract and §1 read 2026-09-19: self-play produces a sequence of stronger agents
  only when "the game is approximately transitive"; in nontransitive games there are
  "strategic cycles" and "we want agents to increase in strength, but against whom is
  unclear". Proposes response oracles against a population (PSRO_rN) instead of the
  latest self. Use for: reading the blunder audit (LONG3 beats LONG2 while hanging a
  piece every ten plies: beating the current self is not the same as getting better).

- [Blog: "AlphaStar: Grandmaster level in StarCraft II using multi-agent RL" — Vinyals et al. (DeepMind, 2019; Nature paper behind it)](https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)
  Read 2026-09-19 (league section): "playing to win is insufficient"; exploiter
  agents "whose sole purpose is to expose weaknesses of the main agent" are added to
  the league so that valid strategies are not forgotten and the main agent becomes
  robust. Use for: the one-ply rules punisher as a hand-made exploiter of gifts
  (hanging pieces, allowed mates); change `punish-gifts`.

- [Paper: "Re-evaluating Evaluation" — Balduzzi, Tuyls, Perolat & Graepel (NeurIPS 2018)](https://arxiv.org/abs/1806.02643)
  Abstract read 2026-09-21: agent-vs-agent and agent-vs-task evaluations are biased by
  redundant or weak opponents and easy tasks; Nash averaging "automatically adapts to
  redundancies in evaluation data, so that results are not biased by the incorporation of
  easy tasks or weak agents". Use for: reading win matrices with many same-lineage
  checkpoints (LONG3 best on puzzle dials, 32-2 loser in games); workshop 3 §2.

- [Paper: "Deep Reinforcement Learning that Matters" — Henderson et al. (AAAI 2018)](https://arxiv.org/abs/1709.06560)
  Abstract read 2026-09-21: "Without significance metrics and tighter standardization of
  experimental reporting, it is difficult to determine whether improvements over the prior
  state-of-the-art are meaningful"; seeds, hyperparameters and evaluation choices change
  conclusions. Use for: the 0.387 (training-time probe) vs 0.232 (40-game eval) trap and our
  60-game matrices' noise band; workshop 3 §6.

- [Paper: "Training Larger Networks for Deep Reinforcement Learning" — Ota et al. (2021)](https://arxiv.org/abs/2102.07920)
  Abstract read 2026-09-21: naively larger networks "do not lead to performance
  improvement" in deep RL because of training instability; they use DenseNet-style width,
  decoupled representation learning and distributed training. Use for: why our 128 -> 192
  gain in RL (from a strong SL prior, PPO trust band, entropy 0.37, KL 0.02) is not the
  default outcome; workshop 3 §4.

- [Paper: "Scaling Laws for Imitation Learning in Single-Agent Games" — Tuyls et al. (2023)](https://arxiv.org/abs/2307.09423)
  Abstract read 2026-09-21: "IL loss and mean return scale smoothly with the compute
  budget (FLOPs) and are strongly correlated, resulting in power laws"; carefully scaling
  model and data brings NLP-like gains to behaviour cloning in games. Use for: our prior
  curve 0.341 (299k) -> 0.405 (1.43M) -> 0.432 (2.85M) and the quantity-vs-quality question
  (high-Elo set); workshop 3 §5.

- [Docs: OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
  Read 2026-09-21: "If the mean KL-divergence of the new policy from the old grows beyond a
  threshold, we stop taking gradient steps"; clipping alone "is still possible to end up with
  a new policy which is too far from the old policy"; default `target_kl` 0.01 (their
  implementation stops an epoch at 1.5x target). Use for: target-KL early stopping in
  `update_model` after LONG256B drifted (KL 0.04 -> 0.06, clip 0.16 -> 0.26); our healthy band
  has been 0.02-0.04, so the threshold is set at 0.03; change `ppo-guards`.

- [Source: Stockfish `src/search.h`, struct Skill (master, read 2026-09-22)](https://github.com/official-stockfish/Stockfish/blob/master/src/search.h)
  "Skill structure is used to implement strength limit. If we have a UCI_Elo, we convert it
  to an appropriate skill level, anchored to the Stash engine. This method is based on a fit
  of the Elo results for games played between Stockfish at various skill levels and various
  versions of the Stash engine. Skill 0 .. 19 now covers CCRL Blitz Elo from 1320 to 3190,
  approximately." `UCI_Elo` range 1320-3190 (`LowestElo`/`HighestElo`); at level L the engine
  picks from a MultiPV search when depth == 1 + L (`time_to_pick`). Use for: the Elo ladder
  evaluation (`scripts/eval_elo.py`): the scale is CCRL blitz, the floor is 1320, and a network
  below the floor is rated by a logistic fit against several levels; change `elo-eval`.

- [Paper: "Amortized Planning with Large-Scale Transformers: A Case Study on Chess" (a.k.a. "Grandmaster-Level Chess Without Search") — Ruoss et al. (DeepMind, 2024)](https://arxiv.org/abs/2402.04494)
  Read 2026-09-22 (HTML v2). 10M Lichess games, 530M boards, Stockfish 16 at 50 ms per
  board/state-action, 15.3B action-values; win% = 100/(1+exp(-0.00368208*cp)) binned into K=128
  with HL-Gauss; 9M/136M/270M transformers; 270M: tournament Elo 2299, Lichess 2299 vs bots,
  2895 vs humans, 95.4% puzzles; Leela policy net 1 node 2292 / 88.6%, AlphaZero policy net 1777
  / 56.1%. Target ablation: action-value 63.0% > state-value 58.5% > behavioural cloning 56.7%.
  Use for: engine-value targets for the value head, the "no search" Elo ceiling, data-vs-model
  scaling; `learning/research/2026-09-22-improving-the-model.md` §1-2.

- [Paper: "Aligning Superhuman AI with Human Behavior: Chess as a Model System" (Maia) — McIlroy-Young, Sen, Kleinberg, Anderson (KDD 2020)](https://arxiv.org/abs/2006.01855)
  Read 2026-09-22 (PDF). Nine 6x64 residual nets, one per rating bin 1100-1900, "12 million games"
  each; first 10 ply and <30 s moves dropped; top-1 "over 52%" vs Leela max 46%, Stockfish 33-41%.
  Use for: the human-move top-1 ceiling and why accuracy is not strength; research §1.2.

- [Paper: "Maia-2: A Unified Model for Human-AI Alignment in Chess" — Tang, McIlroy-Young, Anderson (NeurIPS 2024)](https://arxiv.org/abs/2409.20553)
  Read 2026-09-22 (PDF/HTML). 169M games, 9.1B positions, 23.3M params; rating embeddings injected
  into attention queries (skill-aware attention); macro top-1 53.25%, Masters 53.87%, vs Maia-1
  51.39%, Leela 44.53%, Stockfish 15 40.43%. Use for: rating-conditioned training on all Lichess
  data, the ~53% imitation plateau; research §1.3, §7.

- [Blog: "Transformer Progress" — Daniel Monroe, Leela Chess Zero (2024-02-28)](https://lczero.org/blog/2024/02/transformer-progress/)
  Read 2026-09-22. Raw-policy Elo relative to T78 (194.5M params, 12.45 GFLOPs): BT1 +13, BT2
  +123, BT3 +179, BT4 +270 (191.3M, 7.6 GFLOPs, 15 layers, 1024 emb, 32 heads); 64 square
  tokens, 8 one-hot piece planes for current + 7 past plies, trainable 64x64xh attention bias
  ("play as if 50% larger"), smolgen. Companion post "How well do Lc0 networks compare to the
  greatest transformer network from DeepMind" (2024-02) gives the 1-node test protocol and the
  caveat that value-based puzzle tests run ~150 Elo above policy tests. Use for: architecture
  options once data stops being the constraint; research §1.4, §6.

- [Blog: "Win-Draw-Loss evaluation" (2020-04) and "Lc0 v0.25 has been released" (2020-05) — Leela Chess Zero](https://lczero.org/blog/2020/04/wdl-head/)
  Read 2026-09-22. WDL head since July 2019 because a scalar score "doesn't account for draws";
  moves-left head (v0.25) predicts remaining game length, used to "take the shorter route towards
  winning". No Elo numbers given for either head. Use for: 3-way value head; research §6.

- [Paper: "Stop Regressing: Training Value Functions via Classification for Scalable Deep RL" — Farebrother et al. (2024)](https://arxiv.org/abs/2403.03950)
  Read 2026-09-22 (PDF). HL-Gauss: Gaussian target projected onto histogram bins, "we set
  sigma/bin = 0.75 ... approximately 6 locations"; two-hot ignores ordinal structure; on Ruoss's
  chess setup HL-Gauss "closes the performance gap with ... AlphaZero with 400 MCTS simulations"
  (70% of the gap); MSE plateaus with width, HL-Gauss keeps scaling. Use for: replacing the
  scalar MSE value head; research §2.1.

- [Paper: "Fine-Tuning Language Models from Human Preferences" — Ziegler et al. (OpenAI, 2019)](https://arxiv.org/abs/1909.08593)
  Read 2026-09-22 (PDF). R(x,y) = r(x,y) - beta log(pi(y|x)/rho(y|x)); "prevents the policy
  from moving too far from the range where r is valid"; beta constant or a log-space proportional
  controller to a target KL(pi, rho); without it the policy degenerates (appendix samples). Use
  for: KL-to-prior in PPO as a reward term; research §3.1.

- [Paper: "Grandmaster level in StarCraft II using multi-agent reinforcement learning" — Vinyals et al. (Nature 2019), unformatted manuscript](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)
  Read 2026-09-22 (Methods + Fig. 3). "continually minimise the KL divergence between the
  supervised and current policy"; z = first 20 buildings/units, pseudo-rewards by edit/Hamming
  distance, each active with probability 25%; ablation Elo: no human data 149, supervised 936,
  human init 1020, + supervised KL 1400, + statistics 1540. League: main agents 35% self-play,
  50% PFSP over all past players, 15% forgotten players/past exploiters; PFSP weight f_hard(x) =
  (1-x)^p; FSP 1143 < pFSP 1273 < SP 1519 < pFSP+SP 1540; + main exploiters 1693, + league
  exploiters 1824; "Naive self-play has high Elo, but is more forgetful." Use for: prior-KL term,
  PFSP pool weighting, the forgotten-players slot; research §3.2, §4.1.

- [Paper: "Kickstarting Deep Reinforcement Learning" — Schmitt et al. (DeepMind, 2018)](https://arxiv.org/abs/1803.03835)
  Read 2026-09-22 (PDF). l_kick = l_RL + lambda_k H(pi_T || pi_S), lambda_k -> 0 after T0;
  constant lambda "ultimately plateaus" (41.2), linear 1->0 over 1-2B frames 59-61, PBT
  competitive; "almost 10x fewer steps ... surpassing its final performance by 42%". Use for:
  annealing schedule of a prior-distillation term; research §3.3.

- [Paper: "Implementation Matters in Deep Policy Gradients" — Engstrom et al. (ICLR 2020)](https://arxiv.org/abs/2005.12729)
  Read 2026-09-22 (PDF). Value clipping, reward scaling, orthogonal init, Adam LR annealing:
  "reward normalization, Adam annealing, and network initialization each significantly impact the
  rewards landscape"; PPO-NoClip matches PPO (Humanoid 831 vs 806); the clipped objective's trust
  region depends on the optimiser and "we can end up moving arbitrarily far". Use for: why the
  clip/KL guard bounds one step but not drift; research §5.1.

- [Paper: "What Matters for On-Policy Deep Actor-Critic Methods? A Large-Scale Study" — Andrychowicz et al. (ICLR 2021)](https://arxiv.org/abs/2006.05990)
  Read 2026-09-22 (PDF; 250k agents, MuJoCo). LR decay "may slightly improve performance but is
  of secondary importance" (4/5 tasks, +15% only on Ant); per-minibatch advantage normalisation
  "seems not to affect the performance too much"; value-loss clipping "hurts"; entropy/KL
  regularisers not helpful there, conjectured redundant with the PPO trust region; clip 0.25,
  GAE 0.9. Use for: expectations for the LR-decay arm; research §5.2.

- [Paper: "Phasic Policy Gradient" — Cobbe, Hilton, Klimov, Schulman (ICML 2021)](https://arxiv.org/abs/2009.04416)
  Read 2026-09-22 (PDF). Policy phase (PPO) then auxiliary phase L_joint = L_aux + beta_clone
  KL(pi_old || pi_theta) on an auxiliary value head; value tolerates more sample reuse than
  policy; "relatively infrequent auxiliary phases are critical". Use for: separating value and
  policy training if value interference is diagnosed; research §5.3.

- [Paper: "The Primacy Bias in Deep Reinforcement Learning" — Nikishin et al. (ICML 2022)](https://arxiv.org/abs/2205.07802)
  Read 2026-09-22 (PDF). "a tendency to overfit initial experiences"; heavy priming (1e5 updates
  on 100 transitions) is unrecoverable; remedy "periodically re-initialize the parameters of its
  last few layers while preserving the replay buffer"; SPR+resets IQM 0.478 on Atari-100k; the
  on-policy analogue is ITER (full reset + distillation from the previous generation). Use for:
  reading the rise-then-decay pattern as overfitting to early self-play; research §5.4.

- [Paper: "Loss of plasticity in deep continual learning" — Dohare et al. (Nature 632, 2024)](https://www.nature.com/articles/s41586-024-07711-7)
  Read 2026-09-22 (PDF). PPO on an ant with friction changing every 2M steps "fails
  catastrophically"; tuned Adam helps, "adding continual backpropagation or L2 regularization is
  necessary to perform well indefinitely"; even stationary ant PPO "increased for about 3 million
  steps but then collapsed"; "Adam, Dropout and normalization actually increased loss of
  plasticity"; L2 and Shrink-and-Perturb reduce it. Use for: L2 (to zero or to the prior) as a
  stabiliser with the matching failure story; research §5.5.

- [Paper: "Understanding Plasticity in Neural Networks" — Lyle et al. (ICML 2023)](https://proceedings.mlr.press/v202/lyle23b/lyle23b.pdf)
  Read 2026-09-22 (PDF). Plasticity loss "deeply connected to changes in the curvature of the
  loss landscape ... often occurs in the absence of saturated units"; layer normalisation is the
  best-performing intervention (DQN on ALE). Use for: LayerNorm as a cheap check; research §5.5.

- [Paper: "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning" (PSRO) — Lanctot et al. (NeurIPS 2017)](https://arxiv.org/abs/1711.00832)
  Abstract read 2026-09-22: "approximate best responses to mixtures of policies" with
  "empirical game-theoretic analysis to compute meta-strategies"; generalises iterated best
  response, double oracle and fictitious play; independent RL "overfit[s] to the other agents'
  policies". Use for: vocabulary behind the pool (meta-strategy = sampling weights); research §4.2.

- [Paper: "Fictitious Self-Play in Extensive-Form Games" — Heinrich, Lanctot, Silver (ICML 2015)](http://proceedings.mlr.press/v37/heinrich15.pdf)
  Read 2026-09-22 (intro). Best response to "opponents' average strategies"; the average
  converges to Nash in two-player zero-sum; FSP is the sample-based version. Use for: what a
  uniform pool over all past snapshots is, and why AlphaStar weighted it instead; research §4.2.

## Wisdom (Communities)

<!-- Places to test understanding against practitioners. Optional. -->

## Gaps

Sources still needed, ordered by roadmap priority:

- ~~Engine-based evaluation of learned policies~~ — Stockfish UCI_Elo ladder, 2026-09-22 (see Stockfish `search.h` above)
- Reward decay / shaping schedule in sparse two-player games (roadmap/04_rewards_curriculum.md §1)
