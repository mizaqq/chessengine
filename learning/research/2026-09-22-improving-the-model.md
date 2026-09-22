# How could this model get materially stronger without search? (research, 2026-09-22)

Question: the 256x10 residual policy/value net (SL prior on 5.7M high-Elo Lichess positions,
top-1 0.471; PPO self-play on top) scores +1 =1 -38 against Stockfish `UCI_Elo` 1320 and, from
any prior, improves for 200-600 PPO updates and then decays against the prior once game entropy
falls to ~0.2. What do primary sources say about making a *search-free* chess policy stronger,
and about stopping the improve-then-decay pattern?

Method: every claim below is from a fetched primary source (arXiv/ICML/Nature PDF, official
lc0 blog, official Lichess database page, local `stockfish bench`). Quotes are short and verbatim.
Anything that is my reading rather than the source's is marked **[inference]**. Where a number
could not be found in a primary source I say so.

---

## 1. Searchless / amortised chess policies

### 1.1 Ruoss et al. 2024, "Grandmaster-Level Chess Without Search" / "Amortized Planning with Large-Scale Transformers" (DeepMind)

Source: https://arxiv.org/abs/2402.04494 (HTML v2 read).

What they did. Downloaded "10 million games" from Lichess (Feb 2023), extracted "530M board
states", and annotated them with "Stockfish 16 ... using a time limit of 50ms per board" for
state-values and "50ms per state-action pair" for action-values, "corresponding to a Lichess
blitz Elo of 2713 for our action-value oracle". Largest set: "15.3B action-values". Values are
converted with `win% = 100/(1+exp(-0.00368208 * centipawns))` and the interval 0-100% is
"divide[d] ... uniformly into K bins", "K=128" (ablated 16-256). The network is a classifier
over bins trained with cross-entropy; move choice at play time is "argmax E[Z]" of the predicted
action-value distribution. Models: 9M, 136M, 270M parameter decoder-only transformers, "batch
size of 4096 ... 10 million steps ... Adam ... learning rate of 1e-4" (about 2.67 epochs).

Numbers (Table 1, no search, 1 forward pass per legal move):

| model | tournament Elo (internal) | Lichess Elo vs bots | vs humans | 10k-puzzle acc |
|---|---|---|---|---|
| 9M | 2025 | 2054 | - | 88.9% |
| 136M | 2259 | 2156 | - | 94.5% |
| 270M | 2299 | 2299 | 2895 | 95.4% |
| AlphaZero policy net, 1 node | 1777 | - | - | 56.1% |
| Leela policy net, 1 node | 2292 | - | - | 88.6% |
| Leela 400 MCTS sims | - | - | - | 99.6% |

The human/bot gap: "(i) humans tend to resign when our bot is in an overwhelmingly winning
position but bots do not". Known weakness of no search: "our agent, despite being in a position
of overwhelming win percentage, sometimes fails to take the (virtually) guaranteed win and
might draw or even end up losing since small chances of a mistake accumulate with longer games".

Prediction target ablation (same data, same model): action-value 63.0% action accuracy /
Kendall tau 0.259 vs state-value 58.5% / 0.215 vs behavioural cloning (predict Stockfish's move)
56.7% / 0.116; "Behavioral cloning falls significantly short of state-value learning". Puzzle
accuracy in the ablation: action-value 83.3% vs BC 65.7%.

Data-vs-model ablation: "For small training set size (10K games, left panel) larger
architectures (>= 7M) start to overfit. This effect disappears as the dataset size is increased
to 100K ... and 1M games"; "larger training set size improves the test loss when keeping the
model size constant"; "Across all metrics, increasing model size consistently improves the
scores". With 1M games (about 53M positions) puzzle accuracy already reaches 83-84%.

Meaning for this project. The single most important fact: **Leela's policy head alone, one
node, sits at tournament Elo ~2292 and 88.6% puzzle accuracy; AlphaZero's raw policy at 1777.**
A search-free network in the low-2000s is therefore an established fact, not a hope; the question
is only the training signal and the data volume. Ruoss's targets are *engine values*, not human
moves, and the target choice alone is worth ~7 points of action accuracy and ~18 points of puzzle
accuracy at fixed data. **[inference]** Our SL prior (behavioural cloning on human moves, 5.7M
positions) is on the weakest of their three targets with ~1% of their smallest ablation dataset
that stops small models overfitting; a 256x10 ResNet (roughly 12M parameters) is in the range
their 10K-game ablation says overfits without more data.

Cost/risk. Their annotation budget is enormous: 15.3B action-values x 50 ms = 7.65e8 CPU-seconds
(~24 CPU-years) **[inference from their numbers]**. See section 2 for what a laptop can afford.

### 1.2 Maia (McIlroy-Young, Sen, Kleinberg, Anderson, KDD 2020)

Source: https://arxiv.org/abs/2006.01855 (PDF read).

Data: "9 training sets and 9 validation sets, one for each rating range between 1100 and 1900,
with 12 million games and 120,000 games each"; they "discard the first 10 ply" and "any move
where the player had less than 30 seconds". Rating conditioning is by *separate models*: "We then
trained separate models, one for each rating bin". Architecture: "We chose to use 6 blocks and 64
filters"; a deeper "20 blocks and 320 filters" variant "yielded a small performance boost at a
significant" cost (per the appendix). Accuracy: "Maia's highest accuracy is over 52%", every bin
"consistently above 50%"; Leela versions max out at 46% and Stockfish at 33-41% on human moves.
Telling sentence: "12 million games is enough for Leela to go from random moving to a rating of
3000 (superhuman performance), and therefore far outstrips the usual training size for a single
Leela network." No playing-strength Elo for Maia is given in the paper (not found).

### 1.3 Maia-2 (Tang, McIlroy-Young, Anderson, NeurIPS 2024)

Source: https://arxiv.org/abs/2409.20553 (PDF and HTML read).

Data: "169M games (9.1B positions)" from Lichess "May 2018 and Nov 2023", rapid games, with
over-sampling of uneven matchups. Conditioning: a "skill-aware attention mechanism"; the two
players' ratings are looked up in a categorical embedding table `E = [e(0,1000], e(1000,1100],
..., e(2000,+inf)]` and "we inject skill level embeddings into queries" of the attention blocks
that sit on top of "a standard residual network tower". Size: "a one-for-all model with 23.3M
parameters" vs Maia-1's 9 x 10.3M experts. Accuracy (Table 1, macro-average): Maia-2 53.25%,
best Maia-1 51.39%, Leela 44.53%, Stockfish 15 40.43%; on Master (>2000) players Maia-2 53.87%.
"Maia-2 outperforms all other models by almost 2 full percentage points in overall accuracy".

Meaning for this project. **Human-move top-1 saturates around 52-54% even with 9.1B positions.**
Our 0.471 on >=2000 players is ~6 points below Maia-2's 53.9% on Masters, so more human data
would buy at most that much; and Maia's own baseline shows the *stronger* engines predict human
moves *worse* (Stockfish 40%), i.e. human-move accuracy is not a strength metric past a point.
**[inference]** The SL scaling curve will bend near 0.53; treat imitation as the initialiser,
not the ceiling. Rating conditioning (Maia-2) is a way to train on *all* Lichess data while
still being able to ask the net for the 2400-player's move at play time.

### 1.4 Leela Chess Zero: raw-policy Elo

Sources: https://lczero.org/blog/2024/02/transformer-progress/ and
https://lczero.org/blog/2024/02/how-well-do-lc0-networks-compare-to-the-greatest-transformer-network-from-deepmind/
(both read).

The lc0 team reports raw-policy strength *relative* to T78 (their strongest ResNet, 194.5M
params, 12.45 GFLOPs): BT1 +13, BT2 +123, BT3 +179, BT4 +270 Elo, "The raw policy of BT4 is 270
Elo stronger than that of T78, our strongest convolution-based model, with fewer parameters and
computations" (BT4: 191.3M params, 7.6 GFLOPs, 15 layers, embedding 1024, 32 heads). The
comparison post's methodology: value test with "go nodes 1" per legal move, policy test = top
policy move vs puzzle solution, with the caveat that "puzzle test performance based on value is
typically about 150 Elo stronger than a policy test"; "The latest BT4 network has higher policy
puzzle accuracy than the best transformer network from DeepMind has value puzzle accuracy".
An absolute policy Elo is **not** given by lc0 in either post. Ruoss's Table 1 puts the Leela
policy net (1 node) at internal tournament Elo 2292, next to their own 270M at 2299. A forum
statement that "T40 nets are just above 2300 at 1 node/move" exists
(https://groups.google.com/g/lczero/c/ptHXXUkmwR4) but is a community claim, not primary.

---

## 2. Distilling from an engine

### 2.1 Ruoss again (see 1.1) and "Stop Regressing" (Farebrother et al. 2024)

Source: https://arxiv.org/abs/2403.03950 (PDF read).

HL-Gauss: project a Gaussian `N(target, sigma^2)` onto histogram bins of width `bin = (vmax -
vmin)/m` and train with cross-entropy; "a more interpretable hyper-parameter that we recommend
tuning is sigma/bin ... we set sigma/bin = 0.75 for our experiments, which distributes mass to
approximately 6 locations." Ruoss uses the same: "HL-Gauss loss ... standard deviation of
sigma=0.75/K ~ 0.05" with K=128. Why it helps: classification "mitigate[s] issues inherent to
value-based RL, such as noisy targets and non-stationarity"; two-hot targets are worse than
HL-Gauss because they do "not fully harness the ordinal structure".

Chess result (their section 4.3.2): same 10M-game Stockfish-16 dataset, 9M/137M/270M models,
HL-Gauss vs one-hot targets, "We omit MSE as Ruoss et al. (2024) demonstrate that 1-Hot targets
outperform MSE on this task"; "HL-Gauss closes the performance gap with the substantially
stronger AlphaZero with 400 MCTS simulations" on the 10k Lichess puzzles; Figure 1 caption
reports "70% improvement" in that gap for chess. Also: "while the performance of MSE regression
losses typically plateaus upon increasing model capacity beyond ResNet-34, HL-Gauss is able to
leverage this capacity" (multi-task Atari).

Meaning for this project. Our value head regresses a scalar game outcome with MSE (AlphaZero
style). Both papers say a binned, label-smoothed classification head learns better from noisy
targets and scales better with width. The cheapest architecture change on the list.

### 2.2 How many Stockfish-labelled positions per hour can this laptop produce?

Measured 2026-09-22 with the Homebrew `stockfish` (Stockfish 18), one thread, `bench 16 1 <depth>
default depth` (16 bench positions, NNUE):

| depth | total ms / 16 positions | ms per position | positions/hour, 1 thread | positions/hour, 8 threads **[inference: linear]** |
|---|---|---|---|---|
| 10 | 587 | 37 | ~98,000 | ~780,000 |
| 12 | 1,787 | 112 | ~32,000 | ~260,000 |
| 13 | 2,816 | 176 | ~20,000 | ~160,000 |

Caveats: measured 0.76-0.84 Mnps, which is low for Apple silicon (a MacRumors thread reports
Homebrew SF16 at 4.4 Mnps vs 9.1 Mnps built from source on M2:
https://forums.macrumors.com/threads/stockfish-patch-to-improve-performance-on-m1-by-up-to-80.2326552/
, community numbers); the bench positions are a fixed mix; real mid-game positions cost more.
So: **state-values for the existing 5.7M SL positions at depth 10-12 cost roughly 7-22 hours of
8 CPU threads**, which can run while the GPU trains. **Action-values for every legal move
(Ruoss's best target, ~35 moves/position) are ~35x more, i.e. weeks — not affordable at depth
12 for millions of positions; affordable for ~200k positions or by restricting to the top-k
policy moves [inference].**

Free alternative: the Lichess database page states "About 6% of the games include Stockfish
analysis evaluations" as `[%eval ...]` comments (https://database.lichess.org/). With ~90M
standard rated games per month (2026-08: 91,912,325 in `standard/counts.txt`) that is ~5.4M
evaluated games per month, all Elo levels, tens of millions of evaluated positions per month
before any Elo filter **[inference from the two published numbers]**. The evals are server-side
Stockfish at a depth Lichess does not state on that page (not found). Our `scripts/prepare_games.py`
already streams PGN; parsing `%eval` is a small change.

---

## 3. KL-to-prior regularisation in RL fine-tuning

### 3.1 RLHF-style penalty: Ziegler et al. 2019

Source: https://arxiv.org/abs/1909.08593 (PDF read). (Stiennon 2020 and Ouyang 2022 inherit the
same term; not fetched here — Ziegler is the primary for the mechanism.)

"To keep pi from moving too far from rho, we add a penalty with expectation beta KL(pi, rho) ...
we perform RL on the modified reward R(x, y) = r(x, y) - beta log(pi(y|x)/rho(y|x))." Purpose:
"it plays the role of an entropy bonus, it prevents the policy from moving too far from the
range where r is valid". beta is either constant or adapted by "the log-space proportional
controller" toward a target KL(pi, rho); Table 10 / appendix shows what happens without it
(degenerate repeated tokens, "These These These").

### 3.2 AlphaStar's human-KL term and z-conditioning: Vinyals et al., Nature 2019

Source: unformatted author manuscript
https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf (read).

Mechanism, verbatim: "Each agent is initialized to the parameters of the supervised learning
agent. Subsequently, during reinforcement learning, we either condition the agent on a statistic
z, in which case agents receive a reward for following the strategy corresponding to z, or train
the agent unconditionally ... Agents also receive a penalty whenever their action probabilities
differ from the supervised policy." Methods: "we initialise the policy parameters to the
supervised policy and continually minimise the KL divergence between the supervised and current
policy"; z "encoding each player's build order, defined as the first 20 constructed buildings
and units"; "These pseudo-rewards measure the edit distance between sampled and executed build
orders, and the Hamming distance ... Each type of pseudo-reward is active (i.e. non-zero) with
probability 25%, and separate value functions and losses are computed for each pseudo-reward."

The ablation (Fig. 3E, "Human data usage", test Elo in a simplified setup): No human data 149;
Supervised (init only, frozen) 936; Human init 1020; + Supervised KL 1400; + Statistics (z) 1540.
"We found our use of human data to be critical in achieving good performance with reinforcement
learning (Fig. 3E)." So the always-on KL-to-the-supervised-policy was worth +380 Elo over
initialising alone, and the z pseudo-rewards another +140.

### 3.3 Kickstarting: Schmitt et al. 2018

Source: https://arxiv.org/abs/1803.03835 (PDF read).

Loss: `l_kick = l_RL + lambda_k * H(pi_T(a|x_t) || pi_S(a|x_t))`, "where H(.||.) is the
cross-entropy" to a fixed teacher, "weighted at optimisation iteration k by the scaling lambda_k
>= 0"; "after a certain number of iterations T0 we would like to have lambda_k = 0 for all k >
T0 so that eventually the student becomes independent of the teacher". Schedules (Table 2,
DMLab-30): constant lambda=1 plateaus at 41.2, linear 1->0 over 1B frames reaches 59.4, over 2B
60.8, over 4B 53.1; "Agents with a constant setting of lambda_k perform well at first, but their
performance ultimately plateaus. Linear schedules can work well, but only if chosen correctly.
PBT performance is competitive". Headline: "matching the performance of an agent trained from
scratch in almost 10x fewer steps, and surpassing its final performance by 42%".

### 3.4 Contrast with the project's current guard

The target-KL early stop bounds `KL(pi_old || pi_new)` per update. Over 500 updates at 0.02 each
that permits unbounded drift from the prior; the "prior alarm" only detects the consequence
(losing to the prior) after the fact. AlphaStar's term is `KL(pi_sup || pi_theta)` measured
against the *frozen* supervised net on the *current* states, every update, forever. Ziegler's
term is the same thing inside the reward. Andrychowicz et al. (section 5) found KL-to-behaviour
regularisers unhelpful in MuJoCo, conjecturing "the PPO policy loss already enforces the trust
region which makes KL penalties or constraints redundant" — but that is KL to the *previous*
policy, which is exactly what we already have and exactly what does not prevent slow drift.
**[inference]** A prior-KL term is the direct remedy for the observed failure: it trades a bit of
asymptotic freedom (Kickstarting's plateau with constant lambda) for a floor under the policy.
Anneal it (Kickstarting) or adapt beta to a target prior-KL (Ziegler) to avoid the plateau.

---

## 4. Opponent diversity

### 4.1 AlphaStar League (same source as 3.2)

Verbatim recipe: "Main Agents are trained with a proportion of 35% SP, 50% PFSP against all past
players in the League, and an additional 15% of PFSP matches against forgotten main players the
agent can no longer beat and past main exploiters. If there are no forgotten players or strong
exploiters, the 15% is used for self-play instead. Every 2e9 steps, a copy of the agent is added
as a new player to the League. Main agents never reset." PFSP weighting: opponent B is sampled
with probability proportional to `f(P[A beats B])`, "Choosing f_hard(x) = (1 - x)^p makes PFSP
focus on the hardest players, where p controls how entropic the resulting distribution is. Since
f_hard(1) = 0, no games are played" against opponents already beaten 100%; a variance weighting
`f_var(x) = x(1-x)` is the alternative. "Main exploiter agents play only against the current
iteration of main agents"; "league exploiter agents ... find systemic weaknesses of the entire
league"; both "are periodically reinitialized"; league exploiters snapshot "when they defeat all
players in the League in more than 70% of games" and then reset to supervised with 25% chance.

Evidence (Fig. 3C/D, simplified setup, test Elo and "a proxy for forgetting: the minimum win-rate
against all past versions"): pFSP+SP 1540, SP 1519, pFSP 1273, FSP 1143; "Naive self-play has
high Elo, but is more forgetful." League composition (Fig. 3A): main agents only 1540, + main
exploiters 1693, + league exploiters 1824. Introduction: self-play "may chase cycles (for example,
where A defeats B, and B defeats C, but A loses to C) indefinitely without making progress".

### 4.2 PSRO (Lanctot et al. 2017) and Fictitious Self-Play (Heinrich, Lanctot, Silver 2015)

Sources: https://arxiv.org/abs/1711.00832 (abstract); http://proceedings.mlr.press/v37/heinrich15.pdf (read).
PSRO: "approximate best responses to mixtures of policies generated using deep reinforcement
learning" with "empirical game-theoretic analysis to compute meta-strategies for policy
selection"; it "generalizes previous ones such as InRL, iterated best response, double oracle,
and fictitious play"; motivation: independent RL policies "overfit to the other agents' policies
during training". FSP: "players repeatedly play a game, at each iteration choosing a best
response to their opponents' average strategies. The average strategy profile of fictitious
players converges to a Nash equilibrium in certain classes of games, e.g. two-player zero-sum".

### 4.3 Rectified Nash response (Balduzzi et al. 2019, already cited in RESOURCES)

Source: https://arxiv.org/abs/1901.08106 (PDF read for the algorithm). Algorithm 3 (PSRO_N):
compute the Nash `p_t` over the population's win matrix, train the new agent against the
Nash-weighted mixture. Algorithm 4 (PSRO_rN): "for agent v_t with positive mass in p_t do
v_{t+1} <- oracle(v_t, sum_i p_t[i] * floor(phi_{w_i}(.))_+)" — each Nash-supported agent trains
only against the opponents it already beats (the rectifier drops negative payoffs), which
"amplifies the positive coordinates, of the Nash-supported agents". They find "PSRO_rN
outperforms PSRO_N, both of which greatly outperform self-play" in Blotto-style games, and a
uniform-mixture response "PSRO_U ... performs comparably to PSRO_N". Warning: "A pathological
mode of PSRO_rN is when there are many extremely local niches."

Meaning for this project. Our pool is 4 recent snapshots of one lineage sampled (roughly)
uniformly: that is closest to FSP over a short window, the *weakest* line in AlphaStar's Fig. 3C
(1143). Two ingredients are missing: (a) weighting by current win-rate (PFSP, f_hard), which
keeps the agent training on the opponents it currently loses to, including *the prior* — which
we already know it loses to in the decay phase; (b) the "15% forgotten players" slot, which is
literally the anti-forgetting term. Exploiters are cheap for us only in the hand-made form
(the existing one-ply punisher). **[inference]** Chess is close to transitive (Balduzzi's own
criterion for when self-play works), so the collapse is more likely forgetting/drift than
strategic cycling; PFSP + prior-in-pool is the right-sized fix, a full league is not.

---

## 5. PPO stabilisers for entropy collapse and forgetting

### 5.1 Engstrom et al. 2020, "Implementation Matters"

Source: https://arxiv.org/abs/2005.12729 (PDF read). Studied "Value function clipping ... Reward
scaling ... Orthogonal initialization and layer scaling ... Adam learning rate annealing" (plus
reward/observation clipping, tanh, gradient clipping). Finding: "reward normalization, Adam
annealing, and network initialization each significantly impact the rewards landscape ... and
were necessary for attaining the highest PPO reward within the tested hyperparameter grid."
Clipping itself: "the clipping mechanism is not necessary to achieve high performance — we find
that PPO-NoClip performs uniformly better than PPO-M" (Table 3: Humanoid PPO 806, PPO-NoClip
831, PPO-M without code-level tricks 674). On trust regions: the clipped objective's trust region
"is heavily dependent on the method with which the clipped PPO objective is optimized ... we can
end up moving arbitrarily far from the trust region."

### 5.2 Andrychowicz et al. 2021, "What Matters for On-Policy Deep Actor-Critic Methods"

Source: https://arxiv.org/abs/2006.05990 (PDF read; 250,000 agents, five MuJoCo tasks).
"Linearly decaying the learning rate to 0 increases the performance on 4 out of 5 tasks but the
gains are very small apart from Ant, where it leads to 15% higher scores"; recommendation: "may
slightly improve performance but is of secondary importance." Advantage normalisation:
"per-minibatch advantage normalization (C67) seems not to affect the performance too much."
Value loss: "PPO-style value loss clipping (C13) hurts the performance regardless of the clipping
threshold"; "Use GAE with lambda = 0.9 but neither Huber loss nor PPO-style value loss clipping."
Regularisers: "We do not find evidence that any of the investigated regularizers helps
significantly" (entropy, KL to reference, KL to behaviour), conjecturing the PPO loss already
enforces the trust region. Clipping: "Start with the clipping threshold set to 0.25". Note the
domain: continuous control with fresh initialisation, not a fine-tuned discrete policy with a
strong prior — the entropy/KL conclusion does not transfer directly **[inference]**.

### 5.3 Phasic Policy Gradient (Cobbe, Hilton, Klimov, Schulman 2021)

Source: https://arxiv.org/abs/2009.04416 (PDF read). "Interference between policy and value
function optimization can negatively affect performance" and "value function optimization can
often tolerate a significantly higher level of sample reuse than policy optimization." Two
phases: a PPO policy phase, then an auxiliary phase optimising `L_joint = L_aux + beta_clone *
E[KL(pi_old || pi_theta)]`, "where pi_old is the policy right before the auxiliary phase
begins", with `L_aux = 1/2 (V^pi(s) - V_targ)^2` on an auxiliary value head of the policy
network. Auxiliary phases must be infrequent: "performance suffers when we perform auxiliary
phases too frequently ... relatively infrequent auxiliary phases are critical to success"
(N_pi swept 2-32). "Compared to PPO, we find that PPG significantly improves sample efficiency on
the challenging Procgen Benchmark."

### 5.4 Primacy bias / resets (Nikishin et al. 2022)

Source: https://arxiv.org/abs/2205.07802 (PDF read). "a tendency to overfit initial experiences
that damages the rest of the learning process"; heavy priming: "after collecting 100 data points,
we update the agent 1e5 times ... the agent with heavy priming is unable to recover" even after
a million new transitions. Remedy: "Given an agent's neural network, periodically re-initialize
the parameters of its last few layers while preserving the replay buffer." Atari-100k IQM: SPR
0.38 -> SPR+resets 0.478; DMC: SAC 501 -> SAC+resets 656. The mechanism relies on the replay
buffer as the knowledge carrier; for on-policy the paper points to ITER, which "fully resets the
agent's network in an on-policy buffer-free setting with distillation from the previous
generation". **[inference]** In our setting the SL prior *is* the reset target and distillation
from it is the KL term of section 3, so "reset" and "KL-to-prior" are the same family.

### 5.5 Loss of plasticity (Lyle et al. 2023; Dohare et al. 2024)

Sources: https://proceedings.mlr.press/v202/lyle23b/lyle23b.pdf (read);
https://www.nature.com/articles/s41586-024-07711-7 (PDF read).

Lyle 2023: "loss of plasticity is deeply connected to changes in the curvature of the loss
landscape, but that it often occurs in the absence of saturated units"; they "apply the
best-performing intervention, layer normalization, to a standard DQN architecture and obtain
significant improvements". Dohare 2024 has the one directly relevant on-policy experiment, PPO on
an ant whose friction changes every 2M steps: "The standard PPO learning algorithm fails
catastrophically on the non-stationary ant problem. If the optimizer of PPO (Adam) is tuned in a
custom way, then the failure is less severe, but adding continual backpropagation or L2
regularization is necessary to perform well indefinitely." Even on the stationary ant, "PPO
increased for about 3 million steps but then collapsed. After 20 million steps, the ant is
failing every episode and is unable to learn". General finding: "L2 regularization ... reduced
loss of plasticity in many cases ... L2 regularization stops the weights from becoming too large";
"Shrink and Perturb is L2 regularization plus small random changes in weights"; and,
"Surprisingly, popular methods such as Adam, Dropout and normalization actually increased loss of
plasticity." Continual backpropagation re-initialises "a small fraction of less-used units after
each example".

Which of these has evidence for "gets better then decays"? Only Dohare's PPO-ant (rise, then
collapse under a drifting environment; self-play *is* a drifting environment) and Nikishin's
priming (overfit to early data, cannot use later data). The others (LR anneal, adv-norm, PPG)
are efficiency gains without a rise-then-fall story. Our LR-decay arm therefore tests a
"secondary importance" knob by Andrychowicz's account; L2 (weight decay toward zero, or toward
the prior's weights) is the cheap knob with the matching failure story **[inference: L2-to-prior
is my adaptation; Dohare tests L2-to-zero]**.

---

## 6. Auxiliary targets and architecture

Sources: https://lczero.org/blog/2020/04/wdl-head/, https://lczero.org/blog/2020/05/lc0-v0.25-has-been-released/,
https://lczero.org/blog/2024/02/transformer-progress/ (all read).

WDL head: "All networks starting from July 2019 have the so-called WDL head, and are capable of
predicting win, draw, and loss probabilities separately"; motivation: a scalar expected score
"doesn't account for draws". Moves-left head (v0.25, Apr 2020): "makes it possible for a neural
network to predict the remaining game length", use: "Making search take the shorter route
towards winning (hopefully reducing 'shuffling')". Neither post gives an Elo number for either
head (not found).

ResNet -> transformer: lc0 did not publish a fixed-compute ResNet-vs-transformer curve; the
published table is compute-*cheaper* and stronger: BT4 191M params / 7.6 GFLOPs vs T78 194.5M /
12.45 GFLOPs, +270 policy Elo. The ingredients they credit: a per-layer trainable 64x64xh
attention bias ("play as if 50% larger with a negligible loss in throughput") and smolgen
("compress the current representation of the position into a small vector. For each head,
generate a supplemental set of attention logits to add to those generated by dot-product
attention before the softmax"); 8 one-hot piece planes for the current and last 7 plies as
token embeddings; "our models do best with small head depths". The stated reason ResNets fall
short: "For the a1 square to 'learn' about what piece is on h8, the information must make at
least 7 trips from square to square."

Meaning for this project. At our scale (10 blocks, receptive field radius 10 covers the board)
the long-range argument is weaker than at lc0's 40-block scale **[inference]**. A 64-token
transformer with ~4-8 layers and 256-dim embeddings is a similar parameter budget and would let
us copy the attention-bias/smolgen trick; but the SL data (5.7M) is the binding constraint, not
the body. WDL as a 3-way softmax is a strict superset of the scalar head and pairs naturally with
the HL-Gauss idea (section 2.1). The moves-left head buys nothing without search except as an
auxiliary signal; lc0 does not claim a training benefit for it (not found).

---

## 7. Data

Source: https://database.lichess.org/ and `standard/counts.txt` (fetched 2026-09-22).

Monthly standard rated games in 2026: Mar 90,074,196; Apr 89,962,564; May 90,887,615; Jun
86,483,328; Jul 89,288,421; Aug 91,912,325. Total since 2013-01: 8,130,696,420 games over 165
months. "About 6% of the games include Stockfish analysis evaluations". The page does not
publish the Elo distribution; the share of games with both players >= 2000 is **not** available
from the primary source. For scale: our current prior used 3M positions from ~375k such games of
one month at 8 positions per game (repo logs), i.e. we sample <1% of the positions in those
games and one month out of 165.

How far does imitation scale? Maia-2 (169M games, 9.1B positions, 23M params): 53.25% top-1
(section 1.3). Maia-1 (12M games per bin, 6x64 net): "over 52%". Tuyls et al. 2023
(https://arxiv.org/abs/2307.09423, PDF read): BC loss and return follow "power laws plus a
constant" in compute, `N_opt = a_N C^alpha + b_N, D_opt = a_D C^beta + b_D`; "for the Atari games
the scaling laws hold all the way until expert performance. For NetHack, we find more FLOPs ...
will be required"; "our return power law initially continues to improve with scale, we find it
will plateau well before expert performance in NetHack". Their BC datasets: ~1B Atari samples,
55B NetHack samples. No chess experiments in Tuyls. **[inference]** Human-move imitation for
chess plateaus in the low-50s% top-1 for the same reason (human decisions are noisy); the
Stockfish-value target does not have this ceiling because the label is the *evaluation*, not a
noisy human choice — that is Ruoss's whole argument.

---

## Ranking and shortlist

Ranked by (expected Elo gain) x (evidence quality) / (laptop cost). Baselines for every
experiment: 200-game matches vs LONG256X and vs the SL prior (SLHI2_256), plus the Stockfish
`UCI_Elo` 1320 ladder.

1. **Prior-KL term in PPO (AlphaStar/Ziegler/Kickstarting).** Add `beta * KL(pi_prior ||
   pi_theta)` on the rollout states every update, frozen prior; start beta so the term is ~10% of
   the policy loss and either anneal linearly to a floor (Kickstarting) or adapt beta to a target
   prior-KL of, say, 0.3-0.5 nats (Ziegler controller). Expected: the 200-600-update rise is
   kept and the decline vs the prior disappears; asymptote may be lower than an unconstrained
   run's peak. Cost: one loss term, one extra forward pass of the prior per update (~+15% update
   time); a standard 7,500-plies-per-board run. Success: win-rate vs prior never drops below 50%
   over 1,000 updates while win-rate vs LONG256X rises; ladder score vs 1320 improves from 1.5/40.
   Evidence: +380 Elo in AlphaStar's ablation; strongest direct evidence for our exact symptom.

2. **Stockfish-value head (Ruoss + HL-Gauss + WDL).** Label the existing 5.7M SL positions with
   Stockfish state values at depth 10-12 (7-22 CPU-hours on 8 threads; or parse Lichess `%eval`
   for free), replace the scalar value head with K=64-128 win% bins trained with HL-Gauss
   (sigma/bin 0.75), and train it jointly with the human-move policy head in SL. Expected: a much
   better critic for PPO (lower value loss, better advantages) and a value that can be used to
   *pick* moves by one-ply lookahead over legal moves at eval time — which is still "no search"
   in Ruoss's sense but a policy improvement operator over the raw policy [inference: Ruoss's
   state-value agent needs one forward pass per legal move]. Cost: a labelling script, a head
   change, one SL run (~a night). Success: PPO from the new prior shows lower value loss and
   longer rise before any decline; the one-ply-value player beats the raw policy head in 200
   games; ladder vs 1320 moves off the floor.

3. **PFSP pool with the prior and forgotten snapshots (AlphaStar).** Keep the 4-snapshot pool but
   (a) always include the frozen prior, (b) sample opponents proportional to `(1 - winrate)^p`
   with p=1-2 from a running win-rate estimate, (c) reserve 15% of games for the oldest snapshot
   the agent scores <50% against. Expected: removes the "sharp self" bias, keeps the agent
   training on exactly the opponents it is starting to lose to, which is the prior in our decay
   phase. Cost: small code change; a standard run. Success: min win-rate against all past
   snapshots (AlphaStar's forgetting proxy) stays above 45%; vs prior no decline.
   Compare directly with the running prior-only-pool arm.

4. **L2 toward the prior's weights (Dohare-style regulariser, adapted) + LayerNorm check.** Add
   `lambda * ||theta - theta_prior||^2` (lambda ~1e-4 to 1e-3 relative to loss scale). Expected:
   a parameter-space analogue of experiment 1, cheaper (no extra forward pass) but less targeted
   (it does not know which states matter). Cost: two lines; a standard run. Success: same
   criteria as 1. Run after 1 so the two can be compared; keep whichever preserves more rise.
   Evidence: Dohare's PPO-ant collapse fixed by L2; Lyle's layer-norm result is for DQN.

5. **More/better SL data via rating-conditioning (Maia-2), or a small 64-token transformer body
   (lc0).** Lowest priority: Maia-2 shows human-move accuracy tops out ~53-54% even at 9.1B
   positions, so another 20M positions buys maybe +2-3 points of top-1 with an unknown Elo
   effect; the body change is only worth testing once data is no longer the constraint.
   Cost: days of download/preprocessing and multi-night SL runs. Success: top-1 > 0.50 on the
   >=2000 eval set and a 200-game win vs SLHI2_256.

Not recommended from this survey: LR decay alone (Andrychowicz: "secondary importance"; no
rise-then-fall evidence), per-minibatch advantage normalisation changes (no effect found), value
loss clipping (hurts), PPG (an efficiency gain for a different failure), a full exploiter league
(cost without a cycling diagnosis; chess is close to transitive).

## Open items not resolved by primary sources

- Absolute 1-node Elo of a specific lc0 net on a human/CCRL scale (only Ruoss's internal 2292
  and a forum claim of ~2300 for T40).
- Depth of Lichess server `%eval` annotations.
- Share of Lichess games with both players >= 2000 per month.
- Maia's playing strength in Elo.
