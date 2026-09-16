"""Generate `workshop_rl_chess.ipynb` (the owner's self-test workshop) and
`learning/workshop_solutions.md`.

Usage: python -m scripts.build_workshop [--check]
  --check  fills every task with its solution and executes all code cells to make
           sure the checks pass (matplotlib in Agg mode).
"""
import argparse
import json
import sys
from pathlib import Path

CELLS = []          # (type, source)
TASKS = {}          # task name -> solution source (replaces the whole task cell)


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


def task(name, text, solution):
    """A code cell the owner completes; `solution` is the finished cell."""
    TASKS[name] = solution.strip("\n")
    code(f"# TASK: {name}\n" + text.strip("\n"))


# --------------------------------------------------------------------------- 0
md("""
# Workshop: what this chess engine learned, and whether you did

This notebook walks through every RL idea used in `chessengine` up to the PPO +
mate-in-two runs of 2026-09-16. Each section has three parts:

1. **Plain words** — the idea in one or two paragraphs, with a chess picture.
2. **The real code** — the lines from `src/` that implement it.
3. **Your turn** — a short task: fill in the `...`, run the cell, then run the
   `check_*` cell below it. A ✅ means your answer matches the engine's code or a
   hand-computed value. Some tasks ask you to read a plot; there you fill a dict.

Rules: no peeking at `learning/workshop_solutions.md` before the check fails
twice. Write your reflections in the last cell; that text goes into
`learning/records/`.

Run the setup cell first.
""")
code("""
import json, math, random
from pathlib import Path
import torch
import matplotlib.pyplot as plt

torch.manual_seed(0); random.seed(0)

def check(name, ok, detail=""):
    print(("✅ " if ok else "❌ ") + name + (f"  ({detail})" if detail else ""))

def close(a, b, tol=1e-3):
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    return a.shape == b.shape and bool(torch.allclose(a, b, atol=tol, rtol=0))

def load_run(path):
    d = json.loads(Path(path).read_text())
    return d["logs"], d["config"]

def series(logs, key):
    xs = [e["episode"] for e in logs if e.get(key) is not None]
    ys = [e[key] for e in logs if e.get(key) is not None]
    return xs, ys

print("setup ok")
""")

# --------------------------------------------------------------------------- 1
md("""
## 1. Return, value, advantage

**Plain words.** After a game the only honest signal is the outcome: +2 for a win,
-2 for a loss, -0.5 for a draw. The *return* of a position is everything the
mover will collect from here on, each future reward shrunk by `gamma` per own
move so that a mate now beats a mate in fifty moves. The *value head* is the
network's guess of that return. The *advantage* is return minus guess: how much
better than expected this move turned out. The policy gradient pushes up moves
with positive advantage and down moves with negative advantage, so a bad guess
in the value head adds noise to every move's push.

**Real code** — the backward walk in `src/model/returns.py` (shaping removed for
now, one player):

```python
R = bootstrap_value.clone()
for i in reversed(range(len(steps))):
    R = reward[i] + gamma * R
    returns[i] = R
```

`bootstrap_value` is the value head's guess for the position where the rollout
was cut off; the walk starts from it because the game is not over there.
""")
task("discounted_returns", """
def discounted_returns(rewards, bootstrap, gamma):
    \"\"\"rewards: list of own-move rewards r_0..r_{T-1}; bootstrap: V(s_T).
    Return the list [R_0, ..., R_{T-1}] with R_t = r_t + gamma * R_{t+1}, R_T = bootstrap.\"\"\"
    R = ...
    out = [None] * len(rewards)
    for i in reversed(range(len(rewards))):
        ...
    return out

# By hand first: rewards 1, 2, 3, bootstrap 10, gamma 0.99. Write your three numbers here
# before running the check (last one first is easiest: R_2 = 3 + 0.99 * 10).
MY_HAND_ANSWER = [..., ..., ...]
""", """
def discounted_returns(rewards, bootstrap, gamma):
    R = bootstrap
    out = [None] * len(rewards)
    for i in reversed(range(len(rewards))):
        R = rewards[i] + gamma * R
        out[i] = R
    return out

MY_HAND_ANSWER = [15.62329, 14.771, 12.9]
""")
code("""
_r = discounted_returns([1, 2, 3], 10.0, 0.99)
check("discounted_returns", close(_r, [15.62329, 14.771, 12.9]), f"got {[round(x, 4) for x in _r]}")
check("hand answer", close(MY_HAND_ANSWER, [15.62329, 14.771, 12.9]), f"you wrote {MY_HAND_ANSWER}")
""")

# --------------------------------------------------------------------------- 2
md("""
## 2. The opponent is part of the environment

**Plain words.** From white's chair, black's move is just how the board changes
between two white moves (Sutton & Barto's tic-tac-toe player has "no model of its
opponent of any kind"). So the return walk for white steps *over* black's plies
without discounting, and a game that ends on black's move still hands white its
terminal reward. One rollout holds both colours' steps; each side reads only its
own entries.

**Real code** — `compute_returns_for_model` (`src/model/returns.py`):

```python
is_mine = step.player == model_id
done_f = step.done.float()
R = R * (1.0 - done_f) + terminal_r        # game end: future value := terminal reward
new_R = shaped + gamma * R
R = torch.where(is_mine, new_R, R)         # opponent steps pass R through
```

The rollout below: white moves (value guess 0.2), then black moves and that move
is checkmate. The game ended on black's ply; white's terminal reward is -2.
gamma 0.99, no shaping. Bootstrap 5.0 (should not matter: the game ended).
""")
task("cross_player", """
# What return does white's step 0 receive? Think: the game ended on black's ply, white's
# terminal reward (-2) replaces the future, then white's own move discounts it once.
WHITE_RETURN_STEP0 = ...
""", """
WHITE_RETURN_STEP0 = 0.99 * -2.0
""")
code("""
from src.core.types import StepRecord
from src.model.returns import compute_returns_for_model
def _step(player, done=False, tr_w=0.0, tr_b=0.0, value=0.0):
    n = 1
    return StepRecord(player=torch.tensor([player]), obs=torch.zeros(n, 20, 8, 8), legal_mask=torch.ones(n, 4674),
        action=torch.zeros(n, dtype=torch.long), old_log_prob=torch.zeros(n), old_value=torch.tensor([value]),
        entropy=torch.zeros(n), num_legal=torch.ones(n) * 20, potential_white=torch.zeros(n),
        done=torch.tensor([done]), terminal_r_white=torch.tensor([tr_w]), terminal_r_black=torch.tensor([tr_b]))
_steps = [_step(1, value=0.2), _step(0, done=True, tr_w=-2.0, tr_b=2.0)]
_ret = compute_returns_for_model(_steps, 1, torch.tensor([5.0]), 0.99, torch.tensor([0.0]), 0.0)
check("white return at step 0", close([WHITE_RETURN_STEP0], _ret[0]), f"engine says {_ret[0].item():.4f}, you said {WHITE_RETURN_STEP0}")
""")

# --------------------------------------------------------------------------- 3
md("""
## 3. Potential-based shaping (material as a hint, not a goal)

**Plain words.** Wins are rare early on, so we add a hint: material. But a hint
that pays for captures teaches piece-grabbing over mating. Ng, Harada & Russell
(1999) showed the one safe form: pay `gamma * Phi(next) - Phi(now)` where `Phi`
is any score of the position. Over a whole game those payments telescope: the
sum is `Phi(end) - Phi(start)`, and with `Phi(end) = 0` at the terminal position
the hint contributes nothing net. The optimal policy is unchanged; only the
learning signal is denser. Price: the value head learns `V - Phi`, so to read it
you add material back (see `visualize_trained.ipynb`).

**Real code** (`compute_returns_for_model`):

```python
phi = sign * step.potential_white
shaped = shaping_scale * (gamma * phi_next - phi)
```
""")
task("shaping", """
def shaped_reward(phi_now, phi_next, gamma, scale):
    return ...

# Telescoping check by hand: potentials along white's moves 0 -> 1 -> 4, terminal 0, gamma 1, scale 1.
# Sum of the three shaped rewards = ?
MY_TELESCOPE_SUM = ...
""", """
def shaped_reward(phi_now, phi_next, gamma, scale):
    return scale * (gamma * phi_next - phi_now)

MY_TELESCOPE_SUM = 0.0
""")
code("""
_phis = [0.0, 1.0, 4.0, 0.0]
_sum = sum(shaped_reward(_phis[i], _phis[i + 1], 1.0, 1.0) for i in range(3))
check("shaped_reward telescopes to Phi(end) - Phi(start)", close([_sum], [0.0]), f"sum {_sum}")
check("hand telescope sum", close([MY_TELESCOPE_SUM], [_sum]))
_g = 0.99
_sum2 = sum(shaped_reward(_phis[i], _phis[i + 1], _g, 0.2) for i in range(3))
print(f"with gamma 0.99 the sum is {_sum2:.4f}: not exactly zero, because each step also discounts the next potential")
""")

# --------------------------------------------------------------------------- 4
md("""
## 4. One network for both colours: orient the board to the mover

**Plain words.** The network sees 20 grids of 8x8 yes/no answers. Planes 0-11
are piece types in (white, black) pairs, then empty squares, repetitions, side
to move, no-progress counter, and four castling rights. As white, "my pawns" are
plane 0 with my back rank at row 0. For black we flip the rows and swap each
white plane with its black twin, so the network always sees *my pieces first, my
side at the bottom* (AlphaZero: "the board is oriented to the perspective of the
current player"). Move ids in OpenSpiel are already mover-relative, so only the
observation changes.

**Real code** (`src/model/orientation.py`):

```python
PLANE_PERMUTATION = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14, 15, 18, 19, 16, 17]
def orient_black(obs): return obs.flip(-2)[..., _PERM, :, :]
```
""")
task("orientation", """
# A black pawn stands on rank index 6 (its second rank) in plane 1. After orient_black, which
# plane holds it and on which rank index? Also: which plane index does the castling right
# "black left" (plane 18) move to?
ORIENTED_PAWN = {"plane": ..., "rank": ...}
BLACK_LEFT_CASTLING_GOES_TO = ...
""", """
ORIENTED_PAWN = {"plane": 0, "rank": 1}
BLACK_LEFT_CASTLING_GOES_TO = 16
""")
code("""
from src.model.orientation import orient_black, PLANE_PERMUTATION
_obs = torch.zeros(20, 8, 8); _obs[1, 6, 3] = 1.0      # black pawn, rank index 6, file d
_o = orient_black(_obs)
_plane, _rank = [int(x) for x in _o.nonzero()[0][:2]]
check("pawn plane/rank", ORIENTED_PAWN == {"plane": _plane, "rank": _rank}, f"engine: plane {_plane}, rank {_rank}")
check("castling plane", BLACK_LEFT_CASTLING_GOES_TO == PLANE_PERMUTATION.index(18), f"engine: {PLANE_PERMUTATION.index(18)}")
check("orientation is its own inverse", bool(torch.equal(orient_black(orient_black(_obs)), _obs)))
""")

# --------------------------------------------------------------------------- 5
md("""
## 5. GAE: how far ahead do we trust what actually happened?

**Plain words.** Section 1's advantage uses the whole observed return: honest
but noisy, because one lucky rollout swings it. The other extreme is the
one-step surprise `r + gamma * V(next) - V(now)`: calm, but wrong wherever the
value head is wrong. GAE (Schulman et al. 2016) blends them: sum the surprises
along the rollout, each shrunk by `(gamma * lambda)^k`. Lambda 1 is the full
return minus value; lambda 0 is the one-step surprise; 0.95 leans on the observed
future for a dozen moves, then trusts the value head.

**Real code** (`compute_gae_for_model`, `src/model/returns.py`):

```python
delta   = shaped + gamma * next_value - step.old_value
new_gae = delta + gamma * lam * gae
```
""")
task("gae", """
def gae_single(rewards, values, bootstrap, gamma, lam):
    \"\"\"One player, no dones. rewards r_t, values V_t (same length), bootstrap V_T.
    Return advantages A_t = delta_t + gamma*lam*A_{t+1}, delta_t = r_t + gamma*V_{t+1} - V_t.\"\"\"
    A, next_value = 0.0, bootstrap
    out = [None] * len(rewards)
    for i in reversed(range(len(rewards))):
        delta = ...
        A = ...
        out[i] = A
        next_value = values[i]
    return out

# Hand check: rewards 1,2,3; values 0.5,0.5,0.5; bootstrap 10; gamma 0.99.
# lambda 0 gives the raw surprises. What is the last one, A_2?
MY_A2_LAMBDA0 = ...
""", """
def gae_single(rewards, values, bootstrap, gamma, lam):
    A, next_value = 0.0, bootstrap
    out = [None] * len(rewards)
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_value - values[i]
        A = delta + gamma * lam * A
        out[i] = A
        next_value = values[i]
    return out

MY_A2_LAMBDA0 = 3 + 0.99 * 10 - 0.5
""")
code("""
_r, _v = [1, 2, 3], [0.5, 0.5, 0.5]
check("lambda 0 = one-step surprises", close(gae_single(_r, _v, 10.0, 0.99, 0.0), [0.995, 1.995, 12.4]))
check("lambda 1 = return - value", close(gae_single(_r, _v, 10.0, 0.99, 1.0), [x - 0.5 for x in [15.62329, 14.771, 12.9]]))
_a2 = 12.4; _a1 = 1.995 + 0.99 * 0.95 * _a2; _a0 = 0.995 + 0.99 * 0.95 * _a1
check("lambda 0.95 blend", close(gae_single(_r, _v, 10.0, 0.99, 0.95), [_a0, _a1, _a2]))
check("hand A_2 at lambda 0", close([MY_A2_LAMBDA0], [12.4]))
""")

# --------------------------------------------------------------------------- 6
md("""
## 6. PPO: reuse the rollout, but only inside a trust band

**Plain words.** Plain policy gradient can take exactly one step per rollout:
after the step the policy has changed and the samples describe the old one. PPO
(Schulman et al. 2017) looks at each taken move's ratio `pi_new(a) / pi_old(a)`
and, in the direction that would help, freezes the objective once the ratio
leaves `[1 - eps, 1 + eps]`. Inside the band the update is ordinary; outside it
the sample contributes no gradient. That bound makes four passes over the same
rollout safe. At the very first minibatch the ratio is exactly 1, so PPO's step
equals the old REINFORCE step: A2C is PPO with one pass and no clip.

**Real code** (`ppo_policy_loss`, `src/model/training.py`):

```python
ratio = torch.exp(new_log_prob - old_log_prob)
clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
return -torch.min(ratio * advantages, clipped * advantages).mean(), ratio
```
""")
task("ppo_loss", """
def clipped_surrogate(new_logp, old_logp, adv, eps):
    \"\"\"Return the PPO policy loss (a scalar tensor to minimise).\"\"\"
    ratio = ...
    unclipped = ...
    clipped = ...
    return ...

# Four samples: (advantage, ratio). Which of them give ZERO policy gradient under eps 0.2?
#   s1: adv +2, ratio 1.5     s2: adv +2, ratio 1.1     s3: adv -1, ratio 0.5     s4: adv -1, ratio 0.9
ZERO_GRADIENT = {"s1": ..., "s2": ..., "s3": ..., "s4": ...}   # True / False each
""", """
def clipped_surrogate(new_logp, old_logp, adv, eps):
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    return -torch.min(unclipped, clipped).mean()

ZERO_GRADIENT = {"s1": True, "s2": False, "s3": True, "s4": False}
""")
code("""
from src.model.training import ppo_policy_loss
_old = torch.log(torch.rand(64) * 0.5 + 0.01); _new = _old + torch.randn(64) * 0.3; _adv = torch.randn(64)
_mine = clipped_surrogate(_new, _old, _adv, 0.2); _ref, _ = ppo_policy_loss(_new, _old, _adv, 0.2)
check("clipped_surrogate matches the engine", close([_mine.item()], [_ref.item()], 1e-5))
def _zero_grad(adv, ratio):
    lp = torch.log(torch.tensor([ratio])).requires_grad_(True)
    ppo_policy_loss(lp, torch.zeros(1), torch.tensor([adv]), 0.2)[0].backward()
    return abs(lp.grad.item()) < 1e-9
_truth = {"s1": _zero_grad(2.0, 1.5), "s2": _zero_grad(2.0, 1.1), "s3": _zero_grad(-1.0, 0.5), "s4": _zero_grad(-1.0, 0.9)}
check("zero-gradient samples", ZERO_GRADIENT == _truth, f"engine: {_truth}")
""")

# --------------------------------------------------------------------------- 7
md("""
## 7. Reading the dials: clip fraction and approximate KL

**Plain words.** Two numbers say how far an update moved the policy. *Clip
fraction*: the share of samples whose ratio left the band. *Approximate KL*:
`mean(old_logp - new_logp)`, roughly how surprised the old policy would be by
the new one. Both should be small but not zero; zero means nothing is learning,
0.5 means the trust band is a fiction. On 2026-09-16 both PPO arms sat near clip
fraction 0.24, but KL was 0.12 at learning rate 3e-4 and 0.05 at 1e-4, and only
the lower rate kept the opening policy alive.
""")
task("diagnostics", """
def clip_fraction(ratio, eps):
    return ...

def approx_kl(old_logp, new_logp):
    return ...
""", """
def clip_fraction(ratio, eps):
    return ((ratio - 1).abs() > eps).float().mean()

def approx_kl(old_logp, new_logp):
    return (old_logp - new_logp).mean()
""")
code("""
_ratio = torch.tensor([1.0, 1.25, 0.7, 1.1, 0.85])
check("clip_fraction", close([clip_fraction(_ratio, 0.2).item()], [0.4]))
check("approx_kl", close([approx_kl(torch.tensor([0.0, -1.0]), torch.tensor([-0.1, -1.3])).item()], [0.2]))
""")

# --------------------------------------------------------------------------- 8
md("""
## 8. Reading the plots: which curve is which arm?

Four arms ran on 2026-09-16 from a fresh network, 250 updates of 64 plies, same
seed. **A**: A2C, lambda 1. **B**: A2C, lambda 0.95. **C**: PPO at learning rate
3e-4. **C2**: PPO at 1e-4. The plots below hide the names behind random letters.
Use what you know: PPO fits the value head better (16 steps per rollout), the
lower learning rate keeps entropy high, A2C at 64 plies took 4x fewer gradient
steps than PPO and learned the puzzles worst.
""")
code("""
ARMS = {"A": "experiments/ppo-gae/A_a2c_lam1/run.json", "B": "experiments/ppo-gae/B_a2c_lam095/run.json",
        "C": "experiments/ppo-gae/C_ppo_lr3e-4/run.json", "C2": "experiments/ppo-gae/C2_ppo_lr1e-4/run.json"}
_logs = {k: load_run(v)[0] for k, v in ARMS.items()}
_rng = random.Random(7); _letters = list("WXYZ"); _rng.shuffle(_letters)
HIDDEN = dict(zip(_letters, ARMS))          # letter -> arm, kept secret until the check
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for letter, arm in HIDDEN.items():
    for ax, key, title in zip(axes, ["mean_entropy_normalized_game", "mean_value_loss", "lichess_top1"],
                              ["opening-board entropy", "value loss", "Lichess held-out top-1"]):
        xs, ys = series(_logs[arm], key); ax.plot(xs, ys, marker="o" if key == "lichess_top1" else None, label=letter); ax.set_title(title)
for ax in axes: ax.legend(); ax.set_xlabel("update")
plt.show()
""")
task("read_arms", """
# Assign each hidden letter to an arm name: "A", "B", "C" or "C2". Reason from the three panels.
MY_ARMS = {"W": ..., "X": ..., "Y": ..., "Z": ...}
""", """
MY_ARMS = dict(HIDDEN)
""")
code("""
_right = {k: MY_ARMS.get(k) == v for k, v in HIDDEN.items()}
check("arms identified", all(_right.values()), f"correct letters: {[k for k, v in _right.items() if v]}; truth {HIDDEN}")
""")
md("""
### 8b. Clip fraction versus KL

The next plot shows the two PPO arms only. Clip fraction alone does not separate
them; KL does. Which letter is the lower learning rate, and which dial told you?
""")
code("""
_two = {k: v for k, v in HIDDEN.items() if v in ("C", "C2")}
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for letter, arm in _two.items():
    for ax, key in zip(axes, ["mean_clip_fraction", "mean_approx_kl"]):
        xs, ys = series(_logs[arm], key); ax.plot(xs, ys, label=letter); ax.set_title(key)
for ax in axes: ax.legend(); ax.set_xlabel("update")
plt.show()
""")
task("read_kl", """
LOWER_LR_LETTER = ...           # "W", "X", "Y" or "Z"
DIAL_THAT_SEPARATES = ...       # "clip_fraction" or "approx_kl"
""", """
LOWER_LR_LETTER = [k for k, v in HIDDEN.items() if v == "C2"][0]
DIAL_THAT_SEPARATES = "approx_kl"
""")
code("""
check("lower learning rate", HIDDEN.get(LOWER_LR_LETTER) == "C2")
check("separating dial", DIAL_THAT_SEPARATES == "approx_kl")
""")

# --------------------------------------------------------------------------- 9
md("""
## 9. Why the ratio must depend on parameters only (the BatchNorm trap)

**Plain words.** PPO compares the probability the network gives a move now with
the probability it gave when the move was sampled. If the forward pass depends on
which other positions sit in the batch, the comparison is polluted. BatchNorm in
training mode does exactly that: it normalises with the batch's own statistics.
Measured on a fresh network with no weight change: clip fraction 0.52. The fix in
this repo: keep the network in eval mode during training and refresh the running
statistics from the rollout once per update (`refresh_norm_stats`).

The cell below re-measures it on a tiny network. Predict first: in which mode is
the ratio exactly 1?
""")
task("bn_mode", """
RATIO_IS_ONE_IN_MODE = ...    # "train" or "eval"
""", """
RATIO_IS_ONE_IN_MODE = "eval"
""")
code("""
from torch.distributions import Categorical
from src.model.chess_model import ChessPolicyProbs
_m = ChessPolicyProbs(num_filters=8); _obs = torch.rand(64, 20, 8, 8); _mask = torch.ones(64, 4674)
def _drift(mode):
    _m.train(mode == "train")
    with torch.no_grad():
        p1, _ = _m(_obs[:12], _mask[:12]); a = Categorical(probs=p1).sample(); lp_old = Categorical(probs=p1).log_prob(a)
        p2, _ = _m(_obs, _mask); lp_new = Categorical(probs=p2[:12]).log_prob(a)
    return (torch.exp(lp_new - lp_old) - 1).abs().max().item()
_d = {mode: _drift(mode) for mode in ("train", "eval")}
print({k: round(v, 4) for k, v in _d.items()})
check("mode with ratio 1", RATIO_IS_ONE_IN_MODE == min(_d, key=_d.get))
""")

# --------------------------------------------------------------------------- 10
md("""
## 10. Reverse curriculum: start next to the reward

**Plain words.** A mate from the opening is hundreds of moves of luck away, so a
random policy never finds one and never gets a reward to learn from. Florensa et
al. (2017) start the agent next to the goal and widen from there; Salimans & Chen
(2018) show this turns exponential exploration into roughly quadratic. Here: 4 of
12 boards start from Lichess mate-in-one and mate-in-two positions. Each episode
is one attempt, and it must fail fast: a mate-in-two board ends at once if the
first move is not the verified forcing move, so the network is never paid for a
mate that only happened because its own defence blundered.

**Real code** (`OpenSpielEnv.step`):

```python
if self.mover_moves == 1 and self.key_actions is not None and action[0] not in self.key_actions:
    self.failed = True
```
""")
task("key_move", """
from src.envs.open_spiel_env import OpenSpielEnv
M2_FEN = "7k/8/5K2/8/8/8/8/1R6 w - - 0 1"        # 1.Kg6! Kg8 2.Rb8#  (key move f6g6)

def first_move_is_key(env, action, key_ucis):
    \"\"\"True if `action` (an OpenSpiel action id) plays one of `key_ucis` in env's current position.
    Hint: env.actions_for_uci(list_of_uci) -> set of action ids.\"\"\"
    return ...
""", """
from src.envs.open_spiel_env import OpenSpielEnv
M2_FEN = "7k/8/5K2/8/8/8/8/1R6 w - - 0 1"

def first_move_is_key(env, action, key_ucis):
    return action in env.actions_for_uci(key_ucis)
""")
code("""
_env = OpenSpielEnv(); _env.reset(M2_FEN)
(_key,) = _env.actions_for_uci(["f6g6"]); (_slack,) = _env.actions_for_uci(["b1b2"])
check("key move recognised", first_move_is_key(_env, _key, ["f6g6"]) is True)
check("slack move rejected", first_move_is_key(_env, _slack, ["f6g6"]) is False)
_env.reset(M2_FEN, puzzle_moves=2, key_moves=["f6g6"]); _env.step(_slack)
check("engine ends the episode at once", _env.is_done() and _env.game_result() == "puzzle_miss")
""")

# --------------------------------------------------------------------------- 11
md("""
## 11. Reading the mate-in-two run

The long run (mate-in-one + mate-in-two boards, ply cap 120, PPO at 1e-4) lives
in `experiments/mate-in-two-ply-cap/`. The plot shows training solved rate by
depth next to the held-out accuracy on each set. Two questions follow.
""")
code("""
LONG = Path("experiments/mate-in-two-ply-cap/LONG/run.json")
BOTH = Path("experiments/mate-in-two-ply-cap/BOTH/run.json")
_runs = [p for p in (BOTH, LONG) if p.exists()]
if _runs:
    _logs2, _offset = [], 0
    for p in _runs:
        lg, cfg = load_run(p)
        _logs2 += [{**e, "episode": e["episode"] + _offset} for e in lg]
        _offset += cfg["max_updates"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key in ["puzzle_solved_rate_m1", "puzzle_solved_rate_m2"]:
        xs, ys = series(_logs2, key); axes[0].plot(xs, ys, label=key)
    for key in ["lichess_top1", "lichess_m2_top1"]:
        xs, ys = series(_logs2, key); axes[1].plot(xs, ys, marker="o", label=key)
    axes[0].set_title("training boards: solved rate"); axes[1].set_title("held-out: top-1"); axes[0].legend(); axes[1].legend()
    plt.show()
    LAST = {k: series(_logs2, k)[1][-1] for k in ["puzzle_solved_rate_m1", "puzzle_solved_rate_m2", "lichess_top1", "lichess_m2_top1"]}
    print({k: round(v, 3) for k, v in LAST.items()})
else:
    print("run not finished yet; come back after experiments/mate-in-two-ply-cap/LONG exists")
""")
task("read_m2", """
# 1. Which depth is harder at the end of the run, by held-out top-1?  "m1" or "m2"
HARDER_DEPTH = ...
# 2. The training solved rate for m2 counts a win only after the forcing move AND the mate against
#    the network's own defence; held-out m2 checks only the first move. Which of the two should be
#    LOWER, and why? Write a sentence.
WHICH_IS_LOWER_AND_WHY = \"\"\"...\"\"\"
""", """
HARDER_DEPTH = "m2"
WHICH_IS_LOWER_AND_WHY = \"\"\"The training solved rate should be lower: it needs the key move and the follow-up mate against a live defender, while held-out top-1 needs only the key move.\"\"\"
""")
code("""
if _runs:
    check("harder depth", HARDER_DEPTH == ("m2" if LAST["lichess_m2_top1"] < LAST["lichess_top1"] else "m1"), f"held-out m1 {LAST['lichess_top1']:.3f} vs m2 {LAST['lichess_m2_top1']:.3f}")
    print("your sentence:", WHICH_IS_LOWER_AND_WHY)
    print("expected direction: training m2 solved rate below held-out m2 top-1, because it also demands the mating move against the network's own defence.")
""")

# --------------------------------------------------------------------------- 12
md("""
## 12. Your words

Write three to five sentences: what surprised you, which dial you would watch
first in the next run, and where outside chess the same ideas apply (PPO in RLHF
for language models, curricula in sparse-reward robotics, potential-based
shaping in any hint-heavy reward design). This text is what goes into
`learning/records/` as evidence of understanding.
""")
code("""
MY_REFLECTION = \"\"\"
...
\"\"\"
print(MY_REFLECTION)
""")


def build_notebook():
    cells = []
    for ctype, src in CELLS:
        cell = {"cell_type": ctype, "metadata": {}, "source": src.splitlines(keepends=True)}
        if ctype == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def build_solutions():
    out = ["# Workshop solutions\n", "Only after a check failed twice. One block per task, in notebook order.\n"]
    for name, src in TASKS.items():
        out.append(f"\n## {name}\n\n```python\n{src}\n```\n")
    return "".join(out)


def run_check():
    import matplotlib
    matplotlib.use("Agg")
    ns = {}
    for ctype, src in CELLS:
        if ctype != "code":
            continue
        if src.startswith("# TASK: "):
            name = src.splitlines()[0][len("# TASK: "):]
            src = TASKS[name]
        exec(compile(src, "<cell>", "exec"), ns)
    print("all cells executed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    Path("workshop_rl_chess.ipynb").write_text(json.dumps(build_notebook(), indent=1))
    Path("learning/workshop_solutions.md").write_text(build_solutions())
    print("wrote workshop_rl_chess.ipynb and learning/workshop_solutions.md")
    if args.check:
        run_check()


if __name__ == "__main__":
    main()
