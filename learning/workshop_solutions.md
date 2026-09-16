# Workshop solutions
Only after a check failed twice. One block per task, in notebook order.

## discounted_returns

```python
def discounted_returns(rewards, bootstrap, gamma):
    R = bootstrap
    out = [None] * len(rewards)
    for i in reversed(range(len(rewards))):
        R = rewards[i] + gamma * R
        out[i] = R
    return out

MY_HAND_ANSWER = [15.62329, 14.771, 12.9]
```

## cross_player

```python
WHITE_RETURN_STEP0 = 0.99 * -2.0
```

## shaping

```python
def shaped_reward(phi_now, phi_next, gamma, scale):
    return scale * (gamma * phi_next - phi_now)

MY_TELESCOPE_SUM = 0.0
```

## orientation

```python
ORIENTED_PAWN = {"plane": 0, "rank": 1}
BLACK_LEFT_CASTLING_GOES_TO = 16
```

## gae

```python
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
```

## ppo_loss

```python
def clipped_surrogate(new_logp, old_logp, adv, eps):
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    return -torch.min(unclipped, clipped).mean()

ZERO_GRADIENT = {"s1": True, "s2": False, "s3": True, "s4": False}
```

## diagnostics

```python
def clip_fraction(ratio, eps):
    return ((ratio - 1).abs() > eps).float().mean()

def approx_kl(old_logp, new_logp):
    return (old_logp - new_logp).mean()
```

## read_arms

```python
MY_ARMS = dict(HIDDEN)
```

## read_kl

```python
LOWER_LR_LETTER = [k for k, v in HIDDEN.items() if v == "C2"][0]
DIAL_THAT_SEPARATES = "approx_kl"
```

## bn_mode

```python
RATIO_IS_ONE_IN_MODE = "eval"
```

## key_move

```python
from src.envs.open_spiel_env import OpenSpielEnv
M2_FEN = "7k/8/5K2/8/8/8/8/1R6 w - - 0 1"

def first_move_is_key(env, action, key_ucis):
    return action in env.actions_for_uci(key_ucis)
```

## read_m2

```python
HARDER_DEPTH = "m2"
WHICH_IS_LOWER_AND_WHY = """The training solved rate should be lower: it needs the key move and the follow-up mate against a live defender, while held-out top-1 needs only the key move."""
```
