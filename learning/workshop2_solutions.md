# Workshop 2 solutions
Only after a check failed twice. One block per task, in notebook order.

## depth_curriculum

```python
def next_depth(current, recent, since_rise, window, advance_rate, depth_max):
    if since_rise >= window and current < depth_max:
        rate = sum(recent) / len(recent)
        if rate >= advance_rate:
            return min(current + 2, depth_max)
    return current

MY_DEPTH_61 = 4
MY_DEPTH_59 = 2
```

## read_depth

```python
DEEPEST_TRAINED = 8
NEVER_TRAINED = [10, 20]
WHY_STUCK = """The defender is our own network and leaves the human line after a ply or two, so the demonstrator stops helping; the 0.6 bar then has to be met by the bare network at every depth up to the current one, and it never was."""
```

## guided_mixture

```python
def mixture(p, demo, eps):
    b = p.clone()
    if demo:
        idx = torch.tensor(sorted(demo), dtype=torch.long)
        b = (1.0 - eps) * p
        b[idx] += eps / len(idx)
    return b

MY_MATE_PROB = 0.5 * 0.1 + 0.5          # 0.55
MY_OTHER_PROB = 0.5 * 0.9 / 19          # 0.0237
```

## read_clean

```python
# Read the plot: the arm that races to level 3 counted every win; the one frozen at 0 flagged by action;
# the one that climbs to 1 flips the coin. Replace the letters with what you see.
MY_RULES = {k: {"CURSIL": "every_win", "CURSIL2": "flag_by_action", "CURSIL3": "coin"}[v] for k, v in HIDDEN3.items()}
```

## match_score

```python
def match_score(counts):
    points = total = 0
    score_white = {"white_win": 1.0, "black_win": 0.0, "draw": 0.5, "unfinished": 0.5}
    for key, n in counts.items():
        colour, result = key.split("_", 2)[1], key.split("_", 2)[2]
        s = score_white[result]
        points += n * (s if colour == "white" else 1 - s)
        total += n
    return points / total

MY_SCORE = (3 + 2.5 + 2 + 1) / 20        # 0.425
```

## read_matrix

```python
BEST_ROW_M1 = "SL"
RL_VS_RL_MEAN = 0.5
DRIFT_SENTENCE = """The RL checkpoints are at par with each other and all lose to the prior: self-play optimised beating the current self, and the current self had drifted below the prior."""
```

## pool_members

```python
MEMBERS_AT_120 = 3                      # prior, snap_50, snap_100
PRIOR_PROB_AT_120 = 1 / 3
MEMBERS_AT_300 = 5                      # prior + the last four snapshots
PRIOR_PROB_AT_300 = 1 / 5
SNAPS_AT_300 = ["snap_150", "snap_200", "snap_250", "snap_300"]
```

## read_pool

```python
REACHED_PAR = False
RL_ORDER_AFTER_POOL = "ordered"
```

## good_starts

```python
def bucket_weight(rate, r_min, r_max, replay_share, probe_share, unknown_w=1.0):
    if rate is None:
        return unknown_w
    if rate > r_max:
        return replay_share
    if rate < r_min:
        return probe_share
    return 1.0

MY_WEIGHTS = [1.0, 0.2, 1.0, 0.1]
```

## read_technique

```python
CLEAN_OR_ALL = "clean"
WHY_CLEAN = """The gap between all and clean is the demonstrator's mates; a ladder that climbs on them reaches positions the network alone cannot solve, and the held-out rate stays where it was."""
```

## sil_terms

```python
def sil_terms(logp, ret, value):
    adv = max(ret - value, 0.0)
    return -logp * adv, 0.5 * adv ** 2

MY_POLICY_TERMS = [-math.log(0.2) * 1.5, 0.0, 0.0]
```

## sil_returns

```python
def buffer_returns(sides, tr_white, tr_black, gamma):
    remaining = {1: 0, 0: 0}
    out = [0.0] * len(sides)
    for k in range(len(sides) - 1, -1, -1):
        side = sides[k]
        remaining[side] += 1
        terminal = tr_white if side == 1 else tr_black
        out[k] = terminal * gamma ** remaining[side]
    return out

MY_RETURNS = [2 * 0.99 ** 3, -2 * 0.99 ** 2, 2 * 0.99 ** 2, -2 * 0.99, 2 * 0.99]
```

## read_sil

```python
NEAR_ONE = "sil_valid_share"
WHY_NOT_CONTRADICTION = """Minibatches are drawn with probability proportional to (R - V)+, so nearly every drawn ply is valid even when most of the buffer has priority zero; the buffer share says how much is left to learn."""
```

## lambda_credit

```python
def credit(gamma, lam, k):
    return (gamma * lam) ** k

MY_CREDIT_095 = (0.99 * 0.95) ** 10
MY_CREDIT_1 = 0.99 ** 10
```

## net_gain

```python
PIECE_VALUE = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9}
def net_gain(victim, attacker, recapturable):
    return PIECE_VALUE[victim] - (PIECE_VALUE[attacker] if recapturable else 0)

MY_GAINS = [9, 0, -4]
MY_GIFTS = [True, False, False]
```

## read_audit

```python
def punish_rate(b, p):
    return b + p * (1 - b)
MY_RATE_MATERIAL = 0.54 + 0.5 * 0.46
MY_RATE_MATE = 0.13 + 0.5 * 0.87
WHY_NOT_PRIOR = """The SL row shows the prior takes a free piece about as often as we do and delivers allowed mates even less often, so a stronger-looking opponent is not a punishing one."""
```

## read_calibration

```python
HUMAN_HELPS_AT = 10
CEILING_SENTENCE = """The defender leaves the record after about one ply, so the ceiling is 'human first move, then us'; at depth 20 one good first move is not a plan, which points at plan initiation as the weakness."""
```
