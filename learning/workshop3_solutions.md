# Workshop 3 solutions
Only after a check failed twice. One block per task, in notebook order.

## punish_arms

```python
MY_PUNISH = {"PUNISH":   {"who": "learner",  "explore_shift": True,  "own_m1": "down"},
             "POOL_REF": {"who": "opponent", "explore_shift": False, "own_m1": "flat"}}
```

## dials_vs_strength

```python
BEST_BY_PUZZLES = "LONG3"
BEST_BY_GAMES = "LONG192"
WHAT_EACH_MEASURES = """Puzzles measure the skills the curriculum taught from given positions; games measure what the prior knew about ordinary positions, and the old lineage's prior was starved."""
```

## diagnose_plateau

```python
def train_eval_gap(entries):
    pairs = [(e["train_top1"], e["eval_top1"]) for e in entries if "eval_top1" in e]
    return sum(t - v for t, v in pairs) / len(pairs)

IF_CAPACITY_LIMITED_192_WOULD = "rise"
OBSERVED_192 = "same"
DIAGNOSIS = "data"
```

## read_width

```python
MY_WIDTH = {k: v for k, v in HIDDEN4.items()}   # read the plot: the arm whose prior_score stays near 0.5 with higher entropy and lower KL is LONG192
WHICH_PANEL = """prior_score: the wider run holds par with its prior while the narrow one drifts to 0.35; entropy and KL agree (more options kept, smaller steps)."""
```

## scaling

```python
def log_fit(points):
    x = np.log10([p for p, _ in points]); y = np.array([t for _, t in points])
    b = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    a = y.mean() - b * x.mean()
    return float(a), float(b)

_a_, _b_ = log_fit(_pts)
MY_PREDICTION_5_7M = _a_ + _b_ * math.log10(5.7e6)
WHY_NOT_SAME_AXIS = """Its held-out set is different (positions from Elo>=2000 games), so its top-1 is not comparable; only the win matrix between priors compares across sets."""
```

## two_numbers

```python
PROBE_KEY = "selfplay_top1"
EVAL_KEY = "mate_in_one_rate"
NOISE_BAND = 0.2
```

## amdahl

```python
def speedup_ceiling(total_s, network_s):
    return total_s / (total_s - network_s)
MY_CEILING = 5.0
MY_STEP_SPEEDUP_128 = 977 / 38
WHY_ONLY_5X = """Amdahl's law: environment stepping and rules checks stay on the CPU and become the floor once the network is nearly free."""
```
