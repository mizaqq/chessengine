# Chess RL Engine

A reinforcement learning engine for training chess agents via self-play using PyTorch and OpenSpiel.

## Architecture

One `ChessPolicyProbs` network plays both colours (AlphaZero representation, Silver et al. 2017): black-to-move observations are flipped and colour-swapped so the network always sees the mover at the bottom (`src/model/orientation.py`); OpenSpiel's move ids are already mover-relative. Both sides' samples update the same parameters in one backward pass per update. `shared_network: false` restores the legacy two-network setup.

```
src/
├── core/           # Data types (EnvStep, StepRecord)
├── envs/           # Vectorized environment wrappers (sync + async), start-position sampler
├── eval/           # Held-out puzzle evaluation
├── model/          # Neural network + training loop + returns
├── training/       # Metrics aggregation
├── losses/         # Loss composition
├── entrypoints/    # CLI entry point
└── configs/        # YAML configuration
```

### Key Components

- **`core/types.py`**: `EnvStep` (env output), `StepRecord` (per-step rollout data with per-model fields)
- **`envs/`**: `OpenSpielVectorEnv` (sync) and `OpenSpielAsyncVectorEnv` (multiprocessing) — duck-typed, same API
- **`model/training.py`**: Core training loop — rollout collection, returns computation, backpropagation
- **`model/chess_model.py`**: `ChessPolicyProbs` — ResNet (10 blocks, 128 filters) with policy + value heads
- **`model/orientation.py`**: mover-oriented observations for the shared network
- **`model/checkpoints.py`**: `model_*.pth` (shared) and legacy `white_/black_model_*.pth` loading
- **`model/returns.py`**: Discounted returns with potential-based material shaping and cross-player done propagation
- **`training/metrics.py`**: `MetricsAggregator` — windowed stats (win rates, entropy, illegal moves)
- **`losses/composed_loss.py`**: `total = policy + value - entropy_coef * normalized_entropy`

## Installation

```bash
# Install dependencies
pip install -e .

# Or with requirements.txt
pip install -r requirements.txt
```

**Dependencies:**
- Python >= 3.13
- PyTorch >= 2.10.0
- open-spiel >= 1.6.11
- matplotlib >= 3.10.8
- tqdm >= 4.67.3
- PyYAML (for config loading)

## Usage

### Quick Start

Run training with default configuration:

```bash
python -m src.entrypoints.train
```

### Custom Configuration

Create a YAML config file (see `src/configs/train_default.yaml` for all options):

```yaml
num_envs: 12
max_updates: 30000
steps_per_update: 15
learning_rate: 0.0001
seed: 42
env_type: sync          # sync | async
gamma: 0.99
entropy_coef: 0.01
grad_clip: 1.0
terminal_rewards:
  win: 2.0
  loss: -2.0
  draw: -0.5
```

Run with custom config:

```bash
python -m src.entrypoints.train --config my_config.yaml
```

Override specific parameters:

```bash
python -m src.entrypoints.train --max-updates 5000 --seed 123 --env-type async
```

### Programmatic Usage

```python
from src.entrypoints.train import run_training_from_config

config = {
    "num_envs": 12,
    "max_updates": 30000,
    "steps_per_update": 15,
    "learning_rate": 1e-4,
    "seed": 42,
    "env_type": "sync",
    "gamma": 0.99,
    "entropy_coef": 0.01,
    "grad_clip": 1.0,
    "terminal_rewards": {"win": 2.0, "loss": -2.0, "draw": -0.5},
}

result = run_training_from_config(config)
logs = result["logs"]       # list of dicts with metrics every 10 episodes
losses = result["losses"]   # per-episode total loss
white_model = result["white_model"]
black_model = result["black_model"]
```

## Testing

Run all tests:

```bash
pytest -v
```

Run specific test suites:

```bash
pytest tests/core/          # Core contracts
pytest tests/envs/          # Environment adapters
pytest tests/training/      # Training utilities
pytest tests/losses/        # Loss functions
pytest tests/integration/   # Integration tests
```

Run smoke test:

```bash
pytest tests/integration/test_training_smoke.py -v
```

## Development

### Adding a Custom Environment

Environments use duck typing — implement `reset()`, `step()`, and `num_envs`:

```python
from src.core.types import EnvStep
import torch

class MyCustomEnv:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs

    def reset(self) -> EnvStep:
        return EnvStep(
            obs=torch.zeros(self.num_envs, 20, 8, 8),
            legal_actions_mask=torch.ones(self.num_envs, 4674),
            material=torch.zeros(self.num_envs),  # white-view material of obs
            done=torch.zeros(self.num_envs, dtype=torch.bool),
            current_player=torch.ones(self.num_envs, dtype=torch.long),  # 1=white, 0=black
            info={},
        )

    def step(self, actions: torch.Tensor) -> EnvStep:
        # Must return EnvStep with auto-reset on done.
        # info["game_results"] = {env_idx: "white_win"|"black_win"|"draw"}
        pass
```

### Project Structure

```
.
├── src/
│   ├── core/types.py              # EnvStep, StepRecord dataclasses
│   ├── envs/
│   │   ├── open_spiel_env.py      # Single environment wrapper
│   │   ├── open_spiel_vector_env.py       # Sync vectorized (sequential)
│   │   └── open_spiel_async_vector_env.py # Async vectorized (multiprocessing)
│   ├── model/
│   │   ├── chess_model.py         # ChessPolicyProbs (ResNet + policy/value heads)
│   │   ├── training.py            # A2C training loop
│   │   └── returns.py             # Discounted returns computation
│   ├── training/metrics.py        # MetricsAggregator (windowed stats)
│   ├── losses/composed_loss.py    # Loss composition
│   ├── entrypoints/train.py       # CLI entry point
│   └── configs/train_default.yaml # Default configuration
├── tests/
│   ├── envs/               # Environment contract tests
│   ├── model/              # Returns + entropy normalization tests
│   ├── training/           # Metrics aggregation tests
│   ├── losses/             # Loss function tests
│   └── integration/        # Smoke + desync tests
├── docs/plans/             # Design documents
└── roadmap/                # Phase-based development roadmap
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_envs` | int | 12 | Number of parallel environments |
| `max_updates` | int | 30000 | Maximum training updates |
| `steps_per_update` | int | 15 | Rollout steps per training update |
| `learning_rate` | float | 3e-4 | Initial learning rate (3e-4 is what pre-2026-09-16 runs effectively used) |
| `gamma` | float | 0.99 | Discount factor |
| `entropy_coef` | float | 0.01 | Normalized entropy bonus coefficient |
| `grad_clip` | float | 1.0 | Gradient clipping max norm |
| `env_type` | str | sync | Environment type: `sync` or `async` |
| `terminal_rewards.win` | float | 2.0 | Reward for winning |
| `terminal_rewards.loss` | float | -2.0 | Reward for losing |
| `terminal_rewards.draw` | float | -0.5 | Reward for drawing |
| `shaping` | str | potential | `potential` (material shaping) or `none` (outcome only) |
| `shaping_scale` | float | 0.2 | Multiplier on the shaping term (>= 0) |
| `shared_network` | bool | true | One network for both colours with oriented observations; `false` = two networks |
| `puzzle_fraction` | float | 0.0 (default yaml: 0.5) | Share of boards that play one-move mate-in-one puzzle episodes; `round(fraction * num_envs)` boards |
| `puzzle_train_file` | str | – | CSV of training puzzles (required when `puzzle_fraction` > 0) |
| `puzzle_eval_file` | str | – | CSV of held-out puzzles; enables `puzzle_top1` / `puzzle_mate_prob` in the logs |
| `eval_interval` | int | 50 | Updates between held-out puzzle evaluations (also runs at the end) |

CLI: `python -m src.entrypoints.train --config <yaml> [--max-updates N] [--seed S] [--env-type sync|async] [--save-dir DIR]`. With `--save-dir`, the model is written as `model_<timestamp>_episodes_<N>.pth` (legacy mode: `<color>_model_...pth` pair). `src.model.checkpoints.load_models(dir)` loads either layout.
| `lr_decay_interval` | int | 1000 | LR decay interval (updates) |
| `lr_decay_factor` | float | 0.5 | LR multiplicative decay factor |
| `min_lr` | float | 3e-5 | Learning rate floor; must not exceed `learning_rate` |
| `puzzle_miss_reward` | float | 0.0 | Terminal reward to the mover for a missed mate on a puzzle board |
| `seed` | int | 42 | Random seed |

## Training Loop Details

The training loop (`src/model/training.py`) runs a player-agnostic A2C:

1. **Rollout collection** — steps through vectorized envs, dispatches observations to white/black models based on `current_player`
2. **Returns computation** — backward pass forming the shaped reward per own move, with cross-player done propagation and terminal rewards
3. **Backpropagation** — per-model loss: `policy_loss + value_loss - entropy_coef * normalized_entropy`

**Entropy normalization**: `raw_entropy / log(num_legal_moves)` maps entropy to [0, 1] regardless of position complexity. Forced moves (1 legal action) are clamped to avoid division by zero.

**Start-position curriculum**: `round(puzzle_fraction * num_envs)` boards play one-move episodes from Lichess mate-in-one positions (mate = win; anything else ends as `puzzle_miss` paying the mover `puzzle_miss_reward`), so the sparse outcome reward is one move away and every ply is an attempt; the other boards play full games. Logs report `puzzle_attempts`, `puzzle_solved_rate`, and entropy split into `_game` / `_puzzle` (reverse curriculum by start state; Florensa et al. 2017, Salimans & Chen 2018). Build the puzzle files with `python -m scripts.prepare_puzzles` (downloads the CC0 dump to `data/raw/`). Evaluate checkpoints with `python -m scripts.eval_checkpoint_puzzles <ckpt_dir>` and in-game with `python -m scripts.eval_games <ckpt_dir>`.

**Reward signal**: configurable terminal rewards on game end, plus potential-based material shaping (Ng, Harada & Russell 1999): each own move receives `shaping_scale * (gamma * material(next own position) - material(this position))` from the mover's perspective, with the terminal position counting as 0. Shaping sums to zero over a complete game, so only the outcome is net reward; `shaping: none` disables it.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
