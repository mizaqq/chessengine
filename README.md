# Chess RL Engine

A modular reinforcement learning engine for training chess agents using PyTorch and OpenSpiel.

## Architecture

The codebase follows a **ports-and-adapters** architecture with clear separation of concerns:

```
src/
├── core/           # Core interfaces and types
├── envs/           # Environment adapters (OpenSpiel)
├── training/       # Training utilities (metrics, rollout collection)
├── losses/         # Modular loss functions
├── model/          # Neural network models
├── entrypoints/    # Training entry points
└── configs/        # Configuration files
```

### Key Components

- **`core/`**: Abstract interfaces (`VectorEnv`) and data types (`EnvStep`, `RolloutBatch`)
- **`envs/`**: OpenSpiel environment wrappers implementing the `VectorEnv` interface
- **`training/`**: `MetricsAggregator` for tracking training metrics, `RolloutCollector` for rollout data
- **`losses/`**: Modular loss functions (policy gradient, value MSE, entropy regularization)
- **`entrypoints/`**: Config-driven training entry points

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

Create a YAML config file:

```yaml
# my_config.yaml
num_envs: 12
max_updates: 30000
steps_per_update: 15
learning_rate: 0.0001
seed: 42
```

Run with custom config:

```bash
python -m src.entrypoints.train --config my_config.yaml
```

Override specific parameters:

```bash
python -m src.entrypoints.train --max-updates 5000 --seed 123
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
}

result = run_training_from_config(config)
logs = result["logs"]
losses = result["losses"]
white_model = result["white_model"]
black_model = result["black_model"]
```

### Legacy Entry Point

The original `src/main.py` still works for backward compatibility:

```bash
python -m src.main
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

Implement the `VectorEnv` interface:

```python
from src.core.interfaces import VectorEnv
from src.core.types import EnvStep
import torch

class MyCustomEnv(VectorEnv):
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
    
    def reset(self) -> EnvStep:
        obs = torch.zeros(self.num_envs, 20, 8, 8)
        legal_actions_mask = torch.ones(self.num_envs, 4674)
        reward = torch.zeros(self.num_envs)
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        return EnvStep(obs, legal_actions_mask, reward, done, {})
    
    def step(self, actions):
        # Your implementation
        pass
```

### Adding a Custom Loss

```python
from src.losses.composed_loss import ComposedLoss

# Use existing loss composer
loss_composer = ComposedLoss()
loss_dict = loss_composer.compute(
    policy_loss=my_policy_loss,
    value_loss=my_value_loss,
    entropy_bonus=my_entropy,
    entropy_coef=0.02
)

total_loss = loss_dict["total_loss"]
```

### Project Structure

```
.
├── src/
│   ├── core/               # Core abstractions
│   ├── envs/               # Environment adapters
│   ├── training/           # Training utilities
│   ├── losses/             # Loss functions
│   ├── model/              # Neural networks
│   ├── entrypoints/        # Entry points
│   ├── configs/            # YAML configs
│   └── utils/              # Utilities
├── tests/
│   ├── core/               # Core tests
│   ├── envs/               # Environment tests
│   ├── training/           # Training tests
│   ├── losses/             # Loss tests
│   └── integration/        # Integration tests
├── docs/
│   └── plans/              # Design docs and migration notes
└── README.md
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_envs` | int | 12 | Number of parallel environments |
| `max_updates` | int | 30000 | Maximum training updates |
| `steps_per_update` | int | 15 | Steps per training update |
| `learning_rate` | float | 1e-4 | Learning rate |
| `gamma` | float | 0.99 | Discount factor |
| `entropy_coef` | float | 0.01 | Entropy regularization coefficient |
| `lr_decay_interval` | int | 100 | LR decay interval |
| `lr_decay_factor` | float | 0.5 | LR decay factor |
| `min_lr` | float | 3e-4 | Minimum learning rate |
| `seed` | int | 42 | Random seed |
| `log_interval` | int | 10 | Logging interval |

## Migration from Old Code

See [Migration Notes](docs/plans/2026-02-28-rl-refactor-migration-notes.md) for detailed migration guide.

**TL;DR**: Old code still works. New code provides better modularity and testability.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
