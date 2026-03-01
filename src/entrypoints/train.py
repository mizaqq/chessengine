import torch
import random
import numpy as np
from typing import Dict, Any
from src.environment.environment import EnvSpawner
from src.model.chess_model import ChessPolicyProbs
from src.model.training import run_chess_training


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def run_training_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run training from configuration dictionary.

    Args:
        config: Training configuration with keys:
            - num_envs: Number of parallel environments
            - max_updates: Maximum number of training updates
            - steps_per_update: Steps per update
            - learning_rate: Learning rate (optional, default 1e-4)
            - seed: Random seed (optional, default 42)

    Returns:
        Dictionary with training results including logs and losses
    """
    num_envs = config.get("num_envs", 12)
    max_updates = config.get("max_updates", 30000)
    steps = config.get("steps_per_update", 15)
    lr = config.get("learning_rate", 1e-4)
    seed = config.get("seed", 42)
    lr_decay_interval = config.get("lr_decay_interval", 100)

    set_seed(seed)

    envs = EnvSpawner(num_envs)
    white_model = ChessPolicyProbs()
    black_model = ChessPolicyProbs()
    optimizer_white = torch.optim.Adam(white_model.parameters(), lr=lr)
    optimizer_black = torch.optim.Adam(black_model.parameters(), lr=lr)

    logs, losses, white_model, black_model = run_chess_training(
        envs,
        white_model,
        black_model,
        optimizer_white,
        optimizer_black,
        steps=steps,
        episodes=max_updates,
        lr_decay_interval=lr_decay_interval,
    )

    return {
        "logs": logs,
        "losses": losses,
        "white_model": white_model,
        "black_model": black_model,
    }


def main():
    """Main entry point for training."""
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description="Train chess RL agent")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/train_default.yaml",
        help="Path to config file",
    )
    parser.add_argument("--max-updates", type=int, help="Override max updates")
    parser.add_argument("--seed", type=int, help="Override random seed")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.max_updates:
        config["max_updates"] = args.max_updates
    if args.seed:
        config["seed"] = args.seed

    result = run_training_from_config(config)

    print(f"Training complete. Processed {len(result['logs'])} log entries.")
    return result


if __name__ == "__main__":
    main()
