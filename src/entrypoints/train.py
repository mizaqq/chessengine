import torch
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from src.envs.open_spiel_vector_env import OpenSpielVectorEnv
from src.envs.open_spiel_async_vector_env import OpenSpielAsyncVectorEnv
from src.model.chess_model import ChessPolicyProbs
from src.model.training import run_chess_training
from src.envs.start_positions import StartPositionSampler, load_puzzles
from src.eval.puzzles import evaluate_mate_in_one


SHAPING_MODES = ("potential", "none")


def resolve_shaping(config: Dict[str, Any]) -> float:
    """Resolve `shaping` and `shaping_scale` into the scale used by the returns walk.

    `potential` (default) applies potential-based material shaping with
    `shaping_scale` (default 0.2, must be >= 0). `none` disables shaping, which is
    outcome-only reward, and resolves to a scale of 0.
    """
    mode = config.get("shaping", "potential")
    if mode not in SHAPING_MODES:
        raise ValueError(
            f"Unknown shaping {mode!r}; accepted values: {', '.join(SHAPING_MODES)}"
        )
    scale = float(config.get("shaping_scale", 0.2))
    if scale < 0:
        raise ValueError(f"shaping_scale must be >= 0, got {scale}")
    return 0.0 if mode == "none" else scale


def resolve_puzzle_boards(config: Dict[str, Any]):
    """Return (sampler, num_puzzle_envs) from `puzzle_fraction` / `puzzle_train_file`.

    `puzzle_fraction` (default 0.0) is the share of boards that play one-move
    puzzle episodes instead of full games: num_puzzle_envs = round(fraction *
    num_envs). 0 keeps today's behaviour, 1 means no board ever sees an opening.
    """
    fraction = float(config.get("puzzle_fraction", 0.0))
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"puzzle_fraction must be in [0, 1], got {fraction}")
    num_envs = int(config.get("num_envs", 12))
    num_puzzle_envs = int(round(fraction * num_envs))
    if num_puzzle_envs == 0:
        return None, 0
    path = config.get("puzzle_train_file")
    if not path:
        raise ValueError("puzzle_fraction > 0 requires puzzle_train_file")
    return StartPositionSampler(load_puzzles(path), seed=int(config.get("seed", 42))), num_puzzle_envs


def resolve_eval_fn(config: Dict[str, Any]):
    """Return a held-out puzzle evaluation callable, or None without `puzzle_eval_file`."""
    path = config.get("puzzle_eval_file")
    if not path:
        return None
    puzzles = load_puzzles(path)
    return lambda white, black: evaluate_mate_in_one(white, black, puzzles)


def resolve_eval_interval(config: Dict[str, Any]) -> int:
    interval = int(config.get("eval_interval", 50))
    if interval <= 0:
        raise ValueError(f"eval_interval must be a positive integer, got {interval}")
    return interval


def save_models(white_model, black_model, save_dir, updates: int, timestamp: str = None):
    """Save both state dicts as <color>_model_<timestamp>_episodes_<updates>.pth."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d%H%M%S")
    paths = []
    for name, model in (("white", white_model), ("black", black_model)):
        path = save_dir / f"{name}_model_{timestamp}_episodes_{updates}.pth"
        torch.save(model.state_dict(), path)
        paths.append(path)
    return paths


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _create_envs(env_type: str, num_envs: int, start_sampler=None, num_puzzle_envs=0):
    if env_type == "async":
        return OpenSpielAsyncVectorEnv(num_envs, start_sampler, num_puzzle_envs)
    elif env_type == "sync":
        return OpenSpielVectorEnv(num_envs, start_sampler, num_puzzle_envs)
    else:
        raise ValueError(f"Unknown env_type: {env_type}")


def run_training_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    num_envs = config.get("num_envs", 12)
    max_updates = config.get("max_updates", 30000)
    steps = config.get("steps_per_update", 15)
    lr = config.get("learning_rate", 1e-4)
    seed = config.get("seed", 42)
    lr_decay_interval = config.get("lr_decay_interval", 100)
    lr_decay_factor = float(config.get("lr_decay_factor", 0.5))
    min_lr = float(config.get("min_lr", 3e-5))
    if min_lr > lr:
        raise ValueError(f"min_lr ({min_lr}) must not exceed learning_rate ({lr}); the rate would jump up")
    env_type = config.get("env_type", "sync")
    terminal_rewards = config.get("terminal_rewards", {"win": 2.0, "loss": -2.0, "draw": -0.5})
    gamma = config.get("gamma", 0.99)
    entropy_coef = config.get("entropy_coef", 0.01)
    grad_clip = config.get("grad_clip", 1.0)
    shaping_scale = resolve_shaping(config)
    start_sampler, num_puzzle_envs = resolve_puzzle_boards(config)
    terminal_rewards = {**terminal_rewards, "puzzle_miss": float(config.get("puzzle_miss_reward", 0.0))}
    eval_fn = resolve_eval_fn(config)
    eval_interval = resolve_eval_interval(config)
    log_interval = int(config.get("log_interval", 10))

    set_seed(seed)

    envs = _create_envs(env_type, num_envs, start_sampler, num_puzzle_envs)
    white_model = ChessPolicyProbs()
    black_model = ChessPolicyProbs()
    optimizer_white = torch.optim.Adam(white_model.parameters(), lr=lr)
    optimizer_black = torch.optim.Adam(black_model.parameters(), lr=lr)

    logs, losses, white_model, black_model = run_chess_training(
        envs, white_model, black_model,
        optimizer_white, optimizer_black,
        steps=steps, episodes=max_updates,
        lr_decay_interval=lr_decay_interval,
        lr_decay_factor=lr_decay_factor,
        min_lr=min_lr,
        terminal_rewards=terminal_rewards,
        gamma=gamma,
        entropy_coef=entropy_coef,
        grad_clip=grad_clip,
        shaping_scale=shaping_scale,
        eval_fn=eval_fn,
        eval_interval=eval_interval,
        log_interval=log_interval,
    )

    return {
        "logs": logs,
        "losses": losses,
        "white_model": white_model,
        "black_model": black_model,
    }


def main():
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description="Train chess RL agent")
    parser.add_argument("--config", type=str, default="src/configs/train_default.yaml")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--env-type", type=str, choices=["sync", "async"])
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to write white/black checkpoints after training")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.max_updates:
        config["max_updates"] = args.max_updates
    if args.seed:
        config["seed"] = args.seed
    if args.env_type:
        config["env_type"] = args.env_type

    result = run_training_from_config(config)
    print(f"Training complete. Processed {len(result['logs'])} log entries.")
    if args.save_dir:
        paths = save_models(
            result["white_model"], result["black_model"], args.save_dir, config["max_updates"]
        )
        print("Saved:", ", ".join(str(p) for p in paths))
    return result


if __name__ == "__main__":
    main()
