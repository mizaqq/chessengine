import torch
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from src.envs.open_spiel_vector_env import OpenSpielVectorEnv
from src.envs.open_spiel_async_vector_env import OpenSpielAsyncVectorEnv
from src.model.chess_model import ChessPolicyProbs
from src.model.training import run_chess_training, A2C_PRESET, PPO_PRESET
from src.envs.start_positions import FenSampler, MixedSampler, StartPositionSampler, load_fens, load_puzzles
from src.eval.puzzles import evaluate_mate_in_one


SHAPING_MODES = ("potential", "none")
ALGORITHMS = {"a2c": A2C_PRESET, "ppo": PPO_PRESET}
ALGORITHM_KEYS = {  # config key -> preset key
    "ppo_epochs": "epochs", "ppo_minibatches": "minibatches", "clip_epsilon": "clip_epsilon",
    "normalize_advantage": "normalize_advantage", "value_coef": "value_coef", "gae_lambda": "gae_lambda",
}


def resolve_algorithm(config: Dict[str, Any]) -> Dict[str, Any]:
    """Update-step settings: `algorithm: a2c | ppo` picks a preset (design D3);
    explicit keys override it. a2c = 1 epoch, 1 minibatch, no clip, lambda 1
    (today's REINFORCE-with-baseline step); ppo = Schulman et al. 2017 values."""
    name = config.get("algorithm", "a2c")
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {name!r}; accepted values: {', '.join(ALGORITHMS)}")
    algo = dict(ALGORITHMS[name])
    for key, target in ALGORITHM_KEYS.items():
        if key in config and config[key] is not None:
            algo[target] = config[key]
    lam = float(algo["gae_lambda"])
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got {lam}")
    algo["gae_lambda"] = lam
    if algo["clip_epsilon"] is not None:
        algo["clip_epsilon"] = float(algo["clip_epsilon"])
        if algo["clip_epsilon"] <= 0:
            raise ValueError(f"clip_epsilon must be > 0 (or null for no clipping), got {algo['clip_epsilon']}")
    for key in ("epochs", "minibatches"):
        algo[key] = int(algo[key])
        if algo[key] < 1:
            raise ValueError(f"ppo_{key} must be >= 1, got {algo[key]}")
    algo["value_coef"] = float(algo["value_coef"])
    if algo["value_coef"] < 0:
        raise ValueError(f"value_coef must be >= 0, got {algo['value_coef']}")
    algo["normalize_advantage"] = bool(algo["normalize_advantage"])
    return algo


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
    """Return (sampler, num_puzzle_envs).

    Board count: `puzzle_boards` if given (integer in [0, num_envs]), else
    `round(puzzle_fraction * num_envs)` with `puzzle_fraction` in [0, 1], default 0.
    Sources: `puzzle_train_files: [{file, weight}, ...]` (weighted mix) or a single
    `puzzle_train_file`. 0 boards keeps today's behaviour; num_envs boards means
    no board ever sees an opening.
    """
    num_envs = int(config.get("num_envs", 12))
    fraction = float(config.get("puzzle_fraction", 0.0))
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"puzzle_fraction must be in [0, 1], got {fraction}")
    if "puzzle_boards" in config and config["puzzle_boards"] is not None:
        num_puzzle_envs = int(config["puzzle_boards"])
        if not 0 <= num_puzzle_envs <= num_envs:
            raise ValueError(f"puzzle_boards must be in [0, {num_envs}], got {num_puzzle_envs}")
    else:
        num_puzzle_envs = int(round(fraction * num_envs))
    if num_puzzle_envs == 0:
        return None, 0
    seed = int(config.get("seed", 42))
    files = config.get("puzzle_train_files")
    if files:
        sources = [(load_puzzles(entry["file"]), float(entry.get("weight", 1.0))) for entry in files]
        return MixedSampler(sources, seed=seed), num_puzzle_envs
    path = config.get("puzzle_train_file")
    if not path:
        raise ValueError("puzzle boards > 0 require puzzle_train_file or puzzle_train_files")
    return StartPositionSampler(load_puzzles(path), seed=seed), num_puzzle_envs


def resolve_eval_fn(config: Dict[str, Any]):
    """Held-out evaluation callable, or None.

    `puzzle_eval_files: {name: path}` reports each set under `<name>_top1`,
    `<name>_mate_prob`, `<name>_count`; a single `puzzle_eval_file` keeps the
    unprefixed `puzzle_*` keys.
    """
    named = config.get("puzzle_eval_files")
    if named:
        sets = {name: load_puzzles(path) for name, path in named.items()}

        def eval_many(white, black, oriented=False):
            out = {}
            for name, puzzles in sets.items():
                r = evaluate_mate_in_one(white, black, puzzles, oriented=oriented)
                out.update({f"{name}_top1": r["puzzle_top1"], f"{name}_mate_prob": r["puzzle_mate_prob"],
                            f"{name}_count": r["puzzle_count"]})
            return out
        return eval_many
    path = config.get("puzzle_eval_file")
    if not path:
        return None
    puzzles = load_puzzles(path)
    return lambda white, black, oriented=False: evaluate_mate_in_one(white, black, puzzles, oriented=oriented)


def resolve_midgame_boards(config: Dict[str, Any], num_puzzle_envs: int):
    """Return (FenSampler or None, num_midgame_envs). `midgame_boards` game boards
    start from FENs in `game_start_file` (mid-game human positions); they follow
    the puzzle boards and must fit within num_envs."""
    n = int(config.get("midgame_boards", 0) or 0)
    if n == 0:
        return None, 0
    num_envs = int(config.get("num_envs", 12))
    if n < 0 or num_puzzle_envs + n > num_envs:
        raise ValueError(f"midgame_boards must be in [0, {num_envs - num_puzzle_envs}], got {n}")
    path = config.get("game_start_file")
    if not path:
        raise ValueError("midgame_boards > 0 requires game_start_file")
    return FenSampler(load_fens(path), seed=int(config.get("seed", 42)) + 1000), n


def resolve_max_plies(config: Dict[str, Any]):
    """`max_plies`: game boards end as a draw after this many plies (AlphaZero
    terminated over-long games as draws); null/absent = no cap."""
    value = config.get("max_plies")
    if value is None:
        return None
    value = int(value)
    if value < 2:
        raise ValueError(f"max_plies must be >= 2 or null, got {value}")
    return value


def resolve_eval_interval(config: Dict[str, Any]) -> int:
    interval = int(config.get("eval_interval", 50))
    if interval <= 0:
        raise ValueError(f"eval_interval must be a positive integer, got {interval}")
    return interval


from src.model.checkpoints import load_models, save_model, save_models  # noqa: E402  (re-exported)


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _create_envs(env_type: str, num_envs: int, start_sampler=None, num_puzzle_envs=0, max_plies=None,
                 game_start_sampler=None, num_midgame_envs=0):
    if env_type == "async":
        return OpenSpielAsyncVectorEnv(num_envs, start_sampler, num_puzzle_envs, max_plies,
                                       game_start_sampler, num_midgame_envs)
    elif env_type == "sync":
        return OpenSpielVectorEnv(num_envs, start_sampler, num_puzzle_envs, max_plies,
                                  game_start_sampler, num_midgame_envs)
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
    algorithm = resolve_algorithm(config)
    max_plies = resolve_max_plies(config)
    game_start_sampler, num_midgame_envs = resolve_midgame_boards(config, num_puzzle_envs)

    set_seed(seed)

    envs = _create_envs(env_type, num_envs, start_sampler, num_puzzle_envs, max_plies,
                        game_start_sampler, num_midgame_envs)
    shared = bool(config.get("shared_network", True))
    init_from = config.get("init_from")
    if init_from:
        # Continue training from a saved checkpoint directory (newest file wins).
        w0, b0, oriented, paths = load_models(init_from)
        if shared != oriented:
            raise ValueError(
                f"init_from {init_from} holds a {'shared' if oriented else 'two-network'} "
                f"checkpoint but shared_network={shared}"
            )
        white_model = w0
        black_model = None if shared else b0
        print("initialised from", paths)
    else:
        white_model = ChessPolicyProbs()
        black_model = None if shared else ChessPolicyProbs()
    optimizer_white = torch.optim.Adam(white_model.parameters(), lr=lr)
    # Shared: one network for both colours, observations oriented to the mover.
    optimizer_black = None if shared else torch.optim.Adam(black_model.parameters(), lr=lr)
    if eval_fn is not None:
        eval_puzzles_fn = eval_fn
        eval_fn = lambda w, b: eval_puzzles_fn(w, b, oriented=shared)

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
        algorithm=algorithm,
    )

    return {
        "logs": logs,
        "losses": losses,
        "white_model": white_model,
        "black_model": black_model,
        "shared": shared,
        "model": white_model if shared else None,
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
        if result["shared"]:
            paths = [save_model(result["model"], args.save_dir, config["max_updates"])]
        else:
            paths = save_models(
                result["white_model"], result["black_model"], args.save_dir, config["max_updates"]
            )
        print("Saved:", ", ".join(str(p) for p in paths))
    return result


if __name__ == "__main__":
    main()
