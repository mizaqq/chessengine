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
from src.envs.start_positions import FenSampler, MixedSampler, StartPositionSampler, load_fens, load_puzzles, FinishCurriculum, load_finishes
from src.eval.finishes import evaluate_finishes
from src.eval.matches import play_match, summarize_match
from src.eval.technique import evaluate_technique
from src.envs.technique import DEFAULT_CAPS, TechniqueSampler
from src.training.opponent_pool import OpponentPool, PoolBoards, freeze
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
    finish_path = config.get("finish_eval_file")
    prior_match = resolve_prior_match(config)
    tech_games = int(config.get("technique_eval_games", 0) or 0)
    tech = resolve_technique_sets(config) if tech_games > 0 else None
    if named or finish_path or prior_match is not None or tech is not None:
        sets = {name: load_puzzles(path) for name, path in (named or {}).items()}
        finish_records = load_finishes(finish_path) if finish_path else None
        finish_kwargs = dict(
            depths=tuple(int(d) for d in config.get("finish_eval_depths", (2, 10, 20))),
            games=int(config.get("finish_eval_games", 100)),
            cap_margin=int(config.get("finish_cap_margin", 20)),
        )

        def eval_many(white, black, oriented=False):
            out = {}
            for name, puzzles in sets.items():
                r = evaluate_mate_in_one(white, black, puzzles, oriented=oriented)
                out.update({f"{name}_top1": r["puzzle_top1"], f"{name}_mate_prob": r["puzzle_mate_prob"],
                            f"{name}_count": r["puzzle_count"]})
            if finish_records is not None:
                out.update(evaluate_finishes(white, black, finish_records, oriented=oriented, **finish_kwargs))
            if prior_match is not None:
                out.update(prior_match(white, oriented))
            if tech is not None:
                out.update(evaluate_technique(white, black, tech[0], tech[1], games=tech_games,
                                              oriented=oriented, seed=int(config.get("seed", 42)) + 5000))
            return out
        return eval_many
    path = config.get("puzzle_eval_file")
    if not path:
        return None
    puzzles = load_puzzles(path)
    return lambda white, black, oriented=False: evaluate_mate_in_one(white, black, puzzles, oriented=oriented)


def resolve_prior_match(config: Dict[str, Any]):
    """`prior_score` evaluation: `prior_eval_games` games per colour (0 = off)
    between the learner and the checkpoint in `opponent_prior`, both sampling
    (`src/eval/matches.py`). Returns None when off."""
    games = int(config.get("prior_eval_games", 0) or 0)
    path = config.get("opponent_prior")
    if games <= 0 or not path:
        return None
    prior, _, oriented, _ = load_models(path)
    if not oriented:
        raise ValueError("opponent_prior must be a shared-network checkpoint")
    prior = freeze(prior)
    seed = int(config.get("seed", 42)) + 3000

    def match(model, oriented=True):
        if not oriented:
            raise ValueError("prior_score needs the shared network")
        was_training = model.training
        model.eval()
        counts = play_match(model, prior, games, seed=seed)
        if was_training:
            model.train()
        s = summarize_match(counts)
        return {"prior_score": s["score"], "prior_wins": s["wins"], "prior_losses": s["losses"], "prior_games": s["games"]}
    return match


def resolve_opponent_pool(config: Dict[str, Any], num_curriculum_envs: int):
    """Frozen-opponent pool (Bansal et al. 2017): the first `opponent_pool_boards`
    game boards after the puzzle and finishing boards play the learner against the
    prior in `opponent_prior` or one of its last `opponent_pool_size` snapshots
    taken every `opponent_snapshot_every` updates. Returns a `PoolBoards` or None
    (`opponent_pool_boards` 0 = today's self-play)."""
    n = int(config.get("opponent_pool_boards", 0) or 0)
    if n == 0:
        return None
    num_envs = int(config.get("num_envs", 12))
    if n < 0 or num_curriculum_envs + n > num_envs:
        raise ValueError(f"opponent_pool_boards must be in [0, {num_envs - num_curriculum_envs}], got {n}")
    if not bool(config.get("shared_network", True)):
        raise ValueError("opponent_pool_boards > 0 requires shared_network")
    size = int(config.get("opponent_pool_size", 4))
    every = int(config.get("opponent_snapshot_every", 50))
    if size < 0 or every <= 0:
        raise ValueError("opponent_pool_size must be >= 0 and opponent_snapshot_every > 0")
    path = config.get("opponent_prior")
    if not path:
        raise ValueError("opponent_pool_boards > 0 requires opponent_prior (the pool is empty before the first snapshot)")
    prior, _, oriented, _ = load_models(path)
    if not oriented:
        raise ValueError("opponent_prior must be a shared-network checkpoint")
    seed = int(config.get("seed", 42))
    pool = OpponentPool(prior, size=size, snapshot_every=every, seed=seed + 4000)
    return PoolBoards(range(num_curriculum_envs, num_curriculum_envs + n), pool, seed=seed + 4001)


def resolve_finish_boards(config: Dict[str, Any], num_puzzle_envs: int):
    """Return (FinishCurriculum or None, num_finish_envs). `finish_boards` boards
    (after the puzzle boards) start k plies before a human checkmate with the winner
    to move and must win under a cap of k + `finish_cap_margin` plies (reverse
    curriculum, Florensa et al. 2017). Depth starts at `finish_depth_start`, rises
    by 2 up to `finish_depth_max` when the success rate over the last
    `finish_window` episodes reaches `finish_advance_rate`."""
    n = int(config.get("finish_boards", 0) or 0)
    if n == 0:
        return None, 0
    num_envs = int(config.get("num_envs", 12))
    if n < 0 or num_puzzle_envs + n > num_envs:
        raise ValueError(f"finish_boards must be in [0, {num_envs - num_puzzle_envs}], got {n}")
    source = config.get("finish_source", "human")
    if source == "technique":
        sets, caps = resolve_technique_sets(config)
        return TechniqueSampler(sets, caps, seed=int(config.get("seed", 42)) + 2000), n
    if source != "human":
        raise ValueError(f"finish_source must be 'human' or 'technique', got {source!r}")
    path = config.get("finish_train_file")
    if not path:
        raise ValueError("finish_boards > 0 requires finish_train_file")
    try:
        sampler = FinishCurriculum(
            load_finishes(path),
            depth_start=int(config.get("finish_depth_start", 2)),
            depth_max=int(config.get("finish_depth_max", 40)),
            advance_rate=float(config.get("finish_advance_rate", 0.6)),
            window=int(config.get("finish_window", 100)),
            cap_margin=int(config.get("finish_cap_margin", 20)),
            seed=int(config.get("seed", 42)) + 2000,
        )
    except ValueError as e:
        raise ValueError(f"finishing curriculum config: {e}") from e
    return sampler, n


def resolve_technique_sets(config: Dict[str, Any]):
    """`technique_sets`: material labels (strong side's pieces beyond the king, e.g.
    Q, R, RR, QR); `technique_caps`: plies allowed per label (defaults 40/60/40/40,
    about twice the perfect-play mate distance). Returns (sets, caps)."""
    sets = [str(x) for x in (config.get("technique_sets") or ["Q", "R", "RR", "QR"])]
    caps = {**DEFAULT_CAPS, **{str(k): int(v) for k, v in (config.get("technique_caps") or {}).items()}}
    for label in sets:
        if label not in caps:
            raise ValueError(f"technique_caps has no entry for {label!r}")
        if caps[label] < 2:
            raise ValueError(f"technique cap for {label!r} must be >= 2 plies")
    return sets, caps


def resolve_midgame_boards(config: Dict[str, Any], num_puzzle_envs: int):
    """`num_puzzle_envs` counts the puzzle and finishing boards that precede the
    mid-game boards."""
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


def resolve_exploration(config: Dict[str, Any]):
    """Behaviour noise on curriculum boards (AlphaZero root noise carried into the
    PPO rollout): `explore_epsilon` in [0, 1) (0 = off), `explore_alpha` > 0
    (Dirichlet concentration, 0.3 for chess in Silver et al. 2017), `explore_boards`
    `puzzle` (puzzle boards only), `curriculum` (puzzle + finishing) or `all`.
    Returns None when off."""
    eps = float(config.get("explore_epsilon", 0.0) or 0.0)
    if not 0.0 <= eps < 1.0:
        raise ValueError(f"explore_epsilon must be in [0, 1), got {eps}")
    if eps == 0.0:
        return None
    alpha = float(config.get("explore_alpha", 0.3))
    if alpha <= 0:
        raise ValueError(f"explore_alpha must be > 0, got {alpha}")
    boards = config.get("explore_boards", "curriculum")
    if boards not in ("puzzle", "curriculum", "all"):
        raise ValueError(f"explore_boards must be 'puzzle', 'curriculum' or 'all', got {boards!r}")
    return {"epsilon": eps, "alpha": alpha, "boards": boards}


def resolve_guide(config: Dict[str, Any]):
    """Label-guided exploration: `guide_epsilon` in [0, 1) (0 = off), `guide_boards`
    `puzzle` | `curriculum` | `all`. The demonstrator is the puzzle key move, the
    human line on finishing boards, or a rules-engine mate. Mutually exclusive with
    `explore_epsilon` > 0."""
    eps = float(config.get("guide_epsilon", 0.0) or 0.0)
    if not 0.0 <= eps < 1.0:
        raise ValueError(f"guide_epsilon must be in [0, 1), got {eps}")
    if eps == 0.0:
        return None
    if float(config.get("explore_epsilon", 0.0) or 0.0) > 0:
        raise ValueError("guide_epsilon and explore_epsilon cannot both be > 0")
    boards = config.get("guide_boards", "curriculum")
    if boards not in ("puzzle", "curriculum", "all"):
        raise ValueError(f"guide_boards must be 'puzzle', 'curriculum' or 'all', got {boards!r}")
    return {"epsilon": eps, "boards": boards}


def resolve_layout_schedule(config: Dict[str, Any], num_puzzle_envs: int):
    """`layout_schedule: [{from_update, finish_boards, midgame_boards}, ...]`: board
    roles after the puzzle boards switch at those updates (boards beyond the three
    groups start from the opening). Validated against num_envs; empty by default."""
    schedule = config.get("layout_schedule") or []
    num_envs = int(config.get("num_envs", 12))
    out = []
    for entry in schedule:
        f, m, u = int(entry.get("finish_boards", 0)), int(entry.get("midgame_boards", 0)), int(entry["from_update"])
        if u < 1 or f < 0 or m < 0 or num_puzzle_envs + f + m > num_envs:
            raise ValueError(f"layout_schedule entry {entry} does not fit num_envs={num_envs} "
                             f"with {num_puzzle_envs} puzzle boards or has from_update < 1")
        if f > 0 and not config.get("finish_train_file"):
            raise ValueError("layout_schedule with finish_boards > 0 requires finish_train_file")
        if m > 0 and not config.get("game_start_file"):
            raise ValueError("layout_schedule with midgame_boards > 0 requires game_start_file")
        out.append({"from_update": u, "finish_boards": f, "midgame_boards": m})
    return out


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
                 game_start_sampler=None, num_midgame_envs=0, finish_sampler=None, num_finish_envs=0):
    if env_type == "async":
        return OpenSpielAsyncVectorEnv(num_envs, start_sampler, num_puzzle_envs, max_plies,
                                       game_start_sampler, num_midgame_envs, finish_sampler, num_finish_envs)
    elif env_type == "sync":
        return OpenSpielVectorEnv(num_envs, start_sampler, num_puzzle_envs, max_plies,
                                  game_start_sampler, num_midgame_envs, finish_sampler, num_finish_envs)
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
    finish_sampler, num_finish_envs = resolve_finish_boards(config, num_puzzle_envs)
    game_start_sampler, num_midgame_envs = resolve_midgame_boards(config, num_puzzle_envs + num_finish_envs)
    layout_schedule = resolve_layout_schedule(config, num_puzzle_envs)
    explore = resolve_exploration(config)
    guide = resolve_guide(config)
    pool = resolve_opponent_pool(config, num_puzzle_envs + num_finish_envs)
    if any(e["finish_boards"] > 0 for e in layout_schedule) and finish_sampler is None:
        finish_sampler = resolve_finish_boards({**config, "finish_boards": 1}, num_puzzle_envs)[0]
    if any(e["midgame_boards"] > 0 for e in layout_schedule) and game_start_sampler is None:
        game_start_sampler = resolve_midgame_boards({**config, "midgame_boards": 1}, num_puzzle_envs)[0]

    set_seed(seed)

    envs = _create_envs(env_type, num_envs, start_sampler, num_puzzle_envs, max_plies,
                        game_start_sampler, num_midgame_envs, finish_sampler, num_finish_envs)
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
        layout_schedule=layout_schedule,
        explore=explore,
        guide=guide,
        pool=pool,
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
