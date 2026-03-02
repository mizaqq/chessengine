"""Benchmark sync vs async environments over 1000 episodes."""
import time
import json
import torch
from src.entrypoints.train import run_training_from_config

BASE_CONFIG = {
    "num_envs": 12,
    "max_updates": 1000,
    "steps_per_update": 15,
    "learning_rate": 1e-4,
    "seed": 42,
    "lr_decay_interval": 10001,
    "terminal_rewards": {"win": 2.0, "loss": -2.0, "draw": -0.5},
    "gamma": 0.99,
    "entropy_coef": 0.01,
    "grad_clip": 1.0,
}


def run_benchmark(env_type: str):
    config = {**BASE_CONFIG, "env_type": env_type}
    print(f"\n{'='*60}")
    print(f"  Running {env_type.upper()} — 1000 episodes, 12 envs, 15 steps/update")
    print(f"{'='*60}")

    t0 = time.time()
    result = run_training_from_config(config)
    elapsed = time.time() - t0

    logs = result["logs"]
    losses = result["losses"]

    return {
        "env_type": env_type,
        "elapsed_s": elapsed,
        "losses": losses,
        "logs": logs,
        "final_loss": losses[-1] if losses else None,
    }


def analyze(run):
    losses = run["losses"]
    logs = run["logs"]
    n = len(losses)

    first_50 = sum(losses[:50]) / min(50, n)
    last_50 = sum(losses[-50:]) / 50 if n >= 50 else sum(losses) / n

    total_white = sum(l.get("white_wins", 0) for l in logs)
    total_black = sum(l.get("black_wins", 0) for l in logs)
    total_draws = sum(l.get("draws", 0) for l in logs)
    total_games = total_white + total_black + total_draws

    total_illegal = sum(l.get("illegal_samples", 0) for l in logs)
    total_empty = sum(l.get("empty_masks", 0) for l in logs)

    print(f"\n--- {run['env_type'].upper()} Results ---")
    print(f"  Wall time:       {run['elapsed_s']:.1f}s")
    print(f"  Updates:         {n}")
    print(f"  Loss first 50:   {first_50:.4f}")
    print(f"  Loss last 50:    {last_50:.4f}")
    print(f"  Loss delta:      {last_50 - first_50:+.4f}")
    print(f"  Final loss:      {run['final_loss']:.4f}")
    print(f"  Total games:     {total_games}")
    print(f"    White wins:    {total_white} ({100*total_white/max(1,total_games):.1f}%)")
    print(f"    Black wins:    {total_black} ({100*total_black/max(1,total_games):.1f}%)")
    print(f"    Draws:         {total_draws} ({100*total_draws/max(1,total_games):.1f}%)")
    print(f"  Illegal samples: {total_illegal}")
    print(f"  Empty masks:     {total_empty}")

    loss_is_nan = any(torch.isnan(torch.tensor(l)) for l in losses)
    print(f"  NaN losses:      {'YES — PROBLEM!' if loss_is_nan else 'None'}")

    if n >= 100:
        first_quarter = sum(losses[:n//4]) / (n//4)
        last_quarter = sum(losses[-n//4:]) / (n//4)
        improving = last_quarter < first_quarter
        print(f"  Loss trend:      {'DECREASING (good)' if improving else 'NOT DECREASING — investigate'}")
        print(f"    Q1 avg:        {first_quarter:.4f}")
        print(f"    Q4 avg:        {last_quarter:.4f}")

    return {
        "env_type": run["env_type"],
        "elapsed_s": run["elapsed_s"],
        "first_50_loss": first_50,
        "last_50_loss": last_50,
        "total_games": total_games,
        "white_wins": total_white,
        "black_wins": total_black,
        "draws": total_draws,
    }


if __name__ == "__main__":
    sync_run = run_benchmark("sync")
    async_run = run_benchmark("async")

    sync_stats = analyze(sync_run)
    async_stats = analyze(async_run)

    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")
    speedup = sync_stats["elapsed_s"] / max(0.01, async_stats["elapsed_s"])
    print(f"  Async speedup:   {speedup:.2f}x ({sync_stats['elapsed_s']:.1f}s sync vs {async_stats['elapsed_s']:.1f}s async)")
    loss_diff = abs(sync_run["final_loss"] - async_run["final_loss"])
    print(f"  Final loss diff:  {loss_diff:.4f} (sync={sync_run['final_loss']:.4f}, async={async_run['final_loss']:.4f})")

    same_seed_match = loss_diff < 0.01
    print(f"  Same-seed match: {'YES' if same_seed_match else 'NO — expected due to multiprocessing nondeterminism'}")

    with open("benchmark_results.json", "w") as f:
        json.dump({
            "sync": {"losses": sync_run["losses"], "logs": sync_run["logs"], **sync_stats},
            "async": {"losses": async_run["losses"], "logs": async_run["logs"], **async_stats},
        }, f, indent=2)
    print("\nFull results saved to benchmark_results.json")
