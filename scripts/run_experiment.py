"""Run training from a YAML config with overrides; save checkpoints and run.json.

Usage: python -m scripts.run_experiment --out experiments/<name>/<arm> [--config ...] [--set key=value ...]
"""
import argparse
import json
import time
from pathlib import Path

import yaml

from src.entrypoints.train import run_training_from_config, save_models


def parse_value(raw: str):
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/configs/train_default.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    for item in args.set:
        key, raw = item.split("=", 1)
        config[key] = parse_value(raw)

    start = time.time()
    result = run_training_from_config(config)
    elapsed = time.time() - start

    args.out.mkdir(parents=True, exist_ok=True)
    paths = save_models(result["white_model"], result["black_model"], args.out, config["max_updates"])
    run = {
        "config": config,
        "elapsed_s": elapsed,
        "losses": result["losses"],
        "logs": result["logs"],
        "checkpoints": [str(p) for p in paths],
    }
    (args.out / "run.json").write_text(json.dumps(run, indent=1))
    last = result["logs"][-1] if result["logs"] else {}
    print(f"done in {elapsed:.0f}s; last log: {json.dumps(last)}")


if __name__ == "__main__":
    main()
