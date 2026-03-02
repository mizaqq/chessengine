import torch
import matplotlib.pyplot as plt
from datetime import datetime
import json
from src.entrypoints.train import run_training_from_config

if __name__ == "__main__":
    config = {
        "num_envs": 12,
        "max_updates": 500,
        "steps_per_update": 15,
        "learning_rate": 1e-4,
        "seed": 42,
        "lr_decay_interval": 10001,
        "env_type": "sync",
        "terminal_rewards": {"win": 2.0, "loss": -2.0, "draw": -0.5},
        "gamma": 0.99,
        "entropy_coef": 0.01,
        "grad_clip": 1.0,
    }

    result = run_training_from_config(config)

    logs = result["logs"]
    losses = result["losses"]
    white_model = result["white_model"]
    black_model = result["black_model"]

    plt.plot(losses)
    plt.xlabel("update step")
    plt.ylabel("loss")
    plt.savefig("losses.png")
    plt.close()

    with open("logs.json", "w") as f:
        json.dump(logs, f, indent=4)

    episodes = config["max_updates"]
    torch.save(
        white_model.state_dict(),
        f"white_model_{datetime.now().strftime('%Y%m%d%H%M%S')}_episodes_{episodes}.pth",
    )
    torch.save(
        black_model.state_dict(),
        f"black_model_{datetime.now().strftime('%Y%m%d%H%M%S')}_episodes_{episodes}.pth",
    )
    print("Training complete.")
