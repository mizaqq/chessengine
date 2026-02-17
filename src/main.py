from src.environment.environment import EnvSpawner
from src.model.chess_model import ChessPolicyProbs
from src.model.training import run_chess_training
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import json

if __name__ == "__main__":
    episodes = 30000
    envs = EnvSpawner(12)
    white_model = ChessPolicyProbs()
    black_model = ChessPolicyProbs()
    optimizer_white = torch.optim.Adam(white_model.parameters(), lr=1e-4)
    optimizer_black = torch.optim.Adam(black_model.parameters(), lr=1e-4)
    logs, losses, white_model, black_model = run_chess_training(
        envs,
        white_model,
        black_model,
        optimizer_white,
        optimizer_black,
        steps=15,
        episodes=episodes,
        lr_decay_interval=10001,
    )
    plt.plot(losses)
    plt.xlabel("update step")
    plt.ylabel("loss")
    plt.savefig("losses.png")
    plt.close()
    with open("logs.json", "w") as f:
        json.dump(logs, f, indent=4)
    torch.save(
        white_model.state_dict(),
        f"white_model_{datetime.now().strftime('%Y%m%d%H%M%S')}_episodes_{episodes}.pth",
    )
    torch.save(
        black_model.state_dict(),
        f"black_model_{datetime.now().strftime('%Y%m%d%H%M%S')}_episodes_{episodes}.pth",
    )
    print(
        "Training complete. Model saved to src/models/white_model.pth and src/models/black_model.pth"
    )
