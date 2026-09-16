"""Supervised pretraining on human moves (AlphaGo's first stage, Silver et al. 2016).

Loss per position: -log p(human move) + value_weight * (v - z)^2, where p is the
masked softmax the network already produces and z is the game outcome from the
mover's side. The value weight is small (0.01, the value AlphaGo Zero used for its
supervised comparison) so the value head does not overfit outcomes shared by all
positions of a game; PPO re-fits it later.

Usage: python -m scripts.pretrain_human --out experiments/human-pretraining/sl
         [--train data/games/human_train.npz] [--eval data/games/human_eval.npz]
         [--epochs 3] [--batch 256] [--lr 1e-3] [--value-weight 0.01] [--eval-every 200]
         [--max-batches N]   (smoke runs)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.prepare_games import unpack_mask, unpack_obs
from src.model.chess_model import ChessPolicyProbs
from src.model.checkpoints import save_model


def load(path):
    d = np.load(path)
    return {k: d[k] for k in ("obs", "nop", "mask", "action", "z")}


def batch_tensors(data, idx):
    obs = unpack_obs(data["obs"][idx], data["nop"][idx])
    mask = unpack_mask(data["mask"][idx])
    action = torch.tensor(data["action"][idx].astype(np.int64))
    z = torch.tensor(data["z"][idx])
    return obs, mask, action, z


def loss_terms(model, obs, mask, action, z, value_weight):
    probs, value = model(obs, mask)
    logp = torch.log(probs.gather(1, action[:, None]).squeeze(1) + 1e-8)
    ce = -logp.mean()
    mse = (value.squeeze(-1) - z).pow(2).mean()
    top1 = (probs.argmax(dim=1) == action).float().mean()
    return ce + value_weight * mse, ce, mse, top1


@torch.no_grad()
def evaluate(model, data, batch=512, limit=None):
    model.eval()
    n = len(data["action"]) if limit is None else min(limit, len(data["action"]))
    ce_sum = mse_sum = hits = 0.0
    for start in range(0, n, batch):
        idx = np.arange(start, min(start + batch, n))
        obs, mask, action, z = batch_tensors(data, idx)
        _, ce, mse, top1 = loss_terms(model, obs, mask, action, z, 0.0)
        ce_sum += ce.item() * len(idx); mse_sum += mse.item() * len(idx); hits += top1.item() * len(idx)
    model.train()
    return {"eval_ce": ce_sum / n, "eval_value_mse": mse_sum / n, "eval_top1": hits / n, "eval_n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/games/human_train.npz")
    ap.add_argument("--eval", default="data/games/human_eval.npz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--value-weight", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-limit", type=int, default=None, help="held-out positions per interim eval")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-filters", type=int, default=128)
    args = ap.parse_args()

    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    train, ev = load(args.train), load(args.eval)
    n = len(train["action"])
    print(f"train {n} positions, eval {len(ev['action'])}", flush=True)
    model = ChessPolicyProbs(num_filters=args.num_filters).train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=True)
    log, step, t0 = [], 0, time.time()
    for epoch in range(args.epochs):
        order = rng.permutation(n)
        for start in range(0, n, args.batch):
            idx = order[start:start + args.batch]
            obs, mask, action, z = batch_tensors(train, idx)
            loss, ce, mse, top1 = loss_terms(model, obs, mask, action, z, args.value_weight)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % args.eval_every == 0 or step == 1:
                entry = {"step": step, "epoch": epoch, "train_ce": ce.item(), "train_top1": top1.item(),
                         "train_value_mse": mse.item(), "elapsed_s": time.time() - t0,
                         **evaluate(model, ev, limit=args.eval_limit)}
                log.append(entry)
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in entry.items()}), flush=True)
                (args.out / "pretrain_log.json").write_text(json.dumps(log, indent=1))
            if args.max_batches and step >= args.max_batches:
                break
        if args.max_batches and step >= args.max_batches:
            break
    final = evaluate(model, ev)
    print("final held-out:", json.dumps(final), flush=True)
    path = save_model(model.eval(), args.out, updates=step)
    (args.out / "pretrain_log.json").write_text(json.dumps(log + [{"step": step, "final": final}], indent=1))
    (args.out / "run.json").write_text(json.dumps({"config": vars(args) | {"out": str(args.out)}, "final": final,
                                                     "steps": step, "elapsed_s": time.time() - t0, "checkpoint": str(path)}, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
