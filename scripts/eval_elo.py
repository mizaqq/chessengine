"""Elo ladder against rating-limited Stockfish (change elo-eval, 2026-09-22).

Stockfish's `UCI_LimitStrength` / `UCI_Elo` (1320-3190) maps to a skill level "anchored to
the Stash engine ... Skill 0..19 covers CCRL Blitz Elo from 1320 to 3190, approximately"
(Stockfish src/search.h). We play the network against a ladder of levels, both colours, and
fit the logistic Elo model p = 1 / (1 + 10^((E - R) / 400)) (draws = half a win) by maximum
likelihood, with a bootstrap interval. Below the 1320 floor the number is an extrapolation
and is reported as a bound.

    python -m scripts.eval_elo experiments/width/LONG256X --elos 1320 1400 1500 1700 1900 --games 20
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import chess
import chess.engine
import torch

from src.envs.open_spiel_env import OpenSpielEnv
from src.model.checkpoints import load_models
from src.model.orientation import orient_black

WHITE, BLACK = 1, 0
FLOOR, CEIL = 1320, 3190


def network_action(model, env, oriented, greedy):
    player = env.get_current_player()
    obs = torch.tensor(env.state(), dtype=torch.float32).unsqueeze(0)
    if oriented and player == BLACK:
        obs = orient_black(obs)
    mask = env.get_legal_actions().unsqueeze(0)
    with torch.no_grad():
        probs, _ = model(obs, mask)
    probs = probs.squeeze(0)
    return int(probs.argmax()) if greedy else int(torch.multinomial(probs, 1))


def play_vs_engine(model, oriented, engine, network_is_white, movetime, greedy, max_plies=300):
    """One game; returns 'win' / 'draw' / 'loss' from the network's side."""
    env = OpenSpielEnv()
    env.reset(max_plies=max_plies)
    board = chess.Board()
    while not env.is_done():
        player = env.get_current_player()
        if (player == WHITE) == network_is_white:
            action = network_action(model, env, oriented, greedy)
            san = env.env.get_state.action_to_string(player, action)
            board.push_san(san)
        else:
            result = engine.play(board, chess.engine.Limit(time=movetime))
            board.push(result.move)
            ids = env.actions_for_uci([result.move.uci()])
            if not ids:
                raise RuntimeError(f"engine move {result.move.uci()} not found in OpenSpiel actions at {board.fen()}")
            action = next(iter(ids))
        env.step([action])
    res = env.game_result() or "draw"
    if res == "white_win":
        return "win" if network_is_white else "loss"
    if res == "black_win":
        return "loss" if network_is_white else "win"
    return "draw"


def expected(elo_opp, rating):
    return 1.0 / (1.0 + 10 ** ((elo_opp - rating) / 400.0))


def fit_elo(results, lo=400, hi=3400):
    """results: list of (elo_opp, score) per game, score in {0, 0.5, 1}. Grid MLE of R."""
    best, best_ll = None, -math.inf
    for r in range(lo, hi + 1):
        ll = 0.0
        for e, s in results:
            p = min(max(expected(e, r), 1e-9), 1 - 1e-9)
            ll += s * math.log(p) + (1 - s) * math.log(1 - p)
        if ll > best_ll:
            best, best_ll = r, ll
    return best


def bootstrap(results, n=200, seed=0):
    rng = random.Random(seed)
    fits = sorted(fit_elo([rng.choice(results) for _ in results]) for _ in range(n))
    return fits[int(0.025 * n)], fits[int(0.975 * n) - 1]


def summarise(per_level):
    """per_level: {elo: {'win': w, 'draw': d, 'loss': l}} -> estimate dict."""
    results = []
    for e, c in per_level.items():
        results += [(e, 1.0)] * c["win"] + [(e, 0.5)] * c["draw"] + [(e, 0.0)] * c["loss"]
    scores = {e: (c["win"] + 0.5 * c["draw"]) / max(sum(c.values()), 1) for e, c in per_level.items()}
    est = fit_elo(results)
    lo, hi = bootstrap(results)
    all_lost = all(s == 0 for s in scores.values())
    all_won = all(s == 1 for s in scores.values())
    if all_lost:
        bound = f"below {min(per_level)} (every game lost; fit {est} is not informative)"
    elif all_won:
        bound = f"above {max(per_level)} (every game won)"
    elif est < FLOOR:
        bound = f"below the {FLOOR} floor: extrapolated {est}"
    else:
        bound = None
    return {"elo": est, "ci95": [lo, hi], "bound": bound, "scores": scores,
            "games": sum(sum(c.values()) for c in per_level.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", type=Path)
    ap.add_argument("--elos", type=int, nargs="*", default=[1320, 1400, 1500, 1700, 1900])
    ap.add_argument("--games", type=int, default=20, help="games per colour per level")
    ap.add_argument("--movetime", type=float, default=0.05)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--engine", default="stockfish")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    model, _, oriented, paths = load_models(args.ckpt_dir)
    model.eval()
    torch.manual_seed(args.seed)
    per_level = {}
    t0 = time.time()
    with chess.engine.SimpleEngine.popen_uci(args.engine) as engine:
        for elo in args.elos:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": int(max(FLOOR, min(CEIL, elo)))})
            c = {"win": 0, "draw": 0, "loss": 0}
            for g in range(args.games):
                c[play_vs_engine(model, oriented, engine, True, args.movetime, args.greedy)] += 1
                c[play_vs_engine(model, oriented, engine, False, args.movetime, args.greedy)] += 1
            per_level[elo] = c
            print(f"  UCI_Elo {elo}: +{c['win']} ={c['draw']} -{c['loss']}  score {(c['win'] + 0.5 * c['draw']) / (2 * args.games):.2f}  ({time.time() - t0:.0f}s)", flush=True)
    summary = summarise(per_level)
    print(f"Elo estimate {summary['elo']} (95% {summary['ci95'][0]}-{summary['ci95'][1]}, {summary['games']} games)"
          + (f"  [{summary['bound']}]" if summary["bound"] else ""))
    out = {"checkpoint": paths, "engine": args.engine, "movetime": args.movetime, "greedy": args.greedy, "seed": args.seed,
           "per_level": {str(k): v for k, v in per_level.items()}, **summary,
           "scale": "CCRL blitz (Stockfish UCI_Elo anchoring, src/search.h)"}
    out_path = args.out or (args.ckpt_dir / "elo.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
