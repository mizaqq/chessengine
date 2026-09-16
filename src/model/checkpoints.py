"""Save and load checkpoints in both layouts.

Shared network (2026-09-16 on): one file `model_<ts>_episodes_<N>.pth`, used for
both colours with observations oriented to the mover. Legacy: a
`white_model_*.pth` / `black_model_*.pth` pair, each used unoriented for its colour.
"""
import glob
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch

from src.model.chess_model import ChessPolicyProbs


def _stamp(timestamp=None) -> str:
    return timestamp or datetime.now().strftime("%Y%m%d%H%M%S")


def save_model(model, save_dir, updates: int, timestamp: str = None) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"model_{_stamp(timestamp)}_episodes_{updates}.pth"
    torch.save(model.state_dict(), path)
    return path


def save_models(white_model, black_model, save_dir, updates: int, timestamp: str = None):
    """Legacy pair: <color>_model_<timestamp>_episodes_<updates>.pth."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = _stamp(timestamp)
    paths = []
    for name, model in (("white", white_model), ("black", black_model)):
        path = save_dir / f"{name}_model_{ts}_episodes_{updates}.pth"
        torch.save(model.state_dict(), path)
        paths.append(path)
    return paths


def _load(path) -> ChessPolicyProbs:
    m = ChessPolicyProbs()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m


def load_models(ckpt_dir) -> Tuple[ChessPolicyProbs, ChessPolicyProbs, bool, dict]:
    """Return (white_model, black_model, oriented, paths) from the newest checkpoint(s).

    A single `model_*.pth` is returned twice with oriented=True; a legacy pair is
    returned as two models with oriented=False.
    """
    ckpt_dir = Path(ckpt_dir)
    shared = sorted(glob.glob(str(ckpt_dir / "model_*.pth")))
    if shared:
        m = _load(shared[-1])
        return m, m, True, {"model": shared[-1]}
    white = sorted(glob.glob(str(ckpt_dir / "white_model_*.pth")))
    black = sorted(glob.glob(str(ckpt_dir / "black_model_*.pth")))
    if not white or not black:
        raise FileNotFoundError(f"no model_*.pth or white/black pair in {ckpt_dir}")
    return _load(white[-1]), _load(black[-1]), False, {"white": white[-1], "black": black[-1]}
