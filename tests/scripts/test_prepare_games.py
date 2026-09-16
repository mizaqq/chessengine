import io
import random

import chess
import numpy as np
import torch

from scripts.prepare_games import collect, encode, game_passes, sample_game, unpack_mask, unpack_obs
from src.envs.open_spiel_env import OpenSpielEnv
from src.model.orientation import orient_black

PGN = """[Event "Rated Blitz game"]
[Result "1-0"]
[WhiteElo "1650"]
[BlackElo "1580"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. Nbd2 Bb7 12. Bc2 Re8 1-0

[Event "Rated Blitz game"]
[Result "0-1"]
[WhiteElo "1400"]
[BlackElo "1700"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 h6 7. Bh4 b6 0-1

[Event "Rated Blitz game"]
[Result "1/2-1/2"]
[WhiteElo "1500"]
[BlackElo "1500"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 1/2-1/2
"""


def test_rating_floor_filters_whole_game():
    train, ev, mids = collect(io.StringIO(PGN), min_elo=1500, positions=10_000, per_game=4, eval_frac=0.0, midgame=10, seed=0)
    assert ev == []
    assert {s.game_index for s in train} == {0, 1}           # the 1400 game is skipped
    assert len(train) == 8                                    # 4 per kept game
    assert mids and all(f.split()[0] for f in mids)


def test_positions_of_one_game_share_split():
    rng_hits = {}
    for seed in range(5):
        train, ev, _ = collect(io.StringIO(PGN), 1500, 10_000, 3, eval_frac=0.5, midgame=0, seed=seed)
        for split, samples in (("train", train), ("eval", ev)):
            for s in samples:
                rng_hits.setdefault((seed, s.game_index), set()).add(split)
    assert all(len(v) == 1 for v in rng_hits.values())


def test_sample_outcome_is_from_the_mover():
    game = chess.pgn.read_game(io.StringIO(PGN))            # 1-0
    samples, _ = sample_game(game, 0, per_game=24, rng=random.Random(0))
    assert all((s.z == 1.0) == (s.mover == 1) for s in samples)
    assert all((s.z == -1.0) == (s.mover == 0) for s in samples)


def test_encode_round_trip_matches_engine_view():
    game = chess.pgn.read_game(io.StringIO(PGN))
    samples, _ = sample_game(game, 0, per_game=6, rng=random.Random(1))
    arrays = encode(samples)
    assert arrays["obs"].shape == (len(samples), 160) and arrays["mask"].shape == (len(samples), 585)
    obs, mask = unpack_obs(arrays["obs"], arrays["nop"]), unpack_mask(arrays["mask"])
    env = OpenSpielEnv()
    for i, s in enumerate(samples):
        env.reset(s.fen)
        expected = torch.tensor(env.state(), dtype=torch.float32)
        if s.mover == 0:
            expected = orient_black(expected)
        assert torch.equal(obs[i], expected)
        assert torch.equal(mask[i], env.get_legal_actions())
        assert mask[i, int(arrays["action"][i])] == 1.0           # the human move is legal
        assert env.actions_for_uci([s.move_uci]) == {int(arrays["action"][i])}
