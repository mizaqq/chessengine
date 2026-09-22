import shutil
import pytest

from scripts.eval_elo import fit_elo, summarise, expected


def _level(elo, w, d, l):
    return elo, {"win": w, "draw": d, "loss": l}


def test_fit_recovers_rating_from_scores():
    # score 0.5 vs 1500 and 0.24 vs 1700 (40 games each) -> about 1500
    per = dict([_level(1500, 16, 8, 16), _level(1700, 6, 7, 27)])
    s = summarise(per)
    assert 1440 <= s["elo"] <= 1560 and s["bound"] is None
    assert abs(expected(1700, 1500) - 0.24) < 0.02


def test_all_lost_is_a_bound_not_a_number():
    per = dict([_level(1320, 0, 0, 40), _level(1500, 0, 0, 40)])
    s = summarise(per)
    assert s["bound"] and s["bound"].startswith("below 1320")


def test_below_floor_is_flagged_as_extrapolation():
    per = dict([_level(1320, 4, 6, 30), _level(1400, 2, 4, 34)])
    s = summarise(per)
    assert s["elo"] < 1320 and "extrapolated" in s["bound"]


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="stockfish not installed")
def test_engine_game_runs_to_a_result():
    import chess.engine, torch
    from scripts.eval_elo import play_vs_engine

    class Uniform(torch.nn.Module):
        def forward(self, obs, mask):
            return mask / mask.sum(dim=1, keepdim=True), torch.zeros(obs.shape[0], 1)
    with chess.engine.SimpleEngine.popen_uci("stockfish") as eng:
        eng.configure({"UCI_LimitStrength": True, "UCI_Elo": 1320})
        r = play_vs_engine(Uniform(), True, eng, True, 0.01, False, max_plies=60)
    assert r in ("win", "draw", "loss")
