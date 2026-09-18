import pytest

from src.envs.finish_mix import MixedFinishSampler
from src.envs.start_positions import FinishStart, WHITE
from src.entrypoints.train import resolve_finish_boards


class Human:
    current_depth = 4
    def __init__(self): self.reports = []
    def sample(self): return FinishStart("7k/8/8/8/8/8/8/K6R w - - 0 1", 2, 22, WHITE, ("h1h8",))
    def report(self, d, s): self.reports.append((d, s))


class Tech:
    current_depth = 0
    def __init__(self): self.reports = []
    def sample(self): return FinishStart("7k/8/8/8/8/8/1Q6/K7 w - - 0 1", 0, 40, WHITE, (), "Q")
    def report(self, d, s): pass
    def report_label(self, label, success, clean=True, depth=None): self.reports.append((label, success, clean, depth))
    def dials(self): return {"technique_level_Q": 0}
    def state(self): return {"mode": "levels", "levels": {"Q": 0}}
    def load_state(self, st): self.loaded = st


def test_mix_draws_by_weight_and_routes_reports():
    h, t = Human(), Tech()
    m = MixedFinishSampler([t, h], [0.5, 0.5], seed=0)
    kinds = {bool(m.sample().config) for _ in range(40)}
    assert kinds == {True, False}
    m.report(2, True); m.report_label("Q", False, clean=True, depth=0)
    assert h.reports == [(2, True)] and t.reports == [("Q", False, True, 0)]
    assert m.current_depth == 4 and m.dials() == {"technique_level_Q": 0}
    st = m.state(); m.load_state(st)
    assert t.loaded == {"mode": "levels", "levels": {"Q": 0}}
    with pytest.raises(ValueError):
        MixedFinishSampler([t], [0.0], seed=0)


def test_mixed_source_resolves():
    sampler, n = resolve_finish_boards({"num_envs": 16, "finish_boards": 4, "finish_source": "mixed",
                                        "finish_train_file": "data/finishes/finish_train.csv",
                                        "finish_mix": {"technique": 0.5, "human": 0.5}}, 3)
    assert n == 4 and isinstance(sampler, MixedFinishSampler)
    kinds = {bool(sampler.sample().config) for _ in range(30)}
    assert kinds == {True, False}
