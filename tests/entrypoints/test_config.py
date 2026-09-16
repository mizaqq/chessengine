import pytest
from src.entrypoints.train import resolve_shaping


def test_default_is_potential_with_scale_point_two():
    assert resolve_shaping({}) == 0.2


def test_explicit_potential_and_scale():
    assert resolve_shaping({"shaping": "potential", "shaping_scale": 0.25}) == 0.25


def test_none_resolves_to_zero_scale():
    assert resolve_shaping({"shaping": "none", "shaping_scale": 3.0}) == 0.0


def test_unknown_shaping_lists_accepted_values():
    with pytest.raises(ValueError) as exc:
        resolve_shaping({"shaping": "material_diff"})
    msg = str(exc.value)
    assert "shaping" in msg and "potential" in msg and "none" in msg


def test_negative_scale_rejected():
    with pytest.raises(ValueError) as exc:
        resolve_shaping({"shaping_scale": -1.0})
    assert "shaping_scale" in str(exc.value)


def test_save_models_writes_two_checkpoints(tmp_path):
    import torch
    from src.entrypoints.train import save_models
    from src.model.chess_model import ChessPolicyProbs

    w, b = ChessPolicyProbs(num_filters=8), ChessPolicyProbs(num_filters=8)
    paths = save_models(w, b, tmp_path, updates=7, timestamp="20260915120000")
    assert sorted(p.name for p in paths) == [
        "black_model_20260915120000_episodes_7.pth",
        "white_model_20260915120000_episodes_7.pth",
    ]
    loaded = torch.load(paths[0], map_location="cpu")
    assert set(loaded.keys()) == set(w.state_dict().keys())


FIXTURE = """puzzle_id,fen,mating_moves,rating
p1,6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1,a1a8,600
"""


def test_puzzle_fraction_default_zero_gives_no_boards():
    from src.entrypoints.train import resolve_puzzle_boards
    assert resolve_puzzle_boards({}) == (None, 0)


def test_puzzle_fraction_out_of_range():
    from src.entrypoints.train import resolve_puzzle_boards
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        resolve_puzzle_boards({"puzzle_fraction": 1.5})


def test_puzzle_fraction_without_file():
    from src.entrypoints.train import resolve_puzzle_boards
    with pytest.raises(ValueError, match="puzzle_train_file"):
        resolve_puzzle_boards({"puzzle_fraction": 0.5})


def test_puzzle_fraction_rounds_to_board_count(tmp_path):
    from src.entrypoints.train import resolve_puzzle_boards
    path = tmp_path / "p.csv"
    path.write_text(FIXTURE)
    sampler, n = resolve_puzzle_boards(
        {"puzzle_fraction": 0.5, "num_envs": 12, "puzzle_train_file": str(path), "seed": 3}
    )
    assert n == 6
    assert sampler.sample().fen.startswith("6k1/")


def test_eval_interval_must_be_positive():
    from src.entrypoints.train import resolve_eval_interval
    assert resolve_eval_interval({}) == 50
    with pytest.raises(ValueError, match="eval_interval"):
        resolve_eval_interval({"eval_interval": 0})


def test_puzzle_boards_overrides_fraction(tmp_path):
    from src.entrypoints.train import resolve_puzzle_boards
    path = tmp_path / "p.csv"; path.write_text(FIXTURE)
    _, n = resolve_puzzle_boards({"puzzle_fraction": 0.5, "puzzle_boards": 4, "num_envs": 12,
                                  "puzzle_train_file": str(path)})
    assert n == 4


def test_puzzle_boards_out_of_range():
    from src.entrypoints.train import resolve_puzzle_boards
    with pytest.raises(ValueError, match=r"puzzle_boards"):
        resolve_puzzle_boards({"puzzle_boards": 13, "num_envs": 12})


def test_puzzle_train_files_builds_mixed_sampler(tmp_path):
    from src.entrypoints.train import resolve_puzzle_boards
    from src.envs.start_positions import MixedSampler
    a = tmp_path / "a.csv"; a.write_text(FIXTURE)
    b = tmp_path / "b.csv"; b.write_text(FIXTURE.replace("p1", "p2"))
    sampler, n = resolve_puzzle_boards({"puzzle_boards": 2, "num_envs": 4,
                                        "puzzle_train_files": [{"file": str(a), "weight": 0.5},
                                                               {"file": str(b), "weight": 0.5}]})
    assert n == 2 and isinstance(sampler, MixedSampler)
    assert sampler.sample().fen.startswith("6k1/")


def test_named_eval_sets_prefix_keys(tmp_path):
    from src.entrypoints.train import resolve_eval_fn
    from src.model.chess_model import ChessPolicyProbs
    a = tmp_path / "a.csv"; a.write_text(FIXTURE)
    fn = resolve_eval_fn({"puzzle_eval_files": {"lichess": str(a), "selfplay": str(a)}})
    m = ChessPolicyProbs(num_filters=8)
    out = fn(m, m, oriented=True)
    assert {"lichess_top1", "lichess_mate_prob", "lichess_count", "selfplay_top1"} <= set(out)
    assert out["lichess_count"] == 1


# --- algorithm presets ------------------------------------------------------------

from src.entrypoints.train import resolve_algorithm  # noqa: E402


def test_default_algorithm_is_a2c_preset():
    algo = resolve_algorithm({})
    assert algo == {"epochs": 1, "minibatches": 1, "clip_epsilon": None,
                    "normalize_advantage": False, "value_coef": 1.0, "gae_lambda": 1.0}


def test_ppo_preset_values_from_paper():
    algo = resolve_algorithm({"algorithm": "ppo"})
    assert algo == {"epochs": 4, "minibatches": 4, "clip_epsilon": 0.2,
                    "normalize_advantage": True, "value_coef": 0.5, "gae_lambda": 0.95}


def test_explicit_keys_override_preset():
    algo = resolve_algorithm({"algorithm": "a2c", "gae_lambda": 0.95, "value_coef": 0.5})
    assert algo["gae_lambda"] == 0.95 and algo["value_coef"] == 0.5 and algo["epochs"] == 1


def test_unknown_algorithm_lists_both_names():
    with pytest.raises(ValueError) as e:
        resolve_algorithm({"algorithm": "sac"})
    assert "a2c" in str(e.value) and "ppo" in str(e.value)


@pytest.mark.parametrize("cfg,key", [
    ({"gae_lambda": 1.5}, "gae_lambda"),
    ({"algorithm": "ppo", "clip_epsilon": 0}, "clip_epsilon"),
    ({"algorithm": "ppo", "ppo_epochs": 0}, "ppo_epochs"),
    ({"algorithm": "ppo", "ppo_minibatches": 0}, "ppo_minibatches"),
    ({"value_coef": -1}, "value_coef"),
])
def test_invalid_algorithm_settings_name_the_key(cfg, key):
    with pytest.raises(ValueError) as e:
        resolve_algorithm(cfg)
    assert key in str(e.value)


# --- max_plies --------------------------------------------------------------------

from src.entrypoints.train import resolve_max_plies  # noqa: E402


def test_max_plies_default_none_and_values():
    assert resolve_max_plies({}) is None
    assert resolve_max_plies({"max_plies": None}) is None
    assert resolve_max_plies({"max_plies": 120}) == 120
    with pytest.raises(ValueError):
        resolve_max_plies({"max_plies": 1})
