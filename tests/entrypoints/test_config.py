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


# --- mid-game boards ----------------------------------------------------------------

from src.entrypoints.train import resolve_midgame_boards  # noqa: E402


def test_midgame_boards_resolver(tmp_path):
    assert resolve_midgame_boards({}, 4) == (None, 0)
    path = tmp_path / "mid.csv"; path.write_text("fen\nrnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n")
    sampler, n = resolve_midgame_boards({"midgame_boards": 4, "game_start_file": str(path), "num_envs": 12}, 4)
    assert n == 4 and sampler.sample().startswith("rnbqkbnr")
    with pytest.raises(ValueError):
        resolve_midgame_boards({"midgame_boards": 9, "game_start_file": str(path), "num_envs": 12}, 4)
    with pytest.raises(ValueError):
        resolve_midgame_boards({"midgame_boards": 2}, 4)


from src.entrypoints.train import resolve_finish_boards  # noqa: E402

_FINISH_CSV = "game_id,anchor_fen,moves,winner\ng1,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,e2e4 e7e5 f1c4 b8c6 d1h5 g8f6 h5f7,white\n"


def test_finish_boards_resolver_and_layout(tmp_path):
    f = tmp_path / "fin.csv"
    f.write_text(_FINISH_CSV)
    assert resolve_finish_boards({}, 4) == (None, 0)
    sampler, n = resolve_finish_boards({"num_envs": 12, "finish_boards": 4, "finish_train_file": str(f),
                                        "finish_depth_start": 2, "finish_depth_max": 40}, 4)
    assert n == 4 and sampler.current_depth == 2 and sampler.cap_margin == 20
    # puzzle 4 + finish 4 leaves 4 boards; mid-game 4 fits, 5 does not
    assert resolve_midgame_boards({"num_envs": 12, "midgame_boards": 4, "game_start_file": "data/games/midgame_starts.csv"}, 8)[1] == 4
    with pytest.raises(ValueError, match=r"\[0, 4\]"):
        resolve_midgame_boards({"num_envs": 12, "midgame_boards": 5, "game_start_file": "x"}, 8)
    with pytest.raises(ValueError, match=r"\[0, 8\]"):
        resolve_finish_boards({"num_envs": 12, "finish_boards": 9, "finish_train_file": str(f)}, 4)
    with pytest.raises(ValueError, match="finish_train_file"):
        resolve_finish_boards({"num_envs": 12, "finish_boards": 1}, 4)
    with pytest.raises(ValueError, match="advance_rate"):
        resolve_finish_boards({"num_envs": 12, "finish_boards": 1, "finish_train_file": str(f), "finish_advance_rate": 0}, 4)


from src.entrypoints.train import resolve_layout_schedule  # noqa: E402


def test_layout_schedule_resolver():
    assert resolve_layout_schedule({}, 4) == []
    cfg = {"num_envs": 12, "finish_train_file": "f.csv", "game_start_file": "g.csv",
           "layout_schedule": [{"from_update": 200, "finish_boards": 4, "midgame_boards": 2}]}
    assert resolve_layout_schedule(cfg, 4) == [{"from_update": 200, "finish_boards": 4, "midgame_boards": 2}]
    with pytest.raises(ValueError):
        resolve_layout_schedule({**cfg, "layout_schedule": [{"from_update": 200, "finish_boards": 7, "midgame_boards": 2}]}, 4)
    with pytest.raises(ValueError, match="finish_train_file"):
        resolve_layout_schedule({"num_envs": 12, "layout_schedule": [{"from_update": 1, "finish_boards": 1}]}, 4)


from src.entrypoints.train import resolve_exploration  # noqa: E402


def test_exploration_resolver():
    assert resolve_exploration({}) is None
    assert resolve_exploration({"explore_epsilon": 0.25}) == {"epsilon": 0.25, "alpha": 0.3, "boards": "curriculum"}
    assert resolve_exploration({"explore_epsilon": 0.25, "explore_boards": "puzzle"})["boards"] == "puzzle"
    for bad in ({"explore_epsilon": 1.0}, {"explore_epsilon": 0.2, "explore_alpha": 0},
                {"explore_epsilon": 0.2, "explore_boards": "puzzles"}):
        with pytest.raises(ValueError):
            resolve_exploration(bad)


from src.entrypoints.train import resolve_guide  # noqa: E402


def test_guide_resolver():
    assert resolve_guide({}) is None
    assert resolve_guide({"guide_epsilon": 0.25}) == {"epsilon": 0.25, "boards": "curriculum"}
    with pytest.raises(ValueError, match="both"):
        resolve_guide({"guide_epsilon": 0.25, "explore_epsilon": 0.1})
    with pytest.raises(ValueError):
        resolve_guide({"guide_epsilon": 1.0})


from src.entrypoints.train import resolve_opponent_pool, resolve_prior_match  # noqa: E402


def test_opponent_pool_resolver(tmp_path):
    import torch
    from src.model.chess_model import ChessPolicyProbs
    from src.model.checkpoints import save_model
    save_model(ChessPolicyProbs().eval(), tmp_path, updates=1)
    assert resolve_opponent_pool({"opponent_pool_boards": 0}, 6) is None
    with pytest.raises(ValueError):                                  # no prior
        resolve_opponent_pool({"num_envs": 16, "opponent_pool_boards": 5}, 6)
    with pytest.raises(ValueError):                                  # does not fit
        resolve_opponent_pool({"num_envs": 16, "opponent_pool_boards": 11, "opponent_prior": str(tmp_path)}, 6)
    pool = resolve_opponent_pool({"num_envs": 16, "opponent_pool_boards": 5, "opponent_prior": str(tmp_path),
                                  "opponent_pool_size": 4, "opponent_snapshot_every": 50}, 6)
    assert pool.board_ids == [6, 7, 8, 9, 10]                        # first game boards after 4 puzzle + 2 finishing
    assert [n for n, _ in pool.pool.members] == ["prior"]
    assert pool.learner_mask(torch.ones(16, dtype=torch.long)).shape == (16,)
    assert resolve_prior_match({"opponent_prior": str(tmp_path), "prior_eval_games": 0}) is None
    assert resolve_prior_match({"opponent_prior": str(tmp_path), "prior_eval_games": 1}) is not None


from src.entrypoints.train import resolve_technique_sets  # noqa: E402


def test_technique_resolver_and_finish_source():
    from src.envs.technique import TechniqueSampler
    sets, caps = resolve_technique_sets({})
    assert sets == ["Q", "R", "RR", "QR"] and caps["R"] == 60
    sets, caps = resolve_technique_sets({"technique_sets": ["Q"], "technique_caps": {"Q": 30}})
    assert caps["Q"] == 30
    with pytest.raises(ValueError):
        resolve_technique_sets({"technique_sets": ["BB"]})
    sampler, n = resolve_finish_boards({"num_envs": 16, "finish_boards": 2, "finish_source": "technique"}, 4)
    assert n == 2 and isinstance(sampler, TechniqueSampler) and sampler.sample().config in sets + ["R", "RR", "QR"]
    with pytest.raises(ValueError):
        resolve_finish_boards({"num_envs": 16, "finish_boards": 2, "finish_source": "tablebase"}, 4)


def test_technique_curriculum_config():
    sampler, _ = resolve_finish_boards({"num_envs": 16, "finish_boards": 4, "finish_source": "technique",
                                        "technique_curriculum": True, "technique_level_start": 1,
                                        "technique_advance_rate": 0.5, "technique_window": 10}, 3)
    assert sampler.curriculum and sampler.levels() == {"Q": 1, "R": 1, "RR": 1, "QR": 1}
    with pytest.raises(ValueError):
        resolve_finish_boards({"num_envs": 16, "finish_boards": 4, "finish_source": "technique",
                               "technique_curriculum": True, "technique_level_start": 7}, 3)


from src.entrypoints.train import resolve_sil  # noqa: E402


def test_sil_resolver():
    assert resolve_sil({"sil_updates": 0}) is None
    out = resolve_sil({"sil_updates": 4, "sil_batch": 256, "sil_buffer": 50000})
    assert out["updates"] == 4 and out["loss_weight"] == 0.1 and out["value_weight"] == 0.01
    with pytest.raises(ValueError):
        resolve_sil({"sil_updates": 4, "sil_batch": 300, "sil_buffer": 100})
    with pytest.raises(ValueError):
        resolve_sil({"sil_updates": 4, "shared_network": False})
