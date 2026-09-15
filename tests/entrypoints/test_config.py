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
