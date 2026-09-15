import pytest
from src.entrypoints.train import resolve_shaping


def test_default_is_potential_with_scale_one():
    assert resolve_shaping({}) == 1.0


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
