from dataclasses import fields
from src.core.types import EnvStep


def test_env_step_has_required_fields():
    names = {f.name for f in fields(EnvStep)}
    assert {"obs", "legal_actions_mask", "reward", "done", "current_player", "info"} <= names
