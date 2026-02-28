from dataclasses import fields
from src.core.types import EnvStep, RolloutBatch


def test_env_step_has_required_fields():
    names = {f.name for f in fields(EnvStep)}
    assert {"obs", "legal_actions_mask", "reward", "done", "info"} <= names


def test_rollout_batch_has_minimum_training_fields():
    names = {f.name for f in fields(RolloutBatch)}
    assert {"obs", "actions", "rewards", "dones", "legal_actions_mask"} <= names
