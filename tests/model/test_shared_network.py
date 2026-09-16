import torch

from src.envs.open_spiel_vector_env import OpenSpielVectorEnv
from src.model.chess_model import ChessPolicyProbs
from src.model.checkpoints import load_models, save_model, save_models
from src.model.orientation import orient
from src.model.training import _compute_bootstrap, run_chess_training, WHITE, BLACK
from src.core.types import EnvStep


class CountingAdam(torch.optim.Adam):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


def test_shared_mode_takes_one_optimizer_step_per_update():
    model = ChessPolicyProbs(num_filters=8)
    opt = CountingAdam(model.parameters(), lr=1e-3)
    logs, losses, w, b = run_chess_training(
        OpenSpielVectorEnv(2), model, None, opt, None, episodes=3, steps=2, log_interval=1,
    )
    assert opt.steps == 3
    assert w is b is model
    assert len(losses) == 3


def test_shared_bootstrap_uses_negated_value_of_oriented_black_position():
    model = ChessPolicyProbs(num_filters=8).eval()
    env = OpenSpielVectorEnv(1)
    step = env.reset()
    legal = step.legal_actions_mask[0]
    step = env.step(torch.tensor([int((legal == 1).nonzero()[0])]))   # now black to move
    assert step.current_player[0] == BLACK
    bootstrap_white = _compute_bootstrap(step, model, model, model_id=WHITE, oriented=True)
    with torch.no_grad():
        _, v = model(orient(step.obs, step.current_player), step.legal_actions_mask)
    assert torch.allclose(bootstrap_white, -v.squeeze(-1))


def test_legacy_mode_still_runs_two_models():
    w, b = ChessPolicyProbs(num_filters=8), ChessPolicyProbs(num_filters=8)
    ow, ob = CountingAdam(w.parameters()), CountingAdam(b.parameters())
    run_chess_training(OpenSpielVectorEnv(2), w, b, ow, ob, episodes=2, steps=2)
    assert ow.steps == 2 and ob.steps == 2


def test_checkpoint_single_file_and_legacy_pair(tmp_path):
    m = ChessPolicyProbs()
    save_model(m, tmp_path / "shared", updates=3, timestamp="20260916120000")
    w, b, oriented, paths = load_models(tmp_path / "shared")
    assert oriented and w is b
    assert paths["model"].endswith("model_20260916120000_episodes_3.pth")

    save_models(m, ChessPolicyProbs(), tmp_path / "legacy", updates=3, timestamp="20260916120000")
    w2, b2, oriented2, paths2 = load_models(tmp_path / "legacy")
    assert not oriented2 and w2 is not b2
    assert set(paths2) == {"white", "black"}


def test_config_default_is_shared_and_saves_one_file(tmp_path):
    from src.entrypoints.train import run_training_from_config
    result = run_training_from_config({"num_envs": 1, "max_updates": 1, "steps_per_update": 1, "seed": 0})
    assert result["shared"] and result["model"] is result["white_model"]
    result2 = run_training_from_config({"num_envs": 1, "max_updates": 1, "steps_per_update": 1, "seed": 0,
                                        "shared_network": False})
    assert not result2["shared"] and result2["white_model"] is not result2["black_model"]


def test_init_from_checkpoint_starts_with_saved_weights(tmp_path):
    from src.entrypoints.train import run_training_from_config
    m = ChessPolicyProbs()
    save_model(m, tmp_path, updates=1, timestamp="20260916150000")
    ref = {k: v.clone() for k, v in m.state_dict().items()}
    result = run_training_from_config({"num_envs": 1, "max_updates": 0, "steps_per_update": 1,
                                       "init_from": str(tmp_path)})
    for k, v in result["model"].state_dict().items():
        assert torch.equal(v, ref[k]), k


def test_init_from_layout_mismatch_is_rejected(tmp_path):
    import pytest
    from src.entrypoints.train import run_training_from_config
    save_models(ChessPolicyProbs(), ChessPolicyProbs(), tmp_path, updates=1, timestamp="20260916150000")
    with pytest.raises(ValueError, match="two-network"):
        run_training_from_config({"num_envs": 1, "max_updates": 0, "steps_per_update": 1,
                                  "init_from": str(tmp_path)})
