import torch

from src.model.chess_model import ChessPolicyProbs
from src.model.device import resolve_device


def test_resolve_device_cpu_and_auto():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "mps")


def test_forward_returns_on_the_callers_device_and_matches_cpu():
    torch.manual_seed(0)
    m = ChessPolicyProbs(num_filters=8, num_blocks=1).eval()
    obs, mask = torch.rand(3, 20, 8, 8), (torch.rand(3, 4674) < 0.05).float()
    mask[:, 0] = 1
    p_cpu, v_cpu = m(obs, mask)
    assert p_cpu.device.type == "cpu" and abs(p_cpu.sum(1) - 1).max() < 1e-5
    if torch.backends.mps.is_available():
        m.to("mps")
        p, v = m(obs, mask)
        assert p.device.type == "cpu" and v.device.type == "cpu"
        assert torch.allclose(p, p_cpu, atol=1e-4) and torch.allclose(v, v_cpu, atol=1e-4)
