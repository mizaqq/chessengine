import torch
import math
from torch.testing import assert_close


def test_normalized_entropy_uniform_distribution():
    """Uniform distribution over N actions has normalized entropy = 1.0."""
    num_legal = 20.0
    raw_entropy = math.log(num_legal)
    log_legal = max(math.log(num_legal), 1e-8)
    normalized = raw_entropy / log_legal
    assert_close(torch.tensor(normalized), torch.tensor(1.0))


def test_normalized_entropy_forced_move_no_nan():
    """Forced move (1 legal action) must not produce inf or NaN."""
    raw_entropy = torch.tensor(0.0)
    num_legal = torch.tensor(1.0)
    log_legal = torch.clamp(torch.log(num_legal), min=1e-8)
    normalized = raw_entropy / log_legal
    assert torch.isfinite(normalized), f"Got {normalized}"
    assert_close(normalized, torch.tensor(0.0))


def test_normalized_entropy_batch():
    """Vectorized normalization across a batch."""
    raw = torch.tensor([2.0, 0.5, 0.0])
    num_legal = torch.tensor([10.0, 5.0, 1.0])
    log_legal = torch.clamp(torch.log(num_legal), min=1e-8)
    normalized = raw / log_legal
    assert torch.all(torch.isfinite(normalized))
    assert_close(normalized[0], torch.tensor(2.0 / math.log(10.0)))
    assert_close(normalized[1], torch.tensor(0.5 / math.log(5.0)))
    assert_close(normalized[2], torch.tensor(0.0))
