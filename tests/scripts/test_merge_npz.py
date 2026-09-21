import numpy as np
from scripts.merge_npz import merge


def test_merge_concatenates_rows(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(a, obs=np.zeros((3, 4), np.uint8), action=np.arange(3)); np.savez(b, obs=np.ones((2, 4), np.uint8), action=np.arange(2))
    shapes = merge([str(a), str(b)], str(tmp_path / "m.npz"))
    m = np.load(tmp_path / "m.npz")
    assert shapes["obs"] == (5, 4) and list(m["action"]) == [0, 1, 2, 0, 1]
