from scripts.eval_matrix import build_table


def test_table_is_antisymmetric():
    table = build_table(["x", "y"], [("x", "y", {"a_white_white_win": 1, "a_black_white_win": 1})])
    assert table["x"]["y"] == 0.5 and table["y"]["x"] == 0.5
