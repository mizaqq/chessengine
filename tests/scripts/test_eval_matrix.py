from scripts.eval_matrix import build_table, score_from_counts


def test_score_counts_wins_from_row_side_for_both_colours():
    counts = {"a_white_white_win": 3, "a_white_draw": 1, "a_black_white_win": 2, "a_black_black_win": 2,
              "a_black_unfinished": 2}
    # a as white: 3 wins + 1 draw = 3.5 / 4; a as black: 2 losses, 2 wins, 2 half = 3 / 6
    assert score_from_counts(counts) == (3.5 + 3) / 10


def test_table_is_antisymmetric():
    table = build_table(["x", "y"], [("x", "y", {"a_white_white_win": 1, "a_black_white_win": 1})])
    assert table["x"]["y"] == 0.5 and table["y"]["x"] == 0.5
