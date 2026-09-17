from src.eval.matches import score_from_counts, summarize_match


def test_even_match_scenario():
    counts = {"a_white_white_win": 3, "a_white_black_win": 3, "a_white_draw": 14,
              "a_black_black_win": 3, "a_black_white_win": 3, "a_black_draw": 14}
    s = summarize_match(counts)
    assert s["score"] == 0.5 and s["wins"] == 6 and s["losses"] == 6 and s["games"] == 40


def test_score_counts_wins_from_row_side_for_both_colours():
    counts = {"a_white_white_win": 3, "a_white_draw": 1, "a_black_white_win": 2, "a_black_black_win": 2,
              "a_black_unfinished": 2}
    assert score_from_counts(counts) == (3.5 + 3) / 10
