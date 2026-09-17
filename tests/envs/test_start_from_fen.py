import chess
import pytest
import torch

from src.envs.open_spiel_env import OpenSpielEnv
from src.envs.open_spiel_vector_env import OpenSpielVectorEnv, _material_np
from src.envs.start_positions import FenSampler, Puzzle, StartPositionSampler

ROOK_MATE = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"   # Ra8# available, white +5
BLACK_MATE = "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"  # Ra1# available, black to move
PUZZLES = [Puzzle("p1", ROOK_MATE, ["a1a8"], 600)]
M2 = "7k/8/5K2/8/8/8/8/1R6 w - - 0 1"                # 1.Kg6! Kg8 2.Rb8#; key move f6g6
M2_PUZZLE = Puzzle("m2", M2, ["f6g6"], 900, mate_in=2)


def _winning_terminal_actions(env: OpenSpielEnv):
    state = env.env.get_state
    player = state.current_player()
    return [a for a in state.legal_actions()
            if state.child(a).is_terminal() and state.child(a).returns()[player] > 0]


def _non_mating_action(env: OpenSpielEnv):
    mates = set(_winning_terminal_actions(env))
    return next(a for a in env.env.get_state.legal_actions() if a not in mates)


# --- single env: reset from FEN -------------------------------------------------

def test_reset_from_rook_mate_position():
    env = OpenSpielEnv()
    env.reset(ROOK_MATE)
    assert env.get_current_player() == 1  # white
    assert _material_np(env.state()) == 5.0
    assert env.get_legal_actions().sum() == 20
    assert len(_winning_terminal_actions(env)) == 1
    assert not env.is_done()


def test_reset_from_fen_black_to_move():
    env = OpenSpielEnv()
    env.reset(BLACK_MATE)
    assert env.get_current_player() == 0
    assert _material_np(env.state()) == -5.0


def test_reset_without_fen_is_opening():
    env = OpenSpielEnv()
    env.reset(ROOK_MATE, puzzle_moves=1)
    env.reset()
    assert env.get_current_player() == 1
    assert _material_np(env.state()) == 0.0
    assert env.get_legal_actions().sum() == 20
    assert not env.is_puzzle


def test_playing_the_mate_ends_the_game_as_white_win():
    env = OpenSpielEnv()
    env.reset(ROOK_MATE)
    (mate,) = _winning_terminal_actions(env)
    env.step(mate)
    assert env.is_done()
    assert env.game_result() == "white_win"


# --- single env: one-move puzzle episodes -----------------------------------------

def test_one_move_episode_mate_found():
    env = OpenSpielEnv()
    env.reset(ROOK_MATE, puzzle_moves=1)
    (mate,) = _winning_terminal_actions(env)
    env.step(mate)
    assert env.is_done() and env.game_result() == "white_win"


def test_one_move_episode_mate_missed():
    env = OpenSpielEnv()
    env.reset(ROOK_MATE, puzzle_moves=1)
    env.step(_non_mating_action(env))
    assert env.is_done()
    assert not env.is_terminal()
    assert env.game_result() == "puzzle_miss"


def test_opening_board_non_mating_move_continues():
    env = OpenSpielEnv()
    env.reset(ROOK_MATE)  # no one_move
    env.step(_non_mating_action(env))
    assert not env.is_done()
    assert env.game_result() is None


# --- single env: mate-in-two episodes and ply cap ------------------------------------

def _action(env, uci):
    (a,) = env.actions_for_uci([uci])
    return a


def test_mate_in_two_solved_ends_after_three_plies_with_win():
    env = OpenSpielEnv()
    env.reset(M2, puzzle_moves=2, key_moves=["f6g6"])
    env.step(_action(env, "f6g6"))
    assert not env.is_done()
    env.step(_action(env, "h8g8"))          # forced reply
    assert not env.is_done()
    env.step(_action(env, "b1b8"))
    assert env.is_done() and env.game_result() == "white_win" and env.plies == 3


def test_mate_in_two_wrong_first_move_ends_at_once_as_miss():
    env = OpenSpielEnv()
    env.reset(M2, puzzle_moves=2, key_moves=["f6g6"])
    env.step(_action(env, "b1b2"))
    assert env.is_done() and not env.is_terminal()
    assert env.game_result() == "puzzle_miss" and env.plies == 1


def test_mate_in_two_right_first_move_then_missed_mate():
    env = OpenSpielEnv()
    env.reset(M2, puzzle_moves=2, key_moves=["f6g6"])
    env.step(_action(env, "f6g6"))
    env.step(_action(env, "h8g8"))
    env.step(_action(env, "b1b2"))          # not mate
    assert env.is_done() and env.game_result() == "puzzle_miss" and env.plies == 3


def test_actions_for_uci_handles_castling_and_promotion():
    env = OpenSpielEnv()
    env.reset("r3k2r/8/8/8/8/8/7P/R3K2R w KQkq - 0 1")
    assert len(env.actions_for_uci(["e1g1", "e1c1"])) == 2
    env.reset("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    assert len(env.actions_for_uci(["a7a8q", "a7a8n"])) == 2


def test_ply_cap_ends_game_as_draw_on_game_board_only():
    env = OpenSpielEnv()
    env.reset(max_plies=4)
    for _ in range(4):
        env.step(int((env.get_legal_actions() == 1).nonzero()[0]))
    assert env.is_done() and not env.is_terminal() and env.game_result() == "draw"
    puzzle = OpenSpielEnv()
    puzzle.reset(M2, puzzle_moves=2, key_moves=["f6g6"], max_plies=1)   # cap ignored on puzzles
    puzzle.step(_action(puzzle, "f6g6"))
    assert not puzzle.is_done()


# --- sync vector env: puzzle boards ------------------------------------------------

def test_vector_env_two_of_four_puzzle_boards():
    env = OpenSpielVectorEnv(4, StartPositionSampler(PUZZLES, seed=0), num_puzzle_envs=2)
    step = env.reset()
    assert step.material.tolist() == [5.0, 5.0, 0.0, 0.0]
    assert (step.current_player == 1).all()


def test_vector_env_zero_puzzle_boards_is_opening():
    env = OpenSpielVectorEnv(2, None, num_puzzle_envs=0)
    step = env.reset()
    assert (step.material == 0.0).all()


def test_vector_env_puzzle_board_auto_resets_after_miss_and_mate():
    env = OpenSpielVectorEnv(2, StartPositionSampler(PUZZLES, seed=0), num_puzzle_envs=2)
    env.reset()
    (mate,) = _winning_terminal_actions(env.envs[0])
    miss = _non_mating_action(env.envs[1])
    step = env.step(torch.tensor([mate, miss]))
    assert step.done.tolist() == [True, True]
    assert step.info["game_results"] == {0: "white_win", 1: "puzzle_miss"}
    assert step.info["puzzle_boards"] == {0: True, 1: True}
    assert step.info["puzzle_depth"] == {0: 1, 1: 1}
    assert (step.material == 5.0).all()  # both boards on a new puzzle


def test_vector_env_mate_in_two_board_reports_depth_and_cap_draws_game_board():
    env = OpenSpielVectorEnv(2, StartPositionSampler([M2_PUZZLE], seed=0), num_puzzle_envs=1, max_plies=2)
    step = env.reset()
    wrong = int((step.legal_actions_mask[0] == 1).nonzero()[0])
    if wrong in env.envs[0].key_actions:
        wrong = int((step.legal_actions_mask[0] == 1).nonzero()[1])
    opening = int((step.legal_actions_mask[1] == 1).nonzero()[0])
    step = env.step(torch.tensor([wrong, opening]))
    assert step.info["game_results"] == {0: "puzzle_miss"} and step.info["puzzle_depth"] == {0: 2}
    opening = int((step.legal_actions_mask[1] == 1).nonzero()[0])
    step = env.step(torch.tensor([wrong, opening]))
    assert step.info["game_results"].get(1) == "draw" and step.info["puzzle_boards"].get(1) is False
    assert step.material[1] == 0.0  # game board reset to the opening


def test_vector_env_opening_board_flagged_false_on_game_end():
    # Board 1 is an opening board; force it into the rook-mate position to end a game.
    env = OpenSpielVectorEnv(2, StartPositionSampler(PUZZLES, seed=0), num_puzzle_envs=1)
    env.reset()
    env.envs[1].reset(ROOK_MATE)
    (mate,) = _winning_terminal_actions(env.envs[1])
    step = env.step(torch.tensor([mate, mate]))
    assert step.info["puzzle_boards"] == {0: True, 1: False}
    assert step.material.tolist() == [5.0, 0.0]


def test_vector_env_rejects_puzzle_boards_without_sampler():
    import pytest
    with pytest.raises(ValueError):
        OpenSpielVectorEnv(2, None, num_puzzle_envs=1)


# --- async vector env ------------------------------------------------------------------

def test_async_env_puzzle_boards_and_auto_reset():
    from src.envs.open_spiel_async_vector_env import OpenSpielAsyncVectorEnv
    env = OpenSpielAsyncVectorEnv(3, StartPositionSampler(PUZZLES, seed=0), num_puzzle_envs=2)
    try:
        step = env.reset()
        assert step.material.tolist() == [5.0, 5.0, 0.0]
        probe = OpenSpielEnv()
        probe.reset(ROOK_MATE)
        (mate,) = _winning_terminal_actions(probe)
        miss = _non_mating_action(probe)
        opening_move = int((step.legal_actions_mask[2] == 1).nonzero()[0])
        step = env.step(torch.tensor([mate, miss, opening_move]))
        assert step.done.tolist() == [True, True, False]
        assert step.info["game_results"] == {0: "white_win", 1: "puzzle_miss"}
        assert step.info["puzzle_boards"] == {0: True, 1: True}
        assert step.info["puzzle_depth"] == {0: 1, 1: 1}
        assert step.material.tolist()[:2] == [5.0, 5.0]
    finally:
        env.close()


# --- mid-game boards -------------------------------------------------------------------

MID_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"


def test_midgame_board_starts_from_sampled_fen_and_is_not_a_puzzle():
    env = OpenSpielVectorEnv(3, StartPositionSampler(PUZZLES, seed=0), num_puzzle_envs=1, max_plies=2,
                             game_start_sampler=FenSampler([MID_FEN], seed=0), num_midgame_envs=1)
    step = env.reset()
    assert env.envs[1].env.get_state.to_string().split(" ")[0] == MID_FEN.split(" ")[0]
    assert env.envs[1].is_puzzle is False and env.envs[2].env.get_state.to_string().startswith("rnbqkbnr/pppppppp")
    (mate,) = _winning_terminal_actions(env.envs[0])
    a1 = int((step.legal_actions_mask[1] == 1).nonzero()[0]); a2 = int((step.legal_actions_mask[2] == 1).nonzero()[0])
    step = env.step(torch.tensor([mate, a1, a2]))
    a1 = int((step.legal_actions_mask[1] == 1).nonzero()[0]); a2 = int((step.legal_actions_mask[2] == 1).nonzero()[0])
    step = env.step(torch.tensor([mate, a1, a2]))            # ply cap 2 -> both game boards draw
    assert step.info["game_results"].get(1) == "draw" and step.info["puzzle_boards"].get(1) is False
    assert env.envs[1].env.get_state.to_string().split(" ")[0] == MID_FEN.split(" ")[0]   # reset to a mid-game FEN again


def test_midgame_boards_validation():
    import pytest
    with pytest.raises(ValueError):
        OpenSpielVectorEnv(2, None, 0, None, None, 1)
    with pytest.raises(ValueError):
        OpenSpielVectorEnv(2, StartPositionSampler(PUZZLES), 2, None, FenSampler([MID_FEN]), 1)


def test_async_midgame_board():
    from src.envs.open_spiel_async_vector_env import OpenSpielAsyncVectorEnv
    env = OpenSpielAsyncVectorEnv(2, None, 0, None, FenSampler([MID_FEN], seed=0), 1)
    try:
        step = env.reset()
        assert step.material.tolist() == [0.0, 0.0]
        assert (step.current_player == 1).all()
        # mid-game FEN has 3 pieces developed: its legal-move count differs from the opening's 20
        assert int(step.legal_actions_mask[0].sum()) != 20 and int(step.legal_actions_mask[1].sum()) == 20
    finally:
        env.close()


# --- finishing boards ---------------------------------------------------------
from src.envs.start_positions import WHITE as _W, FinishCurriculum, FinishRecord  # noqa: E402

_SCHOLAR = FinishRecord("g1", chess.STARTING_FEN if False else "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                        ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"], _W)


class _Recording(FinishCurriculum):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.reports = []

    def report(self, depth, success):
        self.reports.append((depth, success))
        super().report(depth, success)


def _finish_env(num_envs=3, depth=0, cap_margin=20, puzzle=0):
    cur = _Recording([_SCHOLAR], depth_start=depth, depth_max=depth, cap_margin=cap_margin, seed=0)
    sampler = StartPositionSampler([Puzzle("p", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", ["a1a8"], 0)]) if puzzle else None
    env = OpenSpielVectorEnv(num_envs, sampler, puzzle, None, None, 0, cur, 1)
    return env, cur


def test_finish_board_layout_between_puzzle_and_midgame_boards():
    cur = FinishCurriculum([_SCHOLAR], depth_start=0, depth_max=0)
    puzzles = StartPositionSampler([Puzzle("p", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", ["a1a8"], 0)])
    env = OpenSpielVectorEnv(12, puzzles, 4, None, FenSampler([chess.STARTING_FEN]), 2, cur, 4)
    assert [env.is_puzzle_env(i) for i in range(12)] == [True] * 4 + [False] * 8
    assert [env.is_finish_env(i) for i in range(12)] == [False] * 4 + [True] * 4 + [False] * 4
    assert [env.is_midgame_env(i) for i in range(12)] == [False] * 8 + [True] * 2 + [False] * 2
    with pytest.raises(ValueError):
        OpenSpielVectorEnv(12, puzzles, 4, None, FenSampler([chess.STARTING_FEN]), 4, cur, 6)
    with pytest.raises(ValueError):
        OpenSpielVectorEnv(4, None, 0, None, None, 0, None, 1)


def test_finish_board_depth_zero_mate_is_a_success():
    env, cur = _finish_env(depth=0)
    step = env.reset()
    assert env.envs[0].get_current_player() == _W and env.envs[0].max_plies == 20
    mate = next(iter(env.envs[0].actions_for_uci(["h5f7"])))
    actions = torch.tensor([mate] + [int(step.legal_actions_mask[i].nonzero()[0]) for i in (1, 2)])
    step = env.step(actions)
    assert step.info["game_results"][0] == "white_win"
    assert step.info["finish_boards"] == {0: True}
    assert step.info["finish_depth"] == {0: 0} and step.info["finish_success"] == {0: True}
    assert cur.reports == [(0, True)]
    assert env.envs[0].plies == 0                         # auto-reset to a new finishing start


def test_finish_board_cap_is_a_draw_and_a_failure():
    env, cur = _finish_env(depth=0, cap_margin=2)
    step = env.reset()
    for uci in ("a2a3", "a7a6"):                         # white quiet move, black quiet move
        quiet = next(iter(env.envs[0].actions_for_uci([uci])))
        actions = torch.tensor([quiet] + [int(step.legal_actions_mask[i].nonzero()[0]) for i in (1, 2)])
        step = env.step(actions)
    assert step.info["game_results"][0] == "draw"
    assert step.info["finish_success"] == {0: False} and cur.reports == [(0, False)]


def test_async_env_rejects_finishing_boards():
    from src.envs.open_spiel_async_vector_env import OpenSpielAsyncVectorEnv
    cur = FinishCurriculum([_SCHOLAR], depth_start=0, depth_max=0)
    with pytest.raises(ValueError, match="sync"):
        OpenSpielAsyncVectorEnv(2, None, 0, None, None, 0, cur, 1)


def test_set_layout_changes_roles_on_next_reset_and_keeps_finished_board_accounting():
    env, cur = _finish_env(num_envs=3, depth=0)
    step = env.reset()
    assert env.is_finish_env(0) and not env.is_finish_env(1)
    env.set_layout(0, 0)                                    # all three boards become opening boards
    assert not env.is_finish_env(0)
    # board 0 still runs its finishing game: its result is still reported as a finish
    mate = next(iter(env.envs[0].actions_for_uci(["h5f7"])))
    step = env.step(torch.tensor([mate] + [int(step.legal_actions_mask[i].nonzero()[0]) for i in (1, 2)]))
    assert step.info["finish_boards"] == {0: True} and cur.reports == [(0, True)]
    assert env.envs[0].plies == 0 and env.envs[0].max_plies is None   # reset as an opening board
    b = chess.Board(env.envs[0].env.get_state.to_string())
    assert b.fen() == chess.STARTING_FEN
    with pytest.raises(ValueError):
        env.set_layout(4, 0)
