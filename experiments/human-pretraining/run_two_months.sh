#!/bin/zsh
# Two-month human set (owner, 2026-09-21: option 2, "more data still"). Download and prepare
# 2014-01 (same Elo floor, 8 positions per game, up to 1.5M positions), merge with the 2013-12
# big set (1.43M) -> human_2m_*, pretrain 192x10 for 2 epochs on the GPU, evaluate, match
# against slbig192. Launch outside the sandbox (download + GPU).
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
echo "=== 2014-01 prepare start $(date +%H:%M:%S)"
$PY -m scripts.prepare_games --month 2014-01 --positions 1500000 --prefix human_2014-01 --midgame-out midgame_starts_2014-01.csv 2>&1 | grep -v OMP | grep -E "downloading|scanned|wrote|Traceback|Error"
echo "=== merge $(date +%H:%M:%S)"
$PY -m scripts.merge_npz --out data/games/human_2m_train.npz data/games/human_big_train.npz data/games/human_2014-01_train.npz
$PY -m scripts.merge_npz --out data/games/human_2m_eval.npz data/games/human_big_eval.npz data/games/human_2014-01_eval.npz
OUT=experiments/human-pretraining/sl2m192; mkdir -p $OUT
echo "=== SL2M192 start $(date +%H:%M:%S)"
$PY -m scripts.pretrain_human --out $OUT --num-filters 192 --epochs 2 --eval-every 500 --eval-limit 2048 --train data/games/human_2m_train.npz --eval data/games/human_2m_eval.npz 2>&1 | grep -E "final|saved|Traceback|Error"
$PY -m scripts.eval_checkpoint_puzzles $OUT --out $OUT/puzzles_m1.json 2>&1 | tail -1
$PY -m scripts.eval_checkpoint_puzzles $OUT --eval data/puzzles/mate_in_2_eval.csv --out $OUT/puzzles_m2.json 2>&1 | tail -1
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_matrix SLBIG192=experiments/human-pretraining/slbig192 SL2M192=$OUT --games 30 --seed 5 --workers 1 --out experiments/win-matrix/matrix_sl2m.json 2>&1 | grep -v OMP
echo "=== TWO MONTHS all done $(date +%H:%M:%S)"
