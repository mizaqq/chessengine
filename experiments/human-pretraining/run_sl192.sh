#!/bin/zsh
# SL192 (owner, 2026-09-21: "lets run a side pretraining with new network"). Same human data
# (299k positions, Elo >= 1500) and recipe as the 2026-09-16 SL run, network 192 filters x 10
# blocks (tower capacity 2.25x, ~2x CPU per step; 256 measured at ~3x and ruled out for RL
# cadence). Then held-out puzzles, games, and a match against the 128-filter prior.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/human-pretraining/sl192
mkdir -p $OUT
echo "=== SL192 start $(date +%H:%M:%S)"
$PY -m scripts.pretrain_human --out $OUT --num-filters 192 --epochs 3 --eval-every 200 --eval-limit 2048 2>&1 | grep -E "final|saved|Traceback|Error"
echo "=== SL192 eval $(date +%H:%M:%S)"
$PY -m scripts.eval_checkpoint_puzzles $OUT --out $OUT/puzzles_m1.json 2>&1 | tail -1
$PY -m scripts.eval_checkpoint_puzzles $OUT --eval data/puzzles/mate_in_2_eval.csv --out $OUT/puzzles_m2.json 2>&1 | tail -1
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_matrix SL=experiments/human-pretraining/sl SL192=$OUT --games 30 --seed 5 --workers 1 --out experiments/win-matrix/matrix_sl192.json 2>&1 | grep -v OMP
echo "=== SL192 all done $(date +%H:%M:%S)"
