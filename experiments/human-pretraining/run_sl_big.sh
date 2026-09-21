#!/bin/zsh
# SL on the big human set (owner, 2026-09-21: "the actual test will be long run, but we
# should fix data there"). 1.5M positions (5x the 2026-09-16 set, same month / Elo / recipe),
# 2 epochs (the small set overfit mildly by epoch 3; 2 epochs of 5x data is 3.3x the old
# gradient steps), both widths sequentially: 128 (~3 h) then 192 (~6 h). Then puzzles, games,
# and a four-way matrix with the small-data priors. Waits for the data build and the width
# comparison so one training runs at a time. Tomorrow: long RL runs from both priors.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
until grep -q "human_big_train.npz" experiments/human-pretraining/prepare_big.log && grep -q "WIDTH all done" experiments/width/run_compare.log; do sleep 120; done
for NF in 128 192; do
  OUT=experiments/human-pretraining/slbig$NF; mkdir -p $OUT
  echo "=== SLBIG$NF start $(date +%H:%M:%S)"
  $PY -m scripts.pretrain_human --out $OUT --num-filters $NF --epochs 2 --eval-every 500 --eval-limit 2048 \
      --train data/games/human_big_train.npz --eval data/games/human_big_eval.npz 2>&1 | grep -E "final|saved|Traceback|Error"
  $PY -m scripts.eval_checkpoint_puzzles $OUT --out $OUT/puzzles_m1.json 2>&1 | tail -1
  $PY -m scripts.eval_checkpoint_puzzles $OUT --eval data/puzzles/mate_in_2_eval.csv --out $OUT/puzzles_m2.json 2>&1 | tail -1
  $PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
  echo "=== SLBIG$NF done $(date +%H:%M:%S)"
done
$PY -m scripts.eval_matrix SL=experiments/human-pretraining/sl SL192=experiments/human-pretraining/sl192 SLBIG128=experiments/human-pretraining/slbig128 SLBIG192=experiments/human-pretraining/slbig192 --games 30 --seed 5 --workers 1 --out experiments/win-matrix/matrix_slbig.json 2>&1 | grep -v OMP
echo "=== SLBIG all done $(date +%H:%M:%S)"
