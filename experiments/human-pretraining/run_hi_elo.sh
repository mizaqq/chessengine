#!/bin/zsh
# High-Elo recent data + 256 filters (owner, 2026-09-21: "newer data", "The next one we will test
# should be 256"). Stream the latest Lichess month, keep games with both players >= 2000, 8
# positions per game, 3M positions; pretrain 256x10 for 2 epochs on the GPU; evaluate; matrix
# against the 192 priors. Waits for the two-month chain (one GPU job at a time).
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
until grep -q "TWO MONTHS all done\|Traceback\|Error" experiments/human-pretraining/run_two_months.log 2>/dev/null; do sleep 120; done
MONTH=${MONTH:-2026-08}
echo "=== HI-ELO prepare start $(date +%H:%M:%S) month $MONTH"
$PY -m scripts.prepare_games --stream --month $MONTH --min-elo 2000 --positions 3000000 --prefix human_hi --midgame-out midgame_starts_hi.csv 2>&1 | grep -v OMP | grep -E "games|scanned|wrote|Traceback|Error"
OUT=experiments/human-pretraining/slhi256; mkdir -p $OUT
echo "=== SLHI256 start $(date +%H:%M:%S)"
$PY -m scripts.pretrain_human --out $OUT --num-filters 256 --epochs 2 --eval-every 500 --eval-limit 2048 --train data/games/human_hi_train.npz --eval data/games/human_hi_eval.npz 2>&1 | grep -E "final|saved|Traceback|Error"
$PY -m scripts.eval_checkpoint_puzzles $OUT --out $OUT/puzzles_m1.json 2>&1 | tail -1
$PY -m scripts.eval_checkpoint_puzzles $OUT --eval data/puzzles/mate_in_2_eval.csv --out $OUT/puzzles_m2.json 2>&1 | tail -1
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
ARGS="SLBIG192=experiments/human-pretraining/slbig192 SLHI256=$OUT"
[[ -d experiments/human-pretraining/sl2m192 ]] && ARGS="$ARGS SL2M192=experiments/human-pretraining/sl2m192"
$PY -m scripts.eval_matrix ${=ARGS} --games 30 --seed 5 --workers 1 --out experiments/win-matrix/matrix_slhi.json 2>&1 | grep -v OMP
echo "=== HI-ELO all done $(date +%H:%M:%S)"
