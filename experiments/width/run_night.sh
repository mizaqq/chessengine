#!/bin/zsh
# Night chain (owner, 2026-09-21: "when current run finish continue with the first 3").
# After LONG256H: (a) finishing calibration on LONG256G and LONG256H (where does finishing
# fail on the new line: sampled / greedy / human-line attacker at depths 2, 10, 20);
# (b) more high-Elo data: stream 2026-07 (Elo >= 2000, 3M positions), merge with 2026-08 ->
# human_hi2 (~5.7M), pretrain 256x10 for 2 epochs, evaluate, 200-game match vs slhi256;
# (c) if the two-month prior beats slhi256, a guarded 600-update run from it (LONG256X) with
# the standard evals and a 200-game matrix vs LONG256G / LONG256H.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
until grep -q "LONG256H all done\|Traceback" experiments/width/run_long256h.log 2>/dev/null; do sleep 120; done
echo "=== NIGHT start $(date +%H:%M:%S)"
for CK in LONG256G LONG256H; do
  echo "=== calibrate $CK $(date +%H:%M:%S)"
  $PY -m scripts.calibrate_finishes experiments/width/$CK --games 100 --out experiments/finishing-calibration/$CK.json 2>&1 | grep -v OMP | tail -6
done
echo "=== 2026-07 prepare start $(date +%H:%M:%S)"
$PY -m scripts.prepare_games --stream --month 2026-07 --min-elo 2000 --positions 3000000 --prefix human_hi07 --midgame-out midgame_starts_hi07.csv 2>&1 | grep -v OMP | grep -E "scanned|wrote|Traceback|Error"
$PY -m scripts.merge_npz --out data/games/human_hi2_train.npz data/games/human_hi_train.npz data/games/human_hi07_train.npz
$PY -m scripts.merge_npz --out data/games/human_hi2_eval.npz data/games/human_hi_eval.npz data/games/human_hi07_eval.npz
OUT=experiments/human-pretraining/slhi2_256; mkdir -p $OUT
echo "=== SLHI2_256 start $(date +%H:%M:%S)"
$PY -m scripts.pretrain_human --out $OUT --num-filters 256 --epochs 2 --eval-every 500 --eval-limit 2048 --train data/games/human_hi2_train.npz --eval data/games/human_hi2_eval.npz 2>&1 | grep -E "final|saved|Traceback|Error"
$PY -m scripts.eval_checkpoint_puzzles $OUT --out $OUT/puzzles_m1.json 2>&1 | tail -1
$PY -m scripts.eval_checkpoint_puzzles $OUT --eval data/puzzles/mate_in_2_eval.csv --out $OUT/puzzles_m2.json 2>&1 | tail -1
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_matrix SLHI256=experiments/human-pretraining/slhi256 SLHI2_256=$OUT --games 100 --seed 11 --workers 1 --out experiments/win-matrix/matrix_slhi2.json 2>&1 | grep -v OMP
BETTER=$($PY - <<'PY'
import json; d=json.load(open('experiments/win-matrix/matrix_slhi2.json')); print("yes" if d["table"]["SLHI2_256"]["SLHI256"] > 0.5 else "no")
PY
)
echo "=== two-month high-Elo prior better than slhi256: $BETTER"
if [[ $BETTER == yes ]]; then
  PRIOR=$OUT; X=experiments/width/LONG256X; mkdir -p $X
  echo "=== LONG256X start $(date +%H:%M:%S) from $PRIOR"
  $PY -m scripts.run_experiment --out $X --set init_from=$PRIOR opponent_prior=$PRIOR max_updates=600 target_kl=0.02 stop_below_prior=0.5 stop_patience=2 prior_eval_games=40 guide_epsilon=0.25 technique_curriculum=levels finish_source=mixed finish_depth_start=20 finish_depth_max=20 min_lr=0.0001 2>&1 \
    | grep -E "^device|done in|Traceback|Error|below" | cut -c1-300 || true
  $PY -m scripts.eval_games $X --games 40 --seed 0 --out $X/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
  $PY -m scripts.eval_finishes $X --out $X/finishes.json 2>&1 | grep finish_rate
  $PY -m scripts.blunder_audit $X self=$X --games 20 --seed 1 --out experiments/blunder-audit/LONG256X.json 2>&1 | grep -v OMP
  $PY -m scripts.eval_matrix SLHI2_256=$PRIOR LONG256G=experiments/width/LONG256G LONG256H=experiments/width/LONG256H LONG256X=$X --games 100 --seed 11 --workers 1 --out experiments/win-matrix/matrix_long256x.json 2>&1 | grep -v OMP
fi
echo "=== NIGHT all done $(date +%H:%M:%S)"
