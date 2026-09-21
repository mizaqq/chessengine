#!/bin/zsh
# ENT10 (owner, 2026-09-21: "ok" to arm 1 then 2). LONG256 + 120 updates with entropy_coef 0.1
# (the bonus is on entropy normalised to [0,1]; 0.01-0.03 was a rounding error next to the
# advantage-scaled policy loss). Then LONG256B = LONG256 + 600 updates with the guard if ENT10
# lifted entropy without a puzzle drop, else without it; prior_score is the alarm.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
START=experiments/width/LONG256; PRIOR=experiments/human-pretraining/slhi256
COMMON="opponent_prior=$PRIOR guide_epsilon=0.25 technique_curriculum=levels finish_source=mixed finish_depth_start=20 finish_depth_max=20 min_lr=0.0001"
OUT=experiments/width/ENT10; mkdir -p $OUT
echo "=== ENT10 start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT --set init_from=$START max_updates=120 entropy_coef=0.1 ${=COMMON} 2>&1 | grep -E "^device|done in|Traceback|Error" | cut -c1-300 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.eval_matrix LONG256=$START ENT10=$OUT --games 30 --seed 5 --workers 1 --out experiments/win-matrix/matrix_ent10.json 2>&1 | grep -v OMP
echo "=== ENT10 done $(date +%H:%M:%S)"
# decide the guard for the long continuation: entropy up by >= 0.03 and mate-in-1 not down by > 0.01
GUARD=$($PY - <<'PY'
import json
a=json.load(open('experiments/width/LONG256/run.json'))['logs'][-1]; b=json.load(open('experiments/width/ENT10/run.json'))['logs'][-1]
ok = b['mean_entropy_normalized_game'] >= a['mean_entropy_normalized_game'] + 0.03 and b['lichess_top1'] >= a['lichess_top1'] - 0.01
print("entropy_coef=0.1" if ok else "")
PY
)
OUT=experiments/width/LONG256B; mkdir -p $OUT
echo "=== LONG256B start $(date +%H:%M:%S) guard='$GUARD'"
$PY -m scripts.run_experiment --out $OUT --set init_from=$START max_updates=600 ${=COMMON} ${=GUARD} 2>&1 | grep -E "^device|done in|Traceback|Error" | cut -c1-300 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.blunder_audit $OUT self=$OUT --games 20 --seed 1 --out experiments/blunder-audit/LONG256B.json 2>&1 | grep -v OMP
$PY -m scripts.eval_matrix SLHI256=$PRIOR LONG256=$START ENT10=experiments/width/ENT10 LONG256B=$OUT --games 30 --seed 5 --workers 1 --out experiments/win-matrix/matrix_long256b.json 2>&1 | grep -v OMP
echo "=== ENT10+LONG256B all done $(date +%H:%M:%S)"
