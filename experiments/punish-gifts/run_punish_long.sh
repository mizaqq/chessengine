#!/bin/zsh
# PUNISH_LONG (owner, 2026-09-19: "make the fix and run a longer run"). DEPTH20 + 600 updates
# with the punisher at 0.5 on every board, mates before captures (fix after arm PUNISH).
# Everything else as DEPTH20 (mixed finishing slot, human depth uniform over [0,20], lambda 1,
# LR floor 1e-4, pool, SIL, guidance 0.25). Then games, finishes, blunder audit, win matrix vs
# SL / LONG3 / DEPTH20 / PUNISH. Named long-run exception (~8.5 h).
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
START=experiments/endgame-technique/DEPTH20
OUT=experiments/punish-gifts/PUNISH_LONG
mkdir -p $OUT
echo "=== PUNISH_LONG start $(date +%H:%M:%S) from $START"
$PY -m scripts.run_experiment --out $OUT --set init_from=$START max_updates=600 guide_epsilon=0.25 technique_curriculum=levels finish_source=mixed finish_depth_start=20 finish_depth_max=20 min_lr=0.0001 punish_epsilon=0.5 2>&1 \
  | grep -E "done in|Traceback|Error|curriculum" | cut -c1-300 || true
echo "=== training done $(date +%H:%M:%S)"
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.blunder_audit $OUT self=$OUT SL=experiments/human-pretraining/sl --games 20 --seed 1 --out experiments/blunder-audit/punish_long.json 2>&1 | grep -v OMP
$PY -m scripts.eval_matrix SL=experiments/human-pretraining/sl LONG3=experiments/endgame-technique/LONG3 DEPTH20=$START PUNISH=experiments/punish-gifts/PUNISH PUNISH_LONG=$OUT --games 30 --seed 5 --workers 1 --out experiments/win-matrix/matrix_punish_long.json 2>&1 | grep -v OMP
echo "=== PUNISH_LONG all done $(date +%H:%M:%S)"
