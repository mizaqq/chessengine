#!/bin/zsh
# Arm CURSIL2 (owner, 2026-09-18: "Ok, agreed on those"). CURSIL with the fixed
# curriculum: levels advance on clean episodes only (no demonstrator move), 0.7 over 40;
# sil_positive_share logged. Same start (POOL_LONG), budget (120) and layout. TECH2 and
# CURSIL are the controls.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/endgame-technique/CURSIL2
BASE=experiments/opponent-pool/POOL_LONG
mkdir -p $OUT
echo "=== CURSIL2 start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT --set init_from=$BASE max_updates=120 guide_epsilon=0.25 2>&1 \
  | grep -E "done in|Traceback|Error" | cut -c1-400 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.probe_finish_values $OUT 2>&1 | grep -v OMP | tee $OUT/probe.txt
echo "=== CURSIL2 all done $(date +%H:%M:%S)"
