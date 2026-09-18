#!/bin/zsh
# Arm GSL (owner, 2026-09-18: "okay so 1 and 2 is next arm"): good-starts bucket
# curriculum (Florensa 2017) + gae_lambda 1.0. Same start (CURSIL_LONG), budget (120),
# queen-only technique layout as QONLY, which is the exact control (levels + lambda 0.95).
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/endgame-technique/GSL
mkdir -p $OUT
echo "=== GSL start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT --set init_from=experiments/endgame-technique/CURSIL_LONG max_updates=120 guide_epsilon=0.25 "technique_sets=[Q]" technique_curriculum=good_starts gae_lambda=1.0 2>&1 \
  | grep -E "done in|Traceback|Error|curriculum" | cut -c1-400 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
echo "=== GSL all done $(date +%H:%M:%S)"
