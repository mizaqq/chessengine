#!/bin/zsh
# Owner 2026-09-18: "okay agreed on both". Two arms in sequence, both queen-only, 120
# updates from CURSIL_LONG (QONLY and GSL are the controls):
#   LAMBDA: levels curriculum (level start 1, as QONLY) + gae_lambda 1.0  -> isolates lambda
#   GSL2:   good starts with the warm-start prior + gae_lambda 1.0        -> curriculum fix
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
BASE=experiments/endgame-technique/CURSIL_LONG
for ARM in LAMBDA GSL2; do
  OUT=experiments/endgame-technique/$ARM
  mkdir -p $OUT
  echo "=== $ARM start $(date +%H:%M:%S)"
  if [ "$ARM" = "LAMBDA" ]; then
    EXTRA=(technique_curriculum=levels technique_level_start=1)
  else
    EXTRA=(technique_curriculum=good_starts technique_warm_start=true)
  fi
  $PY -m scripts.run_experiment --out $OUT --set init_from=$BASE max_updates=120 guide_epsilon=0.25 "technique_sets=[Q]" gae_lambda=1.0 $EXTRA 2>&1 \
    | grep -E "done in|Traceback|Error" | cut -c1-300 || true
  $PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
  echo "=== $ARM all done $(date +%H:%M:%S)"
done
