#!/bin/zsh
# Ladder for the recipe arms once run_recipe_arms.sh has finished (the internal matrix alone
# misled us on LONG256X vs its prior; see experiments/elo/outcome.md).
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
until grep -q "RECIPE ARMS all done" experiments/width/run_recipe_arms.log; do sleep 300; done
for ARM in DECAY NOPOOL; do
  echo "=== ELO $ARM start $(date +%H:%M:%S)"
  OMP_NUM_THREADS=4 $PY -m scripts.eval_elo experiments/width/$ARM --games 20 --movetime 0.02 --out experiments/width/$ARM/elo.json 2>&1 | grep -v OMP
done
echo "=== RECIPE ELO all done $(date +%H:%M:%S)"
