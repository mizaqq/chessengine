#!/bin/zsh
# Prior-anchored self-play arms (change prior-anchored-selfplay, owner 2026-09-22: search-free
# everywhere; "go"). Two guarded 600-update runs from slhi2_256, everything as LONG256X except:
#   KLPRIOR: prior_kl_coef 0.1 -- every minibatch adds 0.1 x KL(pi || prior) (AlphaStar / Ziegler).
#   PFSP:    opponent_pool_sampling pfsp, opponent_pool_size 12 -- pool boards draw the members
#            the learner currently loses to, weight (1 - w)^2 + 0.05 (AlphaStar f_hard).
# Result per arm: the alarm's update or its absence, mean_prior_kl, then the standard evals, the
# Stockfish ladder and a 200-game matrix vs slhi2_256 / LONG256X / DECAY / NOPOOL.
# Waits for the recipe arms so the GPU is not shared.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
PRIOR=experiments/human-pretraining/slhi2_256
until grep -q "RECIPE ARMS all done" experiments/width/run_recipe_arms.log 2>/dev/null; do sleep 300; done
COMMON="init_from=$PRIOR opponent_prior=$PRIOR max_updates=600 target_kl=0.02 stop_below_prior=0.5 stop_patience=2 prior_eval_games=40 guide_epsilon=0.25 technique_curriculum=levels finish_source=mixed finish_depth_start=20 finish_depth_max=20"
for ARM in KLPRIOR PFSP; do
  if [[ $ARM == KLPRIOR ]]; then EXTRA="prior_kl_coef=0.1"; else EXTRA="opponent_pool_sampling=pfsp opponent_pool_size=12"; fi
  OUT=experiments/anchor/$ARM; mkdir -p $OUT
  echo "=== $ARM start $(date +%H:%M:%S) ($EXTRA)"
  $PY -m scripts.run_experiment --out $OUT --set ${=COMMON} ${=EXTRA} 2>&1 | grep -E "^device|done in|Traceback|Error|below" | cut -c1-300 || true
  $PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
  $PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
  $PY -m scripts.blunder_audit $OUT self=$OUT --games 20 --seed 1 --out experiments/blunder-audit/$ARM.json 2>&1 | grep -v OMP
  OMP_NUM_THREADS=4 $PY -m scripts.eval_elo $OUT --games 20 --movetime 0.02 --out $OUT/elo.json 2>&1 | grep -E "^Elo|1320"
  echo "=== $ARM done $(date +%H:%M:%S)"
done
$PY -m scripts.eval_matrix SLHI2_256=$PRIOR LONG256X=experiments/width/LONG256X DECAY=experiments/width/DECAY NOPOOL=experiments/width/NOPOOL KLPRIOR=experiments/anchor/KLPRIOR PFSP=experiments/anchor/PFSP --games 100 --seed 12 --workers 1 --out experiments/win-matrix/matrix_anchor_arms.json 2>&1 | grep -v OMP
echo "=== ANCHOR ARMS all done $(date +%H:%M:%S)"
