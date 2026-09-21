#!/bin/zsh
# RESTART probe (2026-09-21 evening): does a continuation lose strength because Adam restarts
# from zero? A = slhi256 + 200 updates (lr 1e-4 as LONG256, optimizer saved). B1 = A + 100 with
# the Adam state restored; B2 = A + 100 with a fresh Adam. Same lr (1e-4) in both continuations so
# only the optimizer state differs; KL guard on, alarm off. Then 200-game matches from the opening:
# prior, A, B1, B2.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
PRIOR=experiments/human-pretraining/slhi256
COMMON="opponent_prior=$PRIOR target_kl=0.02 stop_below_prior=null prior_eval_games=40 guide_epsilon=0.25 technique_curriculum=levels finish_source=mixed finish_depth_start=20 finish_depth_max=20 min_lr=0.0001"
A=experiments/width/RESTART_A; mkdir -p $A
echo "=== RESTART_A start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $A --set init_from=$PRIOR max_updates=200 ${=COMMON} 2>&1 | grep -E "^device|done in|Traceback|Error" | cut -c1-300 || true
for ARM in B1 B2; do
  OUT=experiments/width/RESTART_$ARM; mkdir -p $OUT
  if [[ $ARM == B1 ]]; then EXTRA="resume_optimizer=true"; else EXTRA="resume_optimizer=false"; fi
  echo "=== RESTART_$ARM start $(date +%H:%M:%S) ($EXTRA)"
  $PY -m scripts.run_experiment --out $OUT --set init_from=$A max_updates=100 ${=COMMON} ${=EXTRA} 2>&1 | grep -E "^device|optimizer state|done in|Traceback|Error" | cut -c1-300 || true
done
$PY -m scripts.eval_matrix SLHI256=$PRIOR A=$A B1=experiments/width/RESTART_B1 B2=experiments/width/RESTART_B2 --games 100 --seed 11 --workers 1 --out experiments/win-matrix/matrix_restart.json 2>&1 | grep -v OMP
echo "=== RESTART all done $(date +%H:%M:%S)"
