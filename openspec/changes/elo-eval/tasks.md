## 1. Script
- [x] 1.1 `scripts/eval_elo.py` (ladder games, W/D/L per level, logistic MLE + bootstrap, JSON out). Tests in `tests/scripts/test_eval_elo.py`: the MLE on synthetic scores (0.5 at 1500, 0.24 at 1700 -> ~1500), bound reporting when all levels are lost, engine move -> action mapping on one position (skipped when stockfish is absent).
- [x] 1.2 Dashboard: `elo` column from `elo.json` in an arm folder.
## 2. Runs
- [x] 2.1 Ladder for slhi2_256, LONG256X, SL (old prior), LONG3; record `experiments/elo/outcome.md`.
## 3. Comprehension
- [ ] 3.1 Comprehension check with owner (CLAUDE.md "Trace by hand"): from one level's W/D/L compute the score and the implied Elo difference by the logistic formula, then compare with the script's fit.
