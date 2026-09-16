## 1. Orientation transform

- [x] 1.1 `src/model/orientation.py`: `orient(obs, player)` (batched) and `PLANE_PERMUTATION`; tests: white unchanged, mirrored start positions identical except plane 14, black pawn a7 -> mover pawn a2, castling swap, black transform applied twice is identity
- [x] 1.2 Legal-mask invariance test: for 20 held-out puzzles, build the colour-mirrored FEN with python-chess (`board.mirror()`) and assert OpenSpiel legal masks are identical

## 2. One network in the loop

- [x] 2.1 `src/model/checkpoints.py`: `save_model`, `load_models(dir)` handling single-file and legacy pair; tests for both layouts
- [x] 2.2 `run_chess_training(envs, model, optimizer, ...)`: `models = {WHITE: model, BLACK: model}`, orientation applied in rollout and bootstrap, one backward per update over both sides' samples; keep the legacy two-model path behind `black_model=None` default so the smoke tests for old behaviour still run; tests: one optimizer step per update, bootstrap sign scenario
- [x] 2.3 `train.py`: single model + optimizer, `--save-dir` writes one file; smoke test 2 updates
- [x] 2.4 `evaluate_mate_in_one` and `play_game` take `oriented` and apply `orient` for black to move; tests: perfect/uniform stubs still pass, legacy pair path unchanged

## 3. Scripts and notebook

- [x] 3.1 `run_experiment`, `eval_checkpoint_puzzles`, `eval_games` use the shared loader; notebook cell 2 loads either layout; verify by evaluating the legacy `frac05-onemove` checkpoints and reproducing top-1 0.352

## 4. Measure

- [x] 4.1 Run 500 updates, seed 0, 6 puzzle boards, lr 3e-4 -> `experiments/shared-network/seed0/`; record run.json, held-out curve per colour, wall-clock
- [x] 4.2 40 sampled games (one training seed, owner's decision 2026-09-16 given the time cost; caveat recorded): mate-in-one rate, decisive rate, white/black split; compare with `frac05-onemove` and with the predictions in proposal.md

## 5. Record

- [ ] 5.1 README (architecture, checkpoint name, config), `learning/RESOURCES.md` (AlphaZero methods p.13 quote), `knowledge/` notes, glossary (orientation, weight sharing)
- [ ] 5.2 Comprehension check with owner (CLAUDE.md "Map paper to code": the AlphaZero input description against `orient`; "Review the diff": which planes flip/swap/stay and why the move index does not), code pasted inline; outcome to `learning/records/`
