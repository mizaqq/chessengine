## 1. Data
- [x] 1.1 `scripts/prepare_games.py` (download, filter, sample, encode, pack, split, mid-game FENs); tests on a small PGN fixture.
- [ ] 1.2 Build the December 2013 set (300k / 15k) and `midgame_starts.csv`.

## 2. Pretraining
- [x] 2.1 `scripts/pretrain_human.py` with held-out accuracy log and checkpoint; smoke test on the fixture.
- [ ] 2.2 Run 3 epochs; evaluate the checkpoint on puzzles and 40 games.

## 3. Mid-game boards
- [x] 3.1 `FenSampler`; vector envs `game_start_sampler` / `num_midgame_envs`; config resolver; tests.
- [x] 3.2 README and default-config notes.

## 4. RL from the pretrained network
- [ ] 4.1 PPO 120 updates `init_from` the SL checkpoint with 4 mid-game boards; 40 games; summary.
- [ ] 4.2 Outcome in the proposal, notes, session, workshop section.

## 5. Comprehension (in chat or workshop)
- [ ] 5.1 Why 8 positions per game beat 70 (correlated outcomes).
- [ ] 5.2 Read SL-alone vs after-PPO numbers: what RL added and what it narrowed.
