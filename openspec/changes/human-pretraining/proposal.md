## Why

After PPO, the curriculum and the ply cap, the network solves puzzles (mate-in-one
0.44, mate-in-two 0.35 held-out) but its games are still 190-ply draws with few
mating chances: model-free self-play from the opening explores blindly. The owner
chose the model-free road with better data before any search: (1) supervised
pretraining on human games, the first stage of AlphaGo (Silver et al. 2016: the RL
policy's "weights rho are initialized to the same values" as the supervised
network, which had learned to predict human moves at 57%); (2) mid-game start
positions for game boards, the start-state curriculum applied to whole games.
Search (AlphaZero) is deferred by the owner's decision.

## Decisions taken in conversation (2026-09-16 night)

- Owner: "I would go with options 1 and 2 together now. The alphazero will be the
  next step later."
- Rating floor 1500 ("1800 is a lot, i think 1500 would even be enough").
- 300,000 training positions, 8 per game (AlphaGo's value net overfit when all
  positions of a game were used: "successive positions are strongly correlated").
- 4 of the 8 game boards start from mid-game human positions (plies 20-60).
- Mate-in-three held back.
- Value loss weight 0.01 in pretraining (AlphaGo Zero's supervised comparison
  used 0.01 to avoid value overfitting).
- Owner's prediction: not given.

## What Changes

- `scripts/prepare_games.py`: download Lichess 2013-12 (578,262 games, CC0),
  keep games with both Elo >= 1500, sample 8 positions per game, encode with
  OpenSpiel (mover-oriented observation, legal mask, human move as action id via
  SAN, outcome from the mover's side: +1 / -1 / -0.25 draw). Writes
  `data/games/human_{train,eval}.npz` (packed bits, gitignored) and
  `data/games/midgame_starts.csv` (FENs at plies 20-60).
- `scripts/pretrain_human.py`: cross-entropy on the human move + 0.01 x squared
  value error, Adam 1e-3, batch 256, 3 epochs, held-out top-1 every 200 batches;
  saves a `model_*.pth` checkpoint loadable by `load_models` / `init_from`.
- Environments: `game_start_sampler` + `num_midgame_envs`; boards
  `[num_puzzle_envs, num_puzzle_envs + num_midgame_envs)` reset from a sampled
  FEN as full games (ply cap applies). Config `game_start_file`, `midgame_boards`.
- Evaluation of the pretrained checkpoint like any other: held-out human-move
  top-1, Lichess m1/m2, 40 games.

## How we will know it worked

1. SL checkpoint alone: human-move top-1 on held-out positions (AlphaGo: 57% in
   Go; no chess reference at hand, so the number is a first measurement), Lichess
   m1/m2 top-1, 40 games (mate chances, plies, decisive rate).
2. PPO 120 updates from the SL checkpoint with 4 mid-game boards vs the fresh
   long run at 120 and 600 (`experiments/mate-in-two-ply-cap/`): puzzle held-out,
   opening entropy, games finished, 40 games.

### Outcome (2026-09-16/17)

Data: 578k games scanned, 55% passed Elo >= 1500; 298,960 train / 16,040
held-out positions; 50,000 mid-game FENs. Pretraining 3 epochs, 3,504 batches,
58 min: held-out human-move top-1 **0.341** (AlphaGo in Go: 0.57).

| checkpoint | human-move top-1 | Lichess m1 | Lichess m2 | opening entropy | 40 games: mate chances / found | mean plies |
|---|---|---|---|---|---|---|
| pretrained only (SL) | 0.341 | 0.169 | 0.132 | - | 139 / 5 (3.6%) | 170 |
| PPO 600 from SL, 4 mid-game boards | 0.165 | **0.570** | **0.510** | 0.28 | 100 / 7 (7.0%) | 176 |
| fresh PPO 600 (mate-in-two-ply-cap/LONG) | 0.078 | 0.444 | 0.351 | 0.42 | 31 / 3 | 186 |

PPO-from-SL curve: m1 0.31 @50, 0.41 @100, 0.52 @300, 0.57 @600; m2 0.23 @50,
0.30 @100, 0.44 @300, 0.51 @600. No plateau on mate-in-two: the human prior put
enough mass on forcing moves that the two-step reward was found from the start
(fresh run: flat at 0.08 until update 200). PPO dials were calm: clip fraction
0.12-0.16, approx KL 0.013-0.036 (fresh run: 0.2-0.4 / 0.04-0.09). Decisive
evaluation games 17.5% vs 7.5%, and three times the mating chances of the fresh
network.

Reading:
- Pretraining + RL beat RL alone on every puzzle number at equal RL budget, and
  the RL stage learned faster from the first update (0.31 vs 0.10 at update 50).
- RL narrowed the beam: human-move agreement fell 0.34 -> 0.17 during PPO (still
  double the fresh network's 0.08), and opening entropy sits at 0.28. This is the
  AlphaGo p. 486 caution observed directly: the outcome objective trades human
  diversity for its own preferences.
- Games are still mostly draws (75%) at ~176 plies, but with 100 mating chances
  and 7 taken. The in-game mate rate is still a small-count number.
- Owner's prediction: not given.

## What to understand

- Pretraining gives a starting point and diversity; RL then optimises for the
  outcome and can narrow the beam (AlphaGo p. 486).
- Why 8 positions per game beat 70: the shared outcome makes them near-copies.
- Mid-game starts are the curriculum idea for whole games: shorter, more varied
  games with tension already on the board.
- Business analogue: pretrain on logs of human decisions, then fine-tune with a
  reward (the SFT -> RLHF pattern for language models).
