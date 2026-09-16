## Decisions

### D1. Data pipeline (`scripts/prepare_games.py`)
Stream the .zst PGN with python-chess; skip games with a missing Elo, either Elo
< 1500, or fewer than 12 plies. For each kept game replay the moves; choose 8
ply indices uniformly (all if fewer); at each chosen ply record FEN, mover, human
move UCI, result from the mover (+1 win, -1 loss, -0.25 draw). Stop when 315,000
positions are collected (300,000 train / 15,000 eval, split by game so the same
game never appears in both). Encode with `OpenSpielEnv.reset(fen)`: observation
`env.state()` oriented via `orient_black` when black moves, legal mask, action
id via `actions_for_uci([move])` (skip if the SAN bridge fails). Pack obs and
mask bits with `numpy.packbits`. Mid-game starts: from the same games, FENs at
plies 20-60 (one per game, up to 50,000) to `midgame_starts.csv`.

### D2. Pretraining (`scripts/pretrain_human.py`)
`loss = -log p[action] + 0.01 * (v - z)^2` where `p` is the masked softmax the
model already returns (illegal moves have probability 0, so the loss also
teaches legality only through the mask). Adam 1e-3, batch 256, 3 epochs, BN in
train mode (no ratio here). Every 200 batches: held-out top-1 and loss to a
JSON log; checkpoint at the end via `save_model` to
`experiments/human-pretraining/sl/`. Wall-clock estimate: 1,170 batches/epoch x
~1.7 s ~ 35 min/epoch.

### D3. Mid-game boards
`FenSampler(fens, seed)` with `sample()` / `with_seed()` (same interface as the
puzzle samplers). Vector envs take `game_start_sampler`, `num_midgame_envs`;
`_reset_env(i)`: puzzle boards as now; `i < num_puzzle_envs + num_midgame_envs`
-> `env.reset(sampler.sample(), max_plies=self.max_plies)`; else opening. Async
workers get their own seeded copy. `info["puzzle_boards"]` stays False for
mid-game boards; their results count in the game rates. Config resolver
`resolve_midgame_boards(config) -> (sampler, n)` with `game_start_file`,
`midgame_boards` (0 <= n <= num_envs - num_puzzle_envs).

### D4. Runs
Chain: data build -> pretrain -> eval SL checkpoint (puzzles, 40 games) -> PPO
120 updates `init_from` SL with `midgame_boards: 4` -> eval. One background
script, `experiments/human-pretraining/run_all.sh`.

## Risks
- 2013 Lichess games at 1500 are amateur blitz; the target is "human-like", not
  "strong". Held-out accuracy is the honest readout.
- The value head learns z in [-1, 1] while PPO's targets are shaped returns on
  the +-2 scale; the 0.01 weight keeps the value head barely trained, so PPO
  re-fits it. Noted as a scale mismatch, not fixed here.
- Mid-game boards make `mean_terminal_return` and win rates mix two start
  distributions; the 40-game evaluation still starts from the opening.
