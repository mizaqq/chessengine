# Elo ladder vs Stockfish (change elo-eval) — outcome, 2026-09-22

Method: `scripts/eval_elo.py`, Stockfish `UCI_LimitStrength` at UCI_Elo 1320/1400/1500/1700/1900,
20 games per colour per level (200 per checkpoint), 20 ms per engine move, network sampled.
Logistic MLE with draws as half a win, 200-sample bootstrap. Scale is CCRL blitz per Stockfish
`src/search.h`; 1320 is the floor, so every number below is an extrapolation and only the
ordering and the raw scores at 1320 should be trusted.

| Checkpoint | Lineage | vs 1320 | vs 1400 | Fit (95%) |
|---|---|---|---|---|
| SLHI2_256 | supervised only, 256x10, 5.7M positions both players >= 2000 | +4 =1 -35 | +3 =1 -36 | 932 (773-1024) |
| LONG256X | SLHI2_256 + guarded PPO (weights of update 200) | +1 =1 -38 | see elo.json | 868 (694-977) |
| SL | old 128x10 prior, 299k positions | +0 =0 -40 | +0 =0 -40 | 428 (400-628) |
| LONG3 | old 128 lineage, most RL | +0 =0 -40 | +0 =0 -40 | every game lost |

Raw files: `experiments/width/LONG256X/elo.json`, `experiments/human-pretraining/slhi2_256/elo.json`,
`experiments/human-pretraining/sl/elo.json`, `experiments/endgame-technique/LONG3/elo.json`.

## Reading

1. The whole jump from the old line to the new one came from supervised learning (data quality,
   data quantity, width). RL added nothing measurable against an outside opponent in either lineage.
2. LONG256X beats its prior 92-62 head to head yet scores no better (slightly worse, intervals
   overlap) on the ladder. Self-play gains were partly opponent-specific: the non-transitive part of
   the game (Balduzzi 2019). The internal 200-game matrix is therefore not sufficient evidence of
   strength; new arms are put on the ladder as well.
3. Owner (2026-09-22): "this is enough, i just wanted to see the estimate, 1 win is good enough for
   me" — no handicap levels or Maia opponents; the ladder stays as the transfer check.

Prediction check: none recorded by the owner. Claude's expectation before the run was "well below
1320 for every checkpoint, LONG256X ahead of SLHI2_256"; the second half was wrong.
