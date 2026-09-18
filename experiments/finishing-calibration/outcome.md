# Finishing-conversion calibration (2026-09-19, 00:0x)

Owner: "run a calibration." Question: how convertible are the rewound human positions
against OUR sampling defender? Three attackers on the same 100 held-out starts per depth:
the checkpoint sampling (the dial), the checkpoint greedy, and the human's recorded moves
while the defender stays on the record, then the checkpoint greedy. `scripts/calibrate_finishes.py`.

| checkpoint | attacker | d2 | d10 | d20 |
|---|---|---|---|---|
| SL prior | sampled / greedy / human-line | 0.09 / 0.10 / 0.57 | 0.05 / 0.09 / 0.11 | 0.04 / 0.12 / 0.11 |
| POOL_LONG | sampled / greedy / human-line | 0.31 / 0.34 / 0.73 | 0.19 / 0.14 / 0.23 | 0.13 / 0.17 / 0.17 |
| LONG2 | sampled / greedy / human-line | 0.45 / 0.43 / **0.75** | 0.12 / 0.15 / **0.28** | 0.08 / 0.14 / **0.22** |

Our defender leaves the human record after 1.0-1.7 plies on average and stays on it to
the end in 0-3% of games at depth 10-20 (45% at depth 2, where one reply is all there is).
So "human-line" is in effect: the human's first one or two moves, then our greedy network.

## Reading

- **The dial has headroom.** With the human's first move(s) and our own greedy play
  afterwards, LONG2 converts 0.28 at depth 10 and 0.22 at depth 20, about double what it
  achieves on its own (0.12-0.15). The positions are not saturated at 0.13; the network's
  weakness is the first move or two from the rewound position, the plan initiation, not
  the technique afterwards.
- **The human first move is a strong demonstrator exactly where our finishing boards use
  it**, and LONG2 had no human finishing board at all (the slot was all technique), so the
  conversion dial had nothing to train on in that run. Depth 2 tells the same story: LONG2
  finds the forcing move 43-45% of the time, the human's move converts 75%.
- Progress is real over checkpoints: sampled d2 0.09 (prior) -> 0.31 -> 0.45; d10 sampled
  is noisy (0.05 / 0.19 / 0.12) but greedy 0.09 -> 0.14 -> 0.15.

## Consequence

Overnight run LONG3: LONG2 + 600 updates with the finishing slot mixed (2 technique + 2
human finishing boards by weight 0.5 / 0.5), everything else as LONG2. Targets: finishing
d10 / d20 sampled toward the human-line ceiling (0.28 / 0.22), d2 toward 0.75; technique
and puzzle dials held. Next code change after that: self-demonstrations (our own winning
line as the finishing demonstrator once we have one for a record).
