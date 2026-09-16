# Training on the network's own mate positions did not reach its games

2026-09-16. Owner's hypothesis: puzzle positions (human games) differ from
self-play positions, hence 43% on Lichess vs 5% in games. Test: continue the
shared checkpoint 500 updates on 4 puzzle boards, A = Lichess + positions
harvested from that checkpoint's own games (50/50), B = Lichess only.

| after 500 updates | A mixed | B Lichess |
|---|---|---|
| Lichess held-out | 0.429 | 0.486 |
| own-positions held-out | 0.140 | 0.083 |
| in-game mate-in-one | 1/29 | 7/157 |
| mean plies / draws | 83 / 33 | 178 / 28 |

Own positions helped only on own positions and cost Lichess accuracy; neither arm
moved the in-game rate. The dominant symptom is now game collapse: opening-board
entropy 0.51 -> 0.37 -> 0.24 across runs, games ending as repetition or
fifty-move draws after ~80-180 plies, so mate chances never arise.

**Evidence:** `experiments/mixed-puzzle-sources/cont-mixed/`, `cont-lichess/`.

**Implications:** the diagnosis (distribution shift) was right, the remedy
(more on-distribution puzzles) was not sufficient; the in-loop position buffer
was dropped on this evidence. Policy sharpening in real games is the next
problem: PPO's clipping and the entropy term are the tools. Default diet is
Lichess only on 4 boards; the mix stays one config line away. Owner's separate
point recorded: mate in one needs no planning, but the network learns it as
pictures, so rare shapes stay hard; one-ply search would make it exact.
