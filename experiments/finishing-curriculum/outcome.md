# Finishing curriculum, arm FIN (2026-09-17)

120 updates from the SL + PPO 600 checkpoint, layout 4 puzzle / 4 finishing /
2 mid-game / 2 opening boards, depth start 2, advance at 0.6 over 100 episodes,
cap k + 20. Elapsed 35 min. (Two identical drivers ran by accident with the same
seed; results agree, one checkpoint kept.)

## Result: flat at the standard budget

| Dial | Start checkpoint | After 120 updates |
|---|---|---|
| Held-out conversion, 2 plies before the human mate | 0.18 | 0.18 |
| Held-out conversion, 10 plies | 0.05 | 0.09 |
| Held-out conversion, 20 plies | 0.08 | 0.05 |
| Own-game mate-in-one, held-out positions (top-1) | 0.07 | 0.06 |
| Own-game mate-in-one rate, 40 sampled games | 0.07 (7/100) | 0.05 (4/81) |
| Lichess m1 / m2 top-1 | 0.570 / 0.510 | 0.578 / 0.490 |
| Curriculum depth | 2 | 2 (never advanced) |

Training-side finishing success per 10-update window (sampled play): depth 0
between 0.29 and 0.46 with no trend; depth 2 rose from 0.10 to about 0.19 in the
first 40 updates and stayed there. Mixture rate 0.19-0.32, always under the 0.6
threshold, so the curriculum stayed at depth 2 for the whole run. About 16
finishing episodes per update (1,900 in total).

## Reading

- Depth-0 success (0.38) is far below Lichess mate-in-one accuracy (0.58): the
  positions where 1500-rated humans deliver mate look different from curated
  puzzles, and the model puts less than half its mass on the mate there.
- No learning signal reached the policy in 120 updates: neither the training
  success rates nor any held-out dial moved beyond noise (100 games, about
  +-0.04). Game boards still drew 76-96% of their games.
- The 0.6 threshold was set assuming depth-0 success near puzzle accuracy; it was
  unreachable from the start, so the adaptive part of the curriculum never ran.
  That alone would not have mattered: the easy rungs did not improve either.

Open before the owner (recorded, not decided): longer run as the named exception
(the mate-in-two rung needed 200+ updates); more finishing boards and fewer
game boards so finishing episodes are a larger share of each batch; a
threshold relative to the starting rate; a frozen or greedy defender.
