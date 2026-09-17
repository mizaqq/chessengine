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

# Arm FIN2 (2026-09-17): 6 finishing boards, threshold 0.4, 300 updates, switch at 200

Flat again. Depth reached 4 near update 70 and stayed. Held-out conversion d2 /
d10 / d20: 0.15 / 0.04 / 0.07 (start 0.18 / 0.05 / 0.08). Own-game mate-in-one 5/41.
Lichess m1 0.578 -> 0.559, m2 0.51 -> 0.477 (slight forgetting).

# Probe (scripts/probe_finish_values.py): the value head is a material counter

Value from the mover's view on positions where the mover mates next move:
Lichess puzzles +1.32 (mate mass 0.57); human games one ply before the mate
-0.85 (mass 0.35); the model's own games -2.61 (mass 0.06). Outcomes alone span
-2..+2; -2.6 is reachable only through the material term. The network learned
material, and mates in puzzles come with a material lead; mates without one are
invisible to both heads.

# Isolation run ISO_d0: 12 boards of depth-0 human-mate positions only, 30 updates

Rules out dilution as the whole story: with 100% finishing positions in every
batch (about 66 episodes per update, 2,000 in total) depth-0 success stayed
0.39-0.46, mate mass 0.35 -> 0.37, value -0.85 -> -0.68, while clip fraction
0.17-0.20 and KL 0.02-0.04 show the policy moving every update. Also relevant:
Lichess m1 puzzle accuracy has not risen above 0.57-0.58 across 900 PPO updates
from the SL checkpoint despite ~150k puzzle episodes, so a plateau of this
network-plus-update on mate recognition is on the table, not only sample count.

Next (owner): shaping off (arm FIN3), pretrained prior supplies the density.

# The ceiling is exploration, not capacity (2026-09-17, evening)

Mass on the mating move over 1,000 Lichess m1 puzzles (SL+PPO 600 checkpoint) is
bimodal: 52% above 0.8, 35% below 0.05, 13% in between. On the confident-wrong
third the mate is sampled less than 1 in 20 times, so policy gradient gets no
signal there; each rare hit is one clipped step. Supervised upper bound: the same
network trained with cross-entropy on the mating moves of the 20k training puzzles
(Adam 3e-4, batch 256) went 0.57 -> 0.745 (1 epoch) -> 0.849 (3) -> 0.902 (12) on
the held-out set, while 900 PPO updates left it at 0.57. Capacity is not the
limit; the update's exploration is. Options, in order: exploration noise on the
curriculum boards (AlphaZero root Dirichlet is the same device; PPO ratio must use
the behaviour log-prob), search as teacher (the agreed AlphaZero step), supervised
target on labelled boards (fallback), bigger network (not needed).
