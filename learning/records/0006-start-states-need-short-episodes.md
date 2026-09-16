# Starting near the reward only helps if the episode ends soon after the attempt

2026-09-16. Run 1 mixed puzzle starts into resets at 0.5: held-out mate-in-one
accuracy stayed flat (0.04-0.08 vs 0.074 baseline) over 500 updates. The owner
first read the flat curve as "not enough episodes" and "network settings"; the
actual cause: a missed mate turned into a 200-ply game, so the run made ~200
puzzle attempts in total, ~10 of them mates. Fix (owner approved): puzzle boards
end the episode after the mover's move (miss = terminal, reward 0), and puzzle
boards are a fixed subset of boards so the share of plies matches the fraction.
Run 2: 45,000 attempts, held-out top-1 0.074 -> 0.352, training solve rate within
one point of held-out (no memorisation), in-game mate rate 4.7% -> 18%, decisive
games 22% -> 40%.

**Evidence:** `experiments/mate-in-one-curriculum/frac05/` vs `frac05-onemove/`.

**Implications:** the Salimans & Chen argument (exponential -> quadratic
exploration) has a hidden assumption: failed attempts must terminate quickly.
Mixing tasks of very different episode lengths in one learner also floods every
game counter, so metrics must be split by source before they can be read.
