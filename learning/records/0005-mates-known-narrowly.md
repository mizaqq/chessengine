# The models know a few mating pictures, not the concept of mate

2026-09-15, from 40 sampled games with the scale-0.2 checkpoints. Hypothesis
(owner): mates are found by luck. Evidence against: in 6 of 8 decisive games the
mating move was the policy's top choice with p 0.53-0.99 (uniform ~0.03). Evidence
for a narrow skill: of 215 positions with a mate-in-one available, 207 were missed
(3.7% found). Greedy play never reaches the known pictures and ends in threefold
repetition; sampling explores into them.

**Evidence:** `experiments/` scripts in conversation; per-game mate probability and
rank table.

**Implications:** mate-in-one rate adopted as the first evaluation metric; the
endgame curriculum (roadmap 04/2) has concrete motivation; supervised puzzle
pre-training is the comparison, not the starting point. Greedy argmax is not a
faithful summary of a policy trained by sampling.
