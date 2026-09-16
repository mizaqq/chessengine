# Black beat white 24-6: equal at finishing mates, unequal at avoiding them

2026-09-16, diagnose-from-symptoms. Owner's hypothesis: 40 games is a small
sample. Test: 2 or fewer white wins in 16 decisive games under a fair coin is
~0.2%; a second seed of 40 games repeated the asymmetry (4-10). Owner's chess
prior (white should win more) correctly flagged the direction as suspicious.
Per-colour split: mate-in-one found 14% (white) vs 19% (black), but white handed
black 74 mate-in-one chances against 14 the other way. Held-out puzzle accuracy
by side to move: 34.0% vs 36.6%. So the curriculum worked equally for both
networks; the gap is in ordinary play of the white network.

**Evidence:** `experiments/mate-in-one-curriculum/frac05-onemove/games*.json`,
per-colour analysis in conversation.

**Implications:** with two independently initialised networks and one seed,
"one network is weaker" and "systematic colour bug" are indistinguishable; the
discriminating experiment is a second training seed (running at time of
writing). A metric that is not split by side can hide a 5x asymmetry.

**Verdict (seed-1 run, later the same day):** white 5, black 3 decisive games.
The asymmetry was the seed: one of two independently initialised networks was
weaker, not a colour bug. Second finding from the same run: held-out top-1 0.364
(same as seed 0) but in-game mate rate 4.7% and 25 of 40 games hit the 300-ply
cap, so the in-game transfer seen for seed 0 (18%) did not reproduce. Puzzle
skill transfers to real games unreliably at this budget; one seed is not a
result.
