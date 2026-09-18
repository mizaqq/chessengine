# 0015 — Self-imitation switches itself off through the priority, not the value

**Date:** 2026-09-18
**Context:** change `self-imitation-curriculum`, comprehension check (trace by hand) on
`src/training/sil.py` after arm CURSIL3.

**Question.** A three-ply technique episode ends in mate (+2 / -2, gamma 0.99). The value
head later learns to say 1.98 on the mating position. What happens to that ply in the
self-imitation buffer?

**Owner's answer:** "values is lowered."

**Correction.** The value is not lowered; it has risen to meet the return (2 * 0.99 = 1.98).
What drops is the ply's *priority*, max(R - V, 0) = 0, so the ply is never drawn again.
Self-imitation replays only plies whose past return still beats the current estimate, and
as the value head absorbs those wins the gap closes and the replay fades on its own. The
dial is the share of the buffer with positive priority (0.42 -> 0.34 over CURSIL3).
Companion fact traced together: the losing side's plies have negative returns and priority
zero from the start, so SIL learns only from the side that did well.

**Why it matters outside chess.** Prioritised replay of "better than expected" outcomes is
self-limiting: a recommender that replays its surprisingly good sessions stops replaying
them once its own predictor expects that quality. The switch-off is a feature, and the
positive share is how you see it working.
