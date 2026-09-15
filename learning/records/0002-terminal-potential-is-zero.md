# Material only matters while the game continues

Established 2026-09-15 during the potential-based-shaping comprehension check.
Tracing the returns loop by hand, the owner initially charged white for a rook lost
on the opponent's mating move (-7). Corrected: the terminal position has potential
0, white never received a loan for that rook, so the return is the outcome alone
(-2). The owner then restated the rule: material is a stand-in for future winning
chances; once the game is decided there is no future for it to stand in for.

**Evidence:** hand trace of `compute_returns_for_model` on the "opponent captures
and mates" scenario, with the bootstrap (5) correctly identified as a value-head
guess that `done` must wipe out.

**Implications:** ready to discuss reward horizon and why sparse terminal reward is
hard; the "shaping is a loan" model is usable for `shaping_scale` decisions.
