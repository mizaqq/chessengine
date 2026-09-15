# The policy learns from the advantage, not the return

Established 2026-09-15 while designing potential-based material shaping. The owner
can state that the policy gradient pushes a move by (return − value estimate), and
that a shaped return such as −7 for a mating move is a fixed offset that cancels
against the shaped value head (V' = V − Φ, Ng et al. 1999, Corollary 2). They also
saw that the cancellation only holds once the value head has learned to track
material; with an untrained head the advantage is dominated by the loan (+11
instead of +0.5), so shaping raises the accuracy demand on the value head.

**Evidence:** traced the ahead/behind mating scenarios in conversation; initially
answered +0.5 for the untrained-head case, corrected to +11 and explained the effect
on policy and value head.

**Implications:** ready for GAE lambda as a bias/variance dial on the same value
head; `shaping_scale` is understood as a trade-off, not a free knob.
