# Predicted the sign flip of value-vs-material under a smaller shaping scale

2026-09-15. After seeing that at `shaping_scale` 1.0 the value head still moved
*with* material (corr +0.46 / +0.39 after 500 updates, i.e. the loan was not
learned), the owner chose to test scale 0.2 and predicted the correlation would
turn negative. It did (-0.28 white, -0.18 black on the seeded game), though the
fitted slope (-0.02) is far from the theoretical -0.2, so the head has only begun
to absorb the offset. Side effect observed: decisive games rose from 13% to 24%
of finished games and the loss scale dropped ~25x because returns shrank.

**Evidence:** prediction stated in chat before the run; result matched in sign.

**Implications:** the owner can reason about how shaping magnitude interacts with
value-head learning; fair comparison of value heads requires the same fixed
positions, not games each model played itself.
