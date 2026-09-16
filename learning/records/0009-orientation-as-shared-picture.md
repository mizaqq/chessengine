# Orienting the board makes white's and black's positions the same picture

2026-09-16, map-paper-to-code on AlphaZero's input description vs `orient_black`.
Owner traced a black pawn on a7 through the transform to a2 "in the same position
a white pawn was, from the other perspective" (correct; the grid also moves from
black-pawns to mover's-pawns). Identified plane 14 as side to move; the reason to
keep it after orientation (white's first-move advantage makes outcomes colour-
dependent, so the value head needs the colour) was supplied. For absolute move
ids the owner proposed translating the network's move numbers back to the
board's: the right fix, at the single line `actions[mask] = a`.

Prediction check: owner predicted "twice as fast" learning; run 2 crossed
held-out top-1 0.30 at update 200, the shared network at about update 60 and
ended 0.427 vs 0.352. Colour split of decisive games went from 6/24 to 3/3.

**Evidence:** `experiments/shared-network/seed0/`, `tests/model/test_orientation.py`.

**Implications:** weight sharing across a symmetry is data augmentation by
construction and removes spurious asymmetries. Wall time did not drop: one
backward on 180 samples costs what two on 90 did. In-game transfer is still
absent (5% mate rate), so the next problem is distribution shift, not capacity.
