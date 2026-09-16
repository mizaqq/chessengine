# A non-zero start potential shifts every return, not the policy

2026-09-16, trace-by-hand. Puzzle board, white to move, material +5, gamma 0.99,
scale 0.2. Owner's answer: miss returns 0, mate returns +2. Actual walk: the
mover is charged 0.2 * 5 = 1.0 of start potential and the episode ends at
terminal potential 0, so the loan is never repaid. Miss returns -1.0, mate
returns 0.98. The owner's conclusion about the advantage was right for the right
reason: both moves share the shift, so the difference (1.98) is untouched and the
value head learns the shifted level (about -1.0 + 0.99 * 2 * hit rate).

**Evidence:** conversation trace of `compute_returns_for_model` on a one-step
episode; Ng, Harada & Russell 1999, Corollary 2 (V' = V - Phi).

**Implications:** a winning puzzle position reads as a negative value until the
potential is added back (same lesson as record 0003). Theorem 1 does not care
where the episode starts.
