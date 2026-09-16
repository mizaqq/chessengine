# Reading three accuracies at once: attempts, not set size; whose positions; and how games end

2026-09-16, diagnose-from-symptoms on the mixed-puzzle-sources run (fresh
network, 4 puzzle boards, Lichess 50k + self-play harvest 50/50). Owner's
reading: "way better"; Lichess fell "because of more positions"; self-play rose
to 0.167; higher mate rate with fewer missed mates; "not enough episodes".

Corrections established in conversation:
- Lichess 0.427 -> 0.266 came from 30 Lichess attempts per update instead of 90,
  not from the 50k set (held-out unchanged; train/held-out already agreed).
- Self-play 0.167 was measured on positions harvested from a *different*
  network's games (the shared checkpoint); the run trained from scratch. It shows
  harvested positions are learnable, not that the network improved on its own
  games. Design flaw owned by Claude: per-run refresh requires continuing from
  the checkpoint the harvest came from.
- 3/39 vs 6/121 in-game mates is not readable. Fewer opportunities arose because
  35/40 games ended as rule draws at ~146 plies (repetition / fifty-move) with
  opening-board entropy at 0.37: a sharp weak policy loops. Not "fewer misses".
- "Not enough episodes" is half right (a third of the Lichess data) but would not
  fix the positions mismatch.

**Evidence:** `experiments/mixed-puzzle-sources/seed0/`, `baseline_selfplay.json`.

**Implications:** next experiment continues from `shared-network/seed0` with the
harvest that came from it, against a Lichess-only continuation. Always ask of a
held-out set: whose positions are these? And of a game count: how did the games
end?
