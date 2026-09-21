
## 2026-09-21: priors on more and better data, wider networks (GPU)

Owner: "we should fix data there"; "newer data"; "the next one we will test should be 256".

| prior | data | positions | held-out top-1 (own set) | m1 / m2 puzzles | own-game m1 | vs slbig192, decisive |
|---|---|---|---|---|---|---|
| sl (128) | 2013-12, Elo >= 1500 | 299k | 0.341 | 0.169 / 0.132 | 0.036 | (loses 24-1 to slbig128) |
| sl192 | same | 299k | 0.338 | 0.161 / 0.144 | 0.070 | - |
| slbig128 | 2013-12 | 1.43M | 0.405 | 0.261 / 0.224 | 0.096 | 15-10 |
| slbig192 | 2013-12 | 1.43M | 0.404 | 0.267 / 0.253 | 0.071 | - |
| sl2m192 | 2013-12 + 2014-01 | 2.85M | 0.432 | 0.369 / 0.333 | 0.184 | **23-7** |
| slhi256 | 2026-08, Elo >= 2000 (streamed) | 2.85M | 0.447 (Elo>=2000 set) | **0.469 / 0.378** | **0.226** | **19-3**; vs sl2m192 16-11 |

Reading.
- **Quantity:** 299k -> 1.43M -> 2.85M at Elo 1500 gives 0.341 -> 0.405 -> 0.432 held-out and
  23-7 in games for the last doubling. The log-linear curve has not bent (Tuyls et al. 2023).
- **Quality:** the same 2.85M positions from players rated >= 2000 (streamed from the 30 GB
  August 2026 dump, headers checked before parsing) with a 256-filter network beat the
  two-month 1500 prior 16-11 and lifted mate-in-1 puzzles 0.369 -> 0.469 with zero RL.
  Its held-out top-1 is on a different set and is not comparable; the matrix is.
- **Width:** null at 299k (sl vs sl192) and at 1.43M (slbig128 vs slbig192, 15-10). The 256 run
  confounds width with data quality; the RL stage (LONG256 vs LONG192B) is where width showed
  before (LONG192 beat LONG128 28-16).
- GPU wall clock: 13 / 32 / 43 min for 1.43M x 2 ep (192), 2.85M x 2 ep (192), 2.85M x 2 ep (256).

Next: LONG256 = slhi256 + 600 updates on today's recipe (LONG192B as the comparison point).

## 2026-09-22 night: two high-Elo months (slhi2_256)

| prior | positions (Elo >= 2000) | held-out top-1 (own set) | eval CE | m1 / m2 puzzles | own-game m1 | vs slhi256, 200 games |
|---|---|---|---|---|---|---|
| slhi256 | 2.85M (2026-08) | 0.447 | 1.784 | 0.469 / 0.378 | 0.226 | - |
| slhi2_256 | 5.70M (2026-07 + 08) | **0.471** | **1.685** | **0.635 / 0.464** | 0.283 | **104-28** |

Doubling the high-Elo data: mate-in-1 puzzles 0.47 -> 0.63 with zero RL (LONG3, the old
lineage's best after ~2,000 RL updates, had 0.81), and 104-28 in games against the one-month
prior. The bare slhi2_256 also beats LONG256G (600 guarded RL updates on the one-month prior)
79-55. The prior curve has not bent; the streamed build took 30 min, the 256 pretraining 77 min
on the GPU. Next month (2026-06) is the same cost again.
