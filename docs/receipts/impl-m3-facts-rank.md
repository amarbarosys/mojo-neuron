## Facts rank falsifier (prereg docs/prereg-facts-rank.md, 20 facts, layer 16, k = 100)

| scale | gold in top 20 (of 20) | median gold NLL | median gap (lower bound when absent) |
|---|---|---|---|
| 0 | 1 | 6.97 | 3.45 |
| 8 | 14 | 3.86 | 2.69 |
| 32 | 2 | 34.62 | 7.51 |

sanity: max |dnll| at scale 0 = 0.00e+00 (must be 0)
R1 (top-20 count up by 3 or more): scale 8 True, scale 32 False
R2 (median NLL down by 2.0 nats or more): scale 8 True, scale 32 False
RESULT: MOVES (grid too small), next step to the boss
