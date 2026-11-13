## Facts late layer (prereg docs/prereg-facts-late.md, 20 facts, k = 20)

implants off: gold rank 1 for 1/20

| layer/scale | real rank 1 | real moved | control rank 1 | trivia kept (of 10) real |
|---|---|---|---|---|
| 28/0.5 | 1 | 1 | 1 | 10 |
| 28/1 | 1 | 2 | 1 | 10 |
| 28/2 | 1 | 3 | 1 | 10 |
| 28/4 | 1 | 11 | 1 | 10 |
| 28/8 | 1 | 18 | 1 | 10 |
| 30/0.5 | 1 | 2 | 1 | 10 |
| 30/1 | 1 | 3 | 1 | 10 |
| 30/2 | 1 | 5 | 1 | 10 |
| 30/4 | 1 | 12 | 1 | 10 |
| 30/8 | 1 | 19 | 1 | 10 |
| 31/0.5 | 1 | 7 | 1 | 10 |
| 31/1 | 1 | 12 | 1 | 10 |
| 31/2 | 1 | 18 | 1 | 10 |
| 31/4 | 1 | 19 | 1 | 10 |
| 31/8 | 6 | 19 | 1 | 10 |

LL-A (rank 1 >= 16/20): MISSED (best 6/20 at 31/8)
LL-B (control rank 1 <= 2 at 31/8): PASS (1/20)
LL-C trivia kept at 31/8: 10/10

Wall time: capture 5 s + real arm 14 s + control 13 s (480 requests each, one engine per arm, logits only). Engine mojo-baro main `.work/engine-9b`,
GGUF = e11b pack source. Note written after the result: at 31/8, 19 of 20 top-1 tokens moved but only 6 became the gold token. The value is
U[right] - U[wrong]: it mostly sinks the wrong token and the next competitor wins. Testing value = U[right] alone needs its own prereg.
