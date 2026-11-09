# Prereg: batched-prefill capture (written 2026-11-09 before the run)

Request (boss, for lane B): capture rate with batched prefill on Qwythos layer 16. Engine neuron2b: `BARO_CAPTURE` now also works through the batched prefill path (prompt chunks of up to 1024 rows): per captured layer the chunk path stages the FFN input (widened), the residual before and after the FFN add
(y = after minus before), and the harness writes the same f16 records (`docs/format-capture.md`) for the whole chunk. Needs BARO_STEER_SERVE unset (it keeps prefill batched). The last prompt token still goes through the single-position path.
Setup: a fixed 2048-token text (first 2048 tokens of `docs/plans/2026-11-09-neuron-v2.md` of the main checkout), one request, n = 1, layer 16, fields xyr, same engine binary twice: batched (default) and one position per step (`BARO_STEER_SERVE=1`).
Bars (any miss = FAIL; the rate decides what happens next, it is not a bar):
1. Format: both runs write the same file sizes (2048 records of 3 x 4096 f16) and identical `tokens.i32` and `req.i32`.
2. Agreement with the one-position capture, per record: field r (residual) cosine >= 0.999 at every position and median relative L2 <= 0.02; field y median relative L2 <= 0.05; field x median relative L2 <= 0.02. (Batched GEMMs and gemv differ numerically; the bars say how much.)
3. Rate = prefill rows / the engine's `prefill_s` of the batched run (includes the file writes), reported with the one-position run's rate. If the batched rate is >= 1000 tokens/s the 400k-token capture of the request is made; otherwise only the rate is reported.
