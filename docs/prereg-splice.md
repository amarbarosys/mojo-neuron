# Prereg: engine splice op for the transcoder spliced-NLL gate (B1 on Qwythos) (written 2026-11-09 before any run)

Request from lane B. Engine request fields (qwen35 dense path, engine.mojo/window.mojo, needs BARO_STEER_SERVE=1 like `gain`): `splice_layer` L plus `splice` = path of an f32 file `[n_positions][hidden]` (row p replaces the
FFN output of layer L at absolute position p of the request), or `splice_zero: 1` (the FFN output of layer L is dropped). The residual update `x += ffn_out` of that layer is replaced by `x += row` (or nothing). Everything else, including later
layers, runs normally. Teacher-forced NLL: the steer-lens line (`slens_tgt`) gains `"nll"`, the mean absolute NLL of the target token; the NLL of a text continuation is one request per position (prompt = text[:i], slens_tgt = text[i]),
the same rows file serving every prefix because rows are indexed by absolute position.
Setup: Qwythos-9B Q4_0 pack, layer 16, a fixed 48-token text (the first lines of `tests/data/eval64.txt`), positions 8 to 47 scored, rows = the unmodified run's own FFN outputs (A1 capture field y, f16 widened to f32).
Bars (any miss = FAIL):
1. No splice field: nll equals the run without the new code path to 1e-9 (same request, the new fields absent).
2. Identity splice (rows = the true y): mean |nll - nll_unspliced| <= 0.01 nats and mean KL <= 1e-3 (the only difference is f16 rounding of y).
3. Zero splice: mean nll rises by >= 0.05 nats over unspliced (the op has an effect), and `splice_zero` equals a splice file of zeros bit for bit.
4. Mutant: rows shifted by 7 positions (wrong tokens' outputs) raise mean nll by >= 0.05 nats over unspliced.
5. rlog: one normal line per scored configuration.
Not claimed: that spliced NLL with a trained transcoder passes B1's own bars (lane B measures that); only that the engine op does what its name says.
