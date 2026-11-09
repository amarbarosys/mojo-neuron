# Prereg A1: capture op (written before the result run, 2026-11-09)

Bars (all must hold, any miss = FAIL, no post hoc change):
1. Bit-equal: for every captured layer, the `r` row at the last prompt position equals `f16(BARO_DUMP slot 2l+1)` byte for byte
   (BARO_DUMP is the engine's existing per-layer residual dump at the last prompt position).
2. Mutant: comparing layer l's `r` with the dump of layer l+1 must differ (the check can fail).
3. `y` consistency: at the same position `r(l) - dump slot 2l` (residual after attention) equals `y(l)` within 1e-2 relative to
   the row's max abs (f16 rounding of y and r), so the `y` row is the FFN output and no other tensor.
4. `x` sane: finite, not all zero; its RMS within 0.1..100 (post-norm rows, scale set by the norm weights).
5. Record count: `tokens.i32` entries == prompt length + generated steps recorded; file sizes equal `n * nf * h * 2`.
Layers: 0, 16, 31 (first, middle, last) on the 13-token pack prompt, greedy.
