# Prereg lens-diff: contrastive lens of matched prompts (written before the result runs, 2026-11-09, lane agent)

Source: papers synthesis 2026-11-09 (single-reader idea, contrastive lens). `nx lens-diff {gguf, cap, layer, a, b, k?}`: for matched prompts
`a_i` (requests `a` of a capture dir) and `b_i` (requests `b`, same length, same order) take the last-token residual (field r) at `layer`,
`d_i = h(a_i) - h(b_i)`, `m = mean(d_i)`, lens of `m` = W_U (m * output_norm), top and bottom k tokens. Not unit-normalised (F3 `concept` unit-normalises
each term; this keeps the size of the difference). Also reports `norm(m)`, `consistency` = share of pairs with cos(d_i, m) > 0, and a paired null.
Paired null = centred sign flips: `d_i' = s_i * (d_i - m)`, `s_i = +-1` from xorshift seeds 1..20, lens of the mean (what the residual variation alone gives).
Data: `.work/cap/qwythos-enfr` (20 French sentences, then their 20 English translations, layers 15, 23, 27), Qwythos-9B.

Bars (all must hold, a miss = FAIL, no post hoc change):
1. Antisymmetry: lens-diff(A,B).top ids equal lens-diff(B,A).bottom ids in the same order, both layers. `A` against `A` gives `norm` exactly 0 and a `zero:true`
   reply with no lens and no NaN.
2. Known pair, FR - EN, k = 50, at layer 23 or 27 (either passes, as F3): top holds >= 4 French function words and <= 1 English; bottom holds >= 4 English
   and <= 1 French. French markers ` le la les de des du et est un une que pour dans qui pas sur avec au aux je il nous vous`; English markers
   ` the of and is to that with for in are was on at by`.
3. Null: at the passing layer, French-marker count in the top 50 of the 20 null lenses: mean <= 1.0 and max <= 3 (the signal is the pairing, not the word list).
4. Consistency >= 0.9 at the passing layer (a shared direction, not two outliers).
5. Mutant `-D LENSDIFF_MUT=1` (sum of the two means instead of the difference) must FAIL bar 2 in the gate (the check can fail).
6. One normal rlog line per call with norm and consistency; a non-finite residual gives an anomaly line.
Info only (no bar): layer 15 counts, the F3 unit-normalised counts next to the raw ones.

Receipts: `.work/lensdiff/<run>/*.json`, `tests/lensdiff.sh` output.
