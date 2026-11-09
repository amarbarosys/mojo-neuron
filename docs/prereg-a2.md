# Prereg A2: ablation scan (written before the result run, 2026-11-09)

Setup: Qwythos-9B Q4_K_M on engine.mojo (baro neuron2), layer 16, `gain 16:N:0` per neuron in one resident process, 64 fixed prompts
(`tests/data/eval64.txt`; context = all tokens but the last, target = the last token), damage = mean KL(base || ablated) and mean NLL change
of the target. Base requests use the same layer-loop path (`gain 16:0:1`).
Bars (any miss = FAIL):
1. 8 of the 24 scanned neurons re-run alone, one engine process each: KL and dNLL equal the batch within 1e-6 abs + 1e-3 relative.
2. Identity gain (k = 1) on neurons 5 and 6 scores KL <= 1e-9 (a no-op is a no-op to the bit).
3. Mutant: a single-neuron result compared with another neuron's batch value must differ.
4. The scan writes a normal rlog line per run (24 + 2 + 8 runs -> at least 10 normal lines in the gate's NX_RLOG dir).
Not claimed: that all 12288 neurons were scanned. Measured cost 57 ms per 10-token request, so 3.7 s per neuron at 64 prompts; a full layer is
about 12 h, so scans take a neuron list (`neurons=`); a full-layer scan is a deliberate overnight job.

## Run 1 verdict (2026-11-09, engine 69aec4f + gain/dnll working tree, receipt `.work/scan-check/`): FAIL as pre-registered

`scripts/scan-check.sh` printed `FAIL mutant: neuron 3 single equals batch neuron 1545`. Bars 1 and 2 passed; bar 3 (mutant) failed because its tolerance
(1e-6 absolute + 1e-3 relative, bar 1) is larger than the spread of the scores: all 24 neurons scored KL 7.7e-6 to 1.2e-5, so two different neurons
"matched" within 1e-6. The 8 single re-runs are bit-identical to the batch (neuron 3: 7.776826308889081e-06 in both), identity gain scores exactly 0.

## v2 bars (post hoc annotation, written AFTER seeing run 1; the FAIL above is not flipped)

Same setup; bar 1 and bar 3 use relative 1e-6 (`tools/scan-verify.py`). Re-evaluated on run 1's saved receipts, no new run (below). A second,
independent question run 1 raised is why every sampled neuron scores about the same: `scripts/scan-scale.sh` (gain k = 0, -1, 10, 100 on neurons 3
and 3087; damage must grow with |k - 1|) is a falsifier for "the scan measures a numeric noise floor, not the neuron".

## Re-evaluation and falsifier results (written after the runs, 2026-11-09)

- v2 bars on run 1's saved receipts (`tools/scan-verify.py`): 8 singles equal the batch to 1e-6 relative, identity max 0.0, wrong-neuron mutant rejected: PASS under v2. The v1 verdict above stays FAIL.
- `scripts/scan-scale.sh` (neurons 3 and 3087, gains 0, -1, 10, 100): FAIL against its own criterion (k = 10 must score > 20x k = 0). Neuron 3: KL 7.8e-6 (k=0), 9.0e-6 (-1), 1.9e-5 (10), 1.4e-3 (100). Neuron 3087:
  1.2e-5, 2.1e-5, 2.0e-4, 2.1e-2. Damage does grow with the change at large gains, but not like (k-1)^2 near zero.
- `scripts/scan-floor.sh` explains it: neuron 3 scores exactly 0.0 at k = 0.999 (the bf16 activation rounds back to the same value, nothing changes) and 7.0e-6 at k = 0.99, then stays at 7e-6 to 9e-6 through
  k = 0.9, 0.5, 0 and -1. Any change that survives bf16 rounding of the FFN activation re-rolls the rounding of everything downstream and lands on a numeric noise floor of about 7e-6 to 1.2e-5 KL, whatever its
  size. All 24 sampled neurons of layer 16 sit on that floor (7.7e-6 to 1.2e-5, ratio 1.75): none has an effect measurable with this metric at k = 0 except as noise. Consequence for the scan: a table of KL is
  ranked noise below roughly 2x the floor, so `scan` takes `ctl=0.99` (a second pass at a gain that cannot matter) and ranks by `excess = kl - kl_ctl`. Gain 100 shows real effects (neuron 3087: 2.1e-2).
- Cost: 57 ms per 10-token request, about 3.7 s per neuron at 64 prompts, so a full 12288-neuron layer is about 12.6 h per pass; the scan takes a neuron list.
