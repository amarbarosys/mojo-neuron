# Prereg A2b: ablation scan with a noise-floor control (written before its result run, 2026-11-09)

Replaces the tolerance of `docs/prereg-a2.md` bars 1 and 3 and adds the floor control. The failed run 1 of that prereg stays reported there as FAIL; this document is a new experiment, run once on a fresh engine run.

Diagnosis that motivates it (from run 1 and `scripts/scan-floor.sh`, already reported): KL of a neuron ablation against the unablated engine sits on a bf16 numeric noise floor of about 7e-6 to 1.2e-5 for any change that
survives bf16 rounding of the FFN activation (gain 0.999 -> exactly 0.0, gain 0.99 -> 7e-6, flat down to gain -1). The 24 neurons of run 1 sit on it.

Setup: Qwythos-9B Q4_0 pack, engine neuron2 (gain and dnll commit 40f9226), layer 16, the 64 prompts of `tests/data/eval64.txt`, `scan ... ctl=0.99` over 16 neurons (the 24 of run 1 less 8), k = 0.
Bars (any miss = FAIL):
1. 4 of the 16 neurons re-run alone in separate engine processes: kl, kl_ctl and dnll equal the batch to 1e-6 relative.
2. Identity gain (k = 1) on neurons 5 and 6 scores KL <= 1e-9, and gain 0.999 on neuron 5 scores KL <= 1e-9 (a change that bf16 rounds away is a no-op).
3. Mutant: a single-neuron result differs from another neuron's batch value by more than 1e-6 relative (compared per column, both kl and kl_ctl).
4. Floor control: kl_ctl of every neuron lies within 0.5x to 2x of the median kl_ctl (the floor is a property of the engine, not of the neuron); the report states the floor and calls a neuron detectable only when its excess `kl - kl_ctl` exceeds twice the median kl_ctl.
5. rlog: one normal line per scan run.
Not claimed: that any of the 16 neurons is detectable (run 1 suggests none are), only that the scan reproduces and separates floor from effect.

## Result (2026-11-09, one run, engine 40f9226 binary, receipt `.work/scan-check/`): PASS, all five bars

`scripts/scan-check.sh` printed `PASS`: 16 neurons of layer 16, floor (median kl_ctl) 7.671e-06, kl 7.710e-06 to 1.211e-05, no neuron detectable (excess over twice the floor: none). The 4 single-process re-runs equal
the batch to 1e-6 relative on kl, dnll and kl_ctl; identity gain and gain 0.999 score exactly 0; the wrong-neuron mutant is rejected in both columns; every kl_ctl lies within 0.5x to 2x of the median; 7 normal rlog lines.
Read: no sampled neuron of layer 16 has an ablation effect this metric can separate from the engine's numeric floor on these 64 prompts. That is a statement about these 16 neurons, not about the layer.
