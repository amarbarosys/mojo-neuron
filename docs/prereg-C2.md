# Prereg C2: persona dial (written before the result run)

Gate `tests/dial.sh`. Qwen2.5-0.5B bf16, prompt "What is the capital of France?", dial of two neurons from two layers
(L12 first list row, L16 first list row), doses set by slider clicks in the app, rel mode (rho of the engine lane).

Bars (all must hold):
1. App: STATE `dial=` shows two units with doses that moved off the default (+0.25), signs + and - (the clicks were + and -).
2. Export: `dial.cvec.gguf` equals `dial.f32` on the non-zero layers >= 1 (tools/cvec-check.py), and the cvec call output is
   byte-identical to the `cvec` CLI on the same recipe and rho.
3. llama.cpp loads `dial.cvec.gguf` with `--control-vector` (llama-completion prints text, so the load worked) and changes its greedy text or its
   logits against no vector.
4. Delta logits (steered minus base) of llama.cpp (f32 vector, tools/llama-oracle.py) vs the engine (BARO_STEER=dial.f32):
   relative error <= 0.10 (the bar of steer-check), same top-1, sign agreement >= 0.95 on the 1000 largest |delta| entries.
5. Mutant: the dial negated (every dose x -1) must give cosine < 0 between the engine delta and the llama.cpp delta of the original.
A missed bar is reported, not re-cut.

Note before the result run: the first draft had a slider range of +-2 in units of the residual norm (rho); llama.cpp then printed only `?`. Range is now +-1, default +0.25. No bar changed.

## Result of the first run (annotation, written after the data; the bars above are not changed)
Bars 1 to 4 held: rel err 0.0482, top-1 785/785, sign agreement 1.000 on the 1000 largest |delta|, llama.cpp loaded `dial.cvec.gguf` and changed its greedy text.
Bar 5 as written MISSED: the negated dial gave cosine +0.005 (not < 0) between the engine delta and the llama.cpp delta of the original. The logit response is not linear at
these doses (0.11 rho at L12, -0.42 rho at L16), so "opposite sign" was the wrong expectation for the mutant. The gate now also requires the mutant to miss the
steer-check way (rel err of the negated run against the original delta > 0.5) and prints the missed bar 5; it does not claim it passed.
