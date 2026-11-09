# Prereg C2b: dial calibration (written before any run; folds in docs/reports/2026-11-09-papers-synthesis.md rows C2/E)

Rows used: steering strength is an inverted U (sweep, pick the peak on held-out, report side-effect NLL); always run the prompt-only baseline (prompting beats steering
in several studies); random-direction control. C2 (docs/prereg-C2.md) is not touched.

**Setup.** Qwen2.5-0.5B bf16 on the spark engine (BARO_STEER, last-position logits via BARO_DUMP_LOGITS, one resident sweep process through gpu-wait). Direction
v = the dial export of two neurons (L12 first list row +, L16 first list row -, as in tests/dial.sh) times a rel dose c in {0, +-0.05, +-0.1, +-0.2, +-0.4, +-0.8}.
Targets T = the 5 tokens the blend preview promotes most. Prompts: 32 neutral prompts from tests/corpus2.txt, split by line parity: 16 tune, 16 held out.
Next-token truth for the NLL = the word that follows in the corpus.

**Metrics (per c, mean over prompts).** lift = mean over T of (log p steered - log p unsteered) at the next position; side = NLL of the true next token steered minus unsteered (nats);
kl = KL(unsteered || steered). The peak c* is chosen on the tune half only; every number reported below is on the held-out half.

**Arms.** steer (above); prompt = no steering, the prompt gets the prefix "Use words like t1, t2, t3, t4, t5." (same metrics); random = a random direction of the same per-layer norm as v, same c grid (control).

**Bars (all preregistered; a miss is reported, not re-cut).**
1. Inverted U: lift(c) rises, peaks at an interior c* (not +-0.8), and falls after it on the held-out half; side(c) grows with |c|.
2. At c*: lift >= +0.5 nats and side <= +0.3 nats.
3. Negative doses: lift(-c) < 0 for every c >= 0.1 (the sign is the lever, not just any perturbation).
4. Random control: max over c of lift_random <= 0.1 nats.
5. Prompt baseline decides the claim: "steering beats prompting" is written only if lift_steer(c*) > lift_prompt AND side_steer(c*) <= side_prompt + 0.1; otherwise the report says "prompt wins" or "tie" with the numbers.
6. Layer sweep: repeat the grid with the L12 neuron moved to L8 and L16 (3 layers); report the layer x c table; the peak layer is chosen on the tune half.
Optional (build only after bars 1-6 ran): CAST condition gate, apply the blend only when a condition direction fires (papers synthesis, ~60 LOC).
