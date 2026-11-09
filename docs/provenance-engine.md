# Provenance (lane engine)

Ideas, not code, come from two MIT projects, both clean-room reimplemented in Mojo; nothing was copied.

- gguf-neuron-explorer (`gne`, MIT): neuron push as a residual add, `rel` dose scaled by residual norm, steer lens
  (next-token log-prob change, KL against random-direction controls), reverse lookup by cosine to a persona contrast,
  neuron gain (`xK`), recipe string `L:N:dose`. We read its README and docs to list features and its
  `lens.py`/`steer.py` for the definition of dose units and the default contexts; our code shares no lines with it.
  It is also the benchmark opponent (docs/bench.md), run unmodified.
- neuron-explorer (MIT): logit lens per neuron, word-cluster mass; belongs to the app lane.

Our own pieces: the fused steer RMSNorm, the micro-sequence kernels and the batched steer lens in mojo-baro, the
resident sweep driver, the cvec writer, the oracles in `tools/` (python, oracle only).
