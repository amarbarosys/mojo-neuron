# App-owned files (additions to docs/format.md, lane engine's file)

`.work/neurons/<model>/rlens.bin`: ridge transport maps of the R-style lens. `"MNRL" | u32 version=1 | u32 hidden | u32 count`, then per
fitted layer `u32 layer | f32 M[hidden][hidden]` (row i = output i, so `t = M g`). Written by `rlens`, read by `lens --rmap`.
`.work/neurons/<model>/<layer>.rlens.bin`: the R readout of every neuron, same layout as `lens.bin` (magic `MNLN`).

The R lens here is ours: M_l = ridge regression from the residual after block l to the residual after the last block (before the
final norm), fitted on a text corpus with the CPU forward. It is NOT the LRP-Jacobian R lens or the J++ lens of jpp_lens
(Jacobians need a backward pass and, for J++, calibration tasks). Parity against the downloaded Qwen3.5-4B maps is UNSUPPORTED here:
no Qwen3.5-4B weights and no CPU forward for the hybrid qwen35 arch. J++ (signed expert mixture) is not built.

## live.json and live/ (app, steered chat)

`<store>/<model>/live.json`: `{"sweep":BIN,"gguf":FILE,"spark":BIN,"pack":DIR,"baro":DIR,"rho":F32FILE?,"runner":"gpu-wait run --label mneuron-live --timeout 300 --"?}`, the engine lane's `sweep` inputs for this model. `nx live` writes `<store>/<model>/live/prompt.txt` and `<layer>.sweep.jsonl` (docs/format.md rows: the neuron at -dose and +dose, plus the `"base"` row), replacing the previous live run of that layer. Ranked sweeps stay in `<store>/<model>/<layer>.sweep.jsonl`.

## dial, sound (app, C2 and C3)

`<store>/<model>/dial.f32` + `dial.cvec.gguf`: written by the `cvec` call (`out` default `<store>/<model>/dial`); same layouts as `cvec` CLI (docs in `src/cvec.mojo`).
`<store>/<model>/sound.wav`: RIFF PCM16 mono 24000 Hz written by the `sound` call. Calls added: `cvec`, `sound`, `scan` (all end in an rlog line).
