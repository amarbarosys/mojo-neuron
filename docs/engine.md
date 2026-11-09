# Engine lane: what ships and how each piece is checked

Everything runs on mojo-baro's `ablate` branch (spark engine, one resident process per sweep) and this repo's Mojo
tools. Python appears only as oracle code in `tools/`.

## Tools (this repo)

| tool | does | gate |
|---|---|---|
| `cvec MODEL L:N:dose,... OUT [RHO.f32]` | writes `OUT.cvec.gguf` (llama.cpp `--control-vector`) and `OUT.f32` (the `BARO_STEER` file) | `steer-check.sh` |
| `sweep MODEL OUT layer= neurons= doses= prompts= ...` | rollouts (`mode=sweep`), batched steer lens (`mode=slens`), neuron gain ablation (`gains=`), random controls (`ctrl=N`) | `sweep-check.sh`, `slens-check.sh`, `gain-check.sh` |
| `reverse MODEL OUT prompts= sys_a= sys_b=` | persona contrast, cosine rank of every neuron of every layer | `reverse-check.sh` |
| `lens` (step 1, app lane extends it) | logit lens of every neuron of a layer | `check.sh` |

Build: `scripts/build-sweep.sh` (sweep, reverse), `scripts/check.sh` (lens), `cvec` is built by `steer-check.sh`.
File formats: `docs/format.md`. Numbers against gne: `docs/bench.md`.

## mojo-baro hooks (branch `ablate`)

- `BARO_STEER=FILE` (`f32[N_LAYERS * H]`, row L is added to the residual after layer L; the llama.cpp control vector
  position). spark fuses the add into the next RMSNorm launch (no extra launch, no host sync), also in batched prefill.
  engine.mojo (qwen35 family) adds the vector after each layer in `step_window` and runs with the megakernel,
  speculation and batched prefill off while it is set.
- spark serve requests: `steer_bank` + `steer_layer` + `steer_row` + `steer_scale` (a row of a bank file times a scale),
  `steer_file`, `gain "L:N:k,..."`, `op: slens` (the batched steer lens).
- `BARO_STEER_SERVE=1` allocates the steering buffer for serve mode.

## Limits

- Layer 0 cannot be steered through llama.cpp (and the cvec file skips it); baro can.
- The steer lens op needs a plain dense profile (no gate, no sliding window), contexts up to 96 tokens, f32 activations.
- `gain` needs f32 activations and runs token by token (no batched prefill).
- MoE experts are out of scope; engine.mojo steering works on MoE layers only as a residual add.
