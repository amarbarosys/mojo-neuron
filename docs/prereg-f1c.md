# Prereg F1c: persona by name through baro-serve (E1 verified at baro neuron2 7c37795)

Adds to `prereg-f1.md` (committed, not changed). E1's `steer` object `{"file": PATH}` forwards to the engine's `steer_file`; the persona store resolves
a name to that object. The server stays unaware of personas.
- `nx persona apply {name}` returns the `steer` field to send (`{"file": <steer.f32>}`); `nx chat --persona NAME` sends it (and refuses
  `--persona` together with `--steer`). A persona whose vector sha changed is refused (existing F1 check).
- Engine gate `scripts/persona-serve-check.sh`: engine and baro-serve REBUILT FROM 7c37795 (the verified binaries in `mojo-baro-lanes/verify`, whose build
  commit the gate prints), one gpu-wait job, Qwythos-9B, recipe `8:100:30,20:50:120`:
  - HTTP `/v1/completions` with `steer: <persona's field>` vs the engine run directly with `steer_file` = the recipe vector built by `cvec`:
    max abs diff of the top-20 logits <= 1e-3 (E1's own tolerance, reached 0 there).
  - the same HTTP request without `steer` differs from the persona request by >= 0.1 on the shared top-k tokens (the persona acts).
  - a second persona (recipe `20:50:120`) over HTTP differs from the first by >= 0.1 (the name picks the vector, not a fixed one).
- `tests/chat.sh` (mock server): `nx chat --persona calm` posts `steer: {"file": ".../calm/steer.f32"}`; `--persona` with `--steer` exits non-zero.
- Every call ends with an Rlog line.
