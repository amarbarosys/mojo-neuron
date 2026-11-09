# implants.bin v1 (engine `BARO_IMPLANTS`, lane causal owns this file; mechanism in `docs/prereg-implants.md`)

Rank-k side FFN per layer, applied by engine.mojo (qwen35 dense path) after the FFN of that layer. Base weights are never touched. All numbers little-endian f32/u32.

```
header   "NIMP0001" (8 bytes) | u32 h | u32 n_sets
per set  u32 layer | u32 k | f32 tau | keys f32[k*h] | gates f32[k] | values f32[k*h]
```

Keys are stored unit-length (the engine normalises again on load, so a raw key also works). `h` must equal the model's hidden size, layers are 0-based and unique.

Per token at layer l with a set: `x` = the layer's FFN input (output of the FFN RMSNorm, the row A1 captures as field `x`), `c_i = dot(x, key_i) / |x|`,
`delta = sum_i gates_i * max(c_i - tau, 0) * values_i`, added to the residual after the layer's FFN (before steering, graft and capture of the layer's `r`).

Engine knobs: `BARO_IMPLANTS=path` loads the file at start; every request uses it unless it carries `"implants":0`. A loaded file turns batched prefill, speculation
and the megakernel off (the host hook needs the layer loop, as `gain` and capture do). No file, or `"implants":0`, is bit-equal to the unmodified engine
(gate `scripts/implants-check.sh`). Cost: `k*h` multiply-adds per token and set on the host, one device-to-host and one host-to-device copy per token and set.
