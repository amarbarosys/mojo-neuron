# Prereg F1: persona store

Bars, fixed before the result run (store now; the baro-serve pick-by-name waits for E1):
CPU gate `tests/persona.sh` (qwen2.5-0.5B):
- `nx persona save` writes persona.json, steer.f32, steer.cvec.gguf; steer.f32 is byte-identical to a direct `cvec` run of the same recipe;
  the cvec passes `tools/cvec-check.py` and llama.cpp (`llama-cli --control-vector`) loads it.
- `list` shows the persona; `apply` returns the engine env and llama flag; a flipped byte in steer.f32 makes `apply` refuse (mutant);
  a truncated cvec makes llama.cpp fail to load (mutant, proves the load check bites); duplicate save and a bad name are refused.
Engine gate `scripts/persona-engine-check.sh` (Qwythos-9B, one resident engine, three requests with per-request `steer_file`):
- logits of the first generated step with the persona's steer.f32 vs the recipe vector built by `cvec` directly: max|diff| = 0.
- the same logits vs the unsteered run: max|diff| >= 0.1 (the test is not vacuous).
- Every call ends with an Rlog line (tool `persona`).
