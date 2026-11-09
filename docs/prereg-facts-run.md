# Facts chain (verdict 2, M-A to M-C): run settings fixed before any facts result, 2026-11-10

Bars are those of `docs/prereg-implants.md` (M-A capacity, M-B interference, M-C cost); nothing here changes them. This note fixes what the prereg left open and what differs from it:
- Layer: 16, from `docs/receipts/impl-m3-select.json` (T1c dev choice, committed with this note before the run; sha256 1ff226acdaa0a9a6). Scale per k: picked by `pick_scale` on exact recall (memorisation is scored on the training prompts, so no held-out split exists for the scale); grid SCALES 0.5,1,2,4,8; k in 1,10,100,1000.
- Facts: generated in `tools/implants-harness.py` `facts()` (seed 7, invented names from 16 syllables, 1 exact question + 3 paraphrases), not read from `tests/data/facts.tsv` as the prereg wrote; no such file exists. Facts the unmodified model already answers are dropped (`facts-base`).
- Unrelated set and perplexity: the amended definitions (amendments 2 and 3 of the prereg), 400 windows of 24 tokens.
- Control: random implants of equal size (`control_sets`), same scale.
- Engine: `neuron2b/.work/engine-9b-g4`, no build. LoRA step time stays UNMEASURED unless the report cites a train log.
- Verdict file: `.work/impl-m3/verdict-facts.md`, printed unchanged.
Harness at commit 7059581 (the facts stages unchanged since the dry runs).
