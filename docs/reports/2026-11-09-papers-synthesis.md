# Papers synthesis: 198 interpretability papers for mojo-neuron (2026-11-09)

## VERDICT
Four reader notes (`.work/papers/notes-{aa,ab,ac,ad}.md`, ~50 papers each, first 12 pages) agree on five protocol
changes that touch running items; none contradicts the v2/F/G design. Claims are the readers', not re-checked.
-> this file. Open risk: arXiv ids and numbers come from pdftotext extracts read by Sonnet, spot-check before citing.

## Agreed across batches (2+ readers independently)
| change | readers | plan item | LOC |
|---|---|---|---:|
| Pre-rank neurons by a cheap linear direct effect (activation x down_proj -> target logit via lens; AtP/EAP style), then run the exact `gain` ablation only on the top slice | aa, ac, ad | A2 | 60 |
| Print the ablation mode (zero / mean / resample, clean / corrupt) and the metric (logit diff) with every scan result and rlog line; scores change with it | aa, ab, ad | A2, A4, F27 | 15 |
| Always run the prompt-only baseline; prompting beats steering, probes and SAE edits in several studies | ab, ad | G3, G4, C2, F3 | 40 |
| Report transcoder seed stability and neuron-feature share; one run is one sample | aa, ab, ac | B1 | 60 |
| Labels: accept a label only if steering with it moves the target; score precision and recall on non-top contexts; check the labeller against synthetic neurons with known labels | aa, ab, ac, ad | B4, F30 | 100 |
| Lie probe test set: paraphrased / rewritten statements, separate transform families, sycophantic and inverted lies, a calibration-matched baseline; layers 40-70 % depth, equal-weight depth mean | aa, ab, ac, ad | F24 | 70 |
| Implants: run a regression set (locality + general ability + earlier implants) after every implant; edits collapse in long chains | ac, ad | G1, G2, G3 | 50 |
| Cross-model maps: compare effects (steer-effect correlation, stitching NLL), not token overlap; neuron-level maps under-report | aa, ab, ac, ad | B3, F5, F25 | 50 |
| Steering strength is an inverted U: sweep layer x coefficient, pick the peak on held-out, report side-effect NLL | ab, ad | C2, F1, F26 | 50 |

## Single-reader ideas worth a slot (NEW)
- Tuned lens (per-layer affine fitted on A1 captures): fixes logit-lens bias. ~150 LOC, new lens mode.
- Contrastive lens `lens-diff A B` (hidden difference through W_U). ~30 LOC, B2/F3/F25.
- CAST condition gate: apply a persona steer only when a condition direction fires. ~60 LOC, C2/F1.
- Implant as localized context distillation (teacher = prompt + input, student = input; distil spans where they disagree). ~250 LOC, G2/G4 alternative.
- Key-routed implant bank (one small adapter per fact or skill, selected by key). ~90 LOC, G2.

## Pitfalls to carry into every brief
Patching results depend on corruption and metric; pruned concepts return after retraining (A3 claims "removed until
retrained"); steering vectors flip sign on ~half of inputs for some datasets; quantised models can know the truth and
still output a falsehood (probe on Qwythos Q4 may disagree with its answers by design); logit lens at middle layers is
biased toward the dominant script; label graders can be steered by injected text.

## Skill
`(private notes)`: patching, steering eval, label check, truth probe, implant check.
