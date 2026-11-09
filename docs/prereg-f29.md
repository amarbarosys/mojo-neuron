# Prereg F29: self-surgery agent, propose only (written before any code or result, 2026-11-09)

Needs A2 (ablation scan table) and A3 (`nx surgery`). Written now so the bars exist before the v2 shas arrive.

Agent = a scripted search (no LLM): input `spec.json` {behaviour regex or token list to remove, eval prompts, layer list},
steps: (1) A2 table, ranked by damage on the eval; (2) for the top K=16 units, simulate through the engine `gain L:N:k`
for k in {0, 0.5} on the eval prompts; (3) score = share of eval generations still showing the behaviour (lower is better)
plus NLL delta on 16 neutral prompts (a surgery must not break the model); (4) write the best proposal to
`.work/surgery/proposals/<id>.md`: unit, k, both numbers before/after, and the exact `nx surgery` command. No GGUF is written.

Plant: the engine runs with `gain 16:N*:8` (a planted neuron amplified 8x) that makes a lens-top token of N* appear in the
eval generations (the planted bad behaviour, an artefact of steering, not of the weights).

Bars (all must hold, any miss = FAIL):
1. Found: N* is rank 1 of the proposals, or rank <= 3 with the same layer.
2. Fix in simulation: behaviour share with the proposal's k applied <= 0.25 x the planted share, and NLL delta on neutral prompts <= 0.1 nats.
3. Control: with no plant, the agent proposes nothing (behaviour share below 0.05 -> proposal file says `no action`).
4. No write outside proposals/: `tools/nowrite.sh .work/surgery/proposals <repo> <model dir> <scan dir> -- <agent>` passes, and its
   own gate (`tests/nowrite.sh`) rejects a mutant that writes beside it.
5. Rlog: one normal line per run, an anomaly line when bar 1 or 2 misses.
