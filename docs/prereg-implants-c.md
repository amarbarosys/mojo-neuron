# Prereg G0 amendment C: T1 has no mistakes, T1b replaces it (written 2026-11-09 after the T1 mining run, before any T1b run)

Receipt: `.work/impl-run/run-mine/out.jsonl` (engine neuron2 build, Qwythos-9B Q4_0 pack, greedy, 6 tokens): the unmodified model answered all 3645 two-digit additions with a carry correctly (0 mistakes; samples ` 71.`, ` 82.`, ` 176.`),
and the 100 unrelated items (10 trivia and 90 non-carry additions) all right. Under G0 this family has fewer than 30 test mistakes: **T1 gets no verdict (UNDERPOWERED, zero mistakes)**. That is a result, not a failure of the harness.
Change: the generalisation family is T1b, two-digit multiplication `Q: What is a * b? A:` with a, b in 10..99 (8100 distinct problems, shuffled with seed 1, first 4000), mined, split (50/20/30) and run exactly as T1 was specified. Everything else in G0 (splits,
k sweep, random control, bars G-A to G-D, the layer chosen on dev, statistics) and in `docs/prereg-papers-fold.md` is unchanged. The unrelated set stays the one mined in the T1 run. If T1b also yields fewer than 30 test mistakes, it gets no verdict either and T3 (code idiom) is the
next family under the same rules. T2 (capitals): 50 countries in `tests/data/capitals.tsv`; expected to yield too few mistakes, not run unless T1b and T3 fail.
Memorisation (verdict 2) and G4 do not need mistakes; they need the layer, which is chosen on T1b's dev split.
