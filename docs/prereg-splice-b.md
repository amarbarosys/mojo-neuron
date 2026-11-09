# Prereg splice-b: the effect of the splice op is measured by KL (written 2026-11-09 AFTER run 1 of docs/prereg-splice.md; post hoc by construction)

Run 1 FAILED its bars 3 and 4 as written: they asked for the mean NLL to rise by >= 0.05 nats under a zero splice and under shifted rows. Measured (layer 16, 40 positions): NLL 2.1457 unspliced, 2.1456 identity splice, 2.1441 zero splice (-0.0015),
2.1748 shifted rows (+0.029). The op acts (mean KL against the unspliced run: identity 1.1e-5, zero 3.4e-2, shifted 4.1e-2, zeros file equals `splice_zero` bit for bit) but the FFN of layer 16 moves this text's mean NLL by less than the 0.05 guessed in the prereg.
The run-1 verdict stays FAIL. Bars 3 and 4 of this document replace the NLL threshold by KL and are applied to run 1's saved receipts (no new run: the engine is deterministic and the numbers would repeat); that makes this an annotation of run 1, not an independent verdict.
Bars 1, 2 and 5 are unchanged. Bar 3b: zero splice mean KL >= 1e-2 and >= 100x the identity splice's, `splice_zero` == zeros file exactly (NLL and KL). Bar 4b: shifted-rows mean KL >= 1e-2 and >= 100x the identity splice's. NLL changes are reported, not barred.
Lesson for lane B: use KL against the unspliced run as the splice-effect metric; mean NLL on a short text is too insensitive at one layer.
