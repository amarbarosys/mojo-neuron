# Prereg lens-diff b: same tool, fresh pairs, case-insensitive markers (written before the run, 2026-11-09, lane agent)

Why: the `docs/prereg-lensdiff.md` run (20 pairs, `qwythos-enfr`) FAILED bar 2. Receipt `.work/lensdiff/fe{15,23,27}.json`: L27 top has 6 French markers and
0 English, null 0, consistency 1.0; but the bottom holds 0 of the preregistered lowercase English markers. The bottom tokens are English sentence starters
(` The`, ` It`, ` While`, ` Although`, ` whose`, ` either`, `though`): the last token is a full stop, so the lens reads the NEXT token, which starts a sentence and is
capitalised. The first list (lowercase, mid-sentence function words, copied from F3) could not see that. That run is not changed and stays a FAIL.
Changes, all fixed here before the run and informed by that run's bottom tokens (said in the report): markers match case-insensitively with the leading space
ignored (`lower(trim(token))`), and each list gains sentence-start function words. Everything else (bars, null, mutant) is the old prereg.
New data: `.work/cap/qwythos-enfr60` (the 61-pair capture of `tests/data/en-fr-60.tsv`), pairs 21..61 only (French requests 21..61, English requests 82..122): 41 pairs,
none of them in the failed run (its pairs were 1..20). Layers 23 and 27 (either passes).
French markers: le la les de des du et est un une que pour dans qui pas sur avec au aux je il elle nous vous ce cette ces mais ou où donc car ne se sa son ses leur
nos mes mon ma ton ta.
English markers: the of and is to that with for in are was on at by it this these those while although though whose either i we you he she they could would can
will not but or if when as from there their his her.
Bars (all hold, a miss = FAIL):
1. Antisymmetry and `A` against `A` zero (as the old bar 1), both layers.
2. FR - EN, k = 50, layer 23 or 27: top >= 4 French and <= 1 English markers; bottom >= 4 English and <= 1 French.
3. Null (20 centred sign flips) at the passing layer: French markers in the top 50, mean <= 1.0, max <= 3 with the new list.
4. Consistency >= 0.9 at the passing layer.
5. Mutant `-D LENSDIFF_MUT=1` must FAIL bar 2.
6. Normal rlog line per call.

Receipts: `.work/lensdiff/b/*.json`, `tests/lensdiff.sh b`.
