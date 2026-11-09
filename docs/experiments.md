# Experiments (lane engine, after the tool was done)

CPU only, on artifacts already on disk (GPU budget left for both lanes: 18 min). Each entry was written, hypothesis and falsifier first,
and committed before its number was computed; the result is a later commit. Data: Qwen2.5-0.5B bf16, layer 12, steer lens at +-0.5 relative
dose over gne's four contexts (`.work/bench2/12.steer.{0.5,-0.5}.bin`, `.kl.f32`, k = 12, all 4864 neurons), persona contrast
`.work/rv1/contrast.f32` (pirate system prompt minus helpful-assistant, 6 prompts). Analysis code: `tools/experiments.py`.

## X1. Does the weights-only logit lens predict the steer lens?

Hypothesis: the tokens a neuron's down-projection column promotes through the unembedding (lens top-12) reappear among the tokens whose
log-prob the steer lens raises most (steer top-12 up, +0.5), well above chance. If true, the 5-second CPU lens is a usable pre-filter for the
GPU steer lens.
Cheapest falsifier: fraction of neurons sharing at least one token between lens top-12 and steer-up top-12, against the same fraction
with the lens rows shuffled across neurons (500 shuffles). Hypothesis holds if observed >= 3 x shuffled mean; otherwise it is falsified.
Lens: z = W_U . (unit(W_down[:, n]) * output_norm), no steering, numpy f32 from the bf16 GGUF.

## X2. Is the effect of a neuron approximately antisymmetric in the dose?

Hypothesis: in the +-0.5 regime pushing a neuron is close to linear, so the tokens raised at +0.5 are the tokens lowered at -0.5.
Cheapest falsifier: over the neurons whose KL(+0.5) is above the median of the 64 random controls' KL, the overlap between steer-up top-12 at
+0.5 and steer-down top-12 at -0.5. Hypothesis holds if a majority of these neurons have overlap >= 6 of 12 AND KL(+)/KL(-) lies in [0.5, 2]
for >= 80 % of them; if either fails, linearity is falsified (and the number says how far).

## X3. Do the neurons most aligned with a persona direction raise that persona's words?

Hypothesis: the 20 layer-12 neurons with the highest cosine to the pirate-minus-assistant residual difference (d_12) raise pirate vocabulary
in the steer lens more often than 200 random layer-12 neurons.
Cheapest falsifier: share of neurons whose steer-up top-12 (+0.5) contains a token whose text, lowercased and stripped, is in the lexicon
{arr, arrr, matey, ahoy, aye, ye, pirate, pirates, ship, sail, sea, treasure, captain, plunder, seas}; enrichment = share(top-20) / share(random 200,
mean of 200 draws). Hypothesis holds if enrichment >= 2; falsified otherwise. The steer-lens contexts are gne's neutral ones, so a null is
a plausible outcome and is reported as such.

### X2 result (falsified)

`tools/experiments.py x2`: of the 2880 neurons whose KL(+0.5) is above the controls' median, only 0.2 % have >= 6 of 12 tokens shared between the tokens raised at
+0.5 and the tokens lowered at -0.5 (median overlap 0, mean 0.14 tokens; shuffling the down rows across neurons gives 0.008, so there is a weak real link, about 17 x
chance, but nowhere near antisymmetry). KL(+)/KL(-) lies in [0.5, 2] for 63.4 % (threshold 80 %), median ratio 1.44: pushing up costs more KL than pushing down.
Both criteria fail. At the doses a persona search uses (+-0.5 of the residual norm) a neuron's effect is not close to linear; a recipe with a negative
dose is a different experiment from the positive one and needs its own steer lens file (which the format already has: one file per dose).

### X1 result (criterion met, effect small)

`tools/experiments.py x1`: 5.82 % of the 4864 neurons share at least one token between the lens top-12 and the steer-lens up top-12 at +0.5; with the lens rows shuffled across
neurons it is 0.37 % (500 shuffles): ratio 15.9, above the pre-registered 3. The hypothesis holds as stated, but the effect is small: 94 % of neurons share nothing, mean 0.099 shared
tokens of 12. Post hoc, not pre-registered: among the 5 % highest-KL neurons the share is 9.8 %, among the rest 5.6 %. So the weights-only lens is a real but weak predictor: it can
rank which neurons are worth a steer lens only a little; it does not replace it. What it does not capture is the part of the effect that goes through later layers' attention and MLPs
(the lens reads one matrix product).

### X3 result (falsified as stated, and underpowered)

`tools/experiments.py x3`: 0 of the 20 neurons with the highest cosine to the pirate-minus-assistant direction (cosines 0.21 down to 0.155 for the top five) raise any lexicon token;
enrichment 0 against the pre-registered 2, hypergeometric p = 1.0. The criterion fails, but the test cannot say much: only 7 of 4864 neurons raise a lexicon token in their
top-12 at all (0.14 %), so even a 10-fold enrichment predicts 0.3 hits among 20. The contexts (gne's four neutral ones) never invite pirate text. A null here is "no evidence", not
"evidence against". X3b below repeats it where the effect has room to show.

## X3b. Same question with contexts that invite pirate text (follow-up, written before running)

Hypothesis: with four pirate-inviting contexts ("Ahoy! Tell me about", "The pirate captain said", "Yo ho ho and a bottle of", "On the high seas we"), the 100 layer-12 neurons with
the highest cosine to d_12 raise lexicon tokens in the steer lens (+0.5, top-12 up) at >= 2 x the rate of 400 random layer-12 neurons.
Cheapest falsifier: one steer-lens run of 500 neurons (a few seconds of GPU); enrichment of the top-100 over the random-400 rate with a one-sided hypergeometric p < 0.05.
Holds if enrichment >= 2 and p < 0.05, falsified otherwise. Lexicon as in X3.

### X3b result (falsified by the pre-registered rule; counts too small to conclude)

`tools/experiments.py x3b` on one steer-lens run of 500 neurons (5.3 s of GPU): 2 of the top-100 neurons by cosine to d_12 raise a lexicon token (2.0 %), 3 of 400 random
neurons do (0.75 %): enrichment 2.67, one-sided hypergeometric p = 0.26. The rule needed enrichment >= 2 AND p < 0.05, so the hypothesis is not supported. The two hits are
both the token "pirates" (neurons 148 and 540), which is the kind of hit the hypothesis predicts, but 2 versus 3 events cannot separate it from chance. A test with power needs many
more persona-aligned neurons (more layers, or a persona with a larger vocabulary footprint than a handful of single tokens) and pirate contexts rather than four.

## Summary

| | hypothesis | pre-registered rule | result |
|---|---|---|---|
| X1 | the weights-only lens predicts the steer lens | >= 3 x shuffled | holds, 15.9 x, but only 5.8 % of neurons share a token |
| X2 | +-0.5 is antisymmetric | majority >= 6/12 and 80 % KL ratio in [0.5, 2] | falsified: 0.2 % and 63 % (median ratio 1.44) |
| X3 | persona-aligned neurons raise persona words (neutral contexts) | enrichment >= 2 | falsified, no power (7 hits in 4864) |
| X3b | same, pirate contexts | enrichment >= 2 and p < 0.05 | falsified: 2.67, p 0.26, 2 vs 3 events |

GPU used by the experiments: one steer-lens run of 5 s. Everything else was CPU on existing files.

# Experiments

Each entry: hypothesis, cheapest falsifier, how to run it, result (a null counts), caveats. All CPU; no GPU minutes. Model: qwen2.5-0.5b-instruct q4_K_M (24 layers, FFN 4864) unless stated. Store: `MNEURON_DIR` (the shared `.work/neurons` or a lane copy).

## E1. Does the logit lens of a neuron predict what steering it does?

**Hypothesis.** The tokens a neuron's write direction promotes through the unembedding (lens top) are among the tokens whose next-token log-probability rises most when the neuron is pushed up (+dose) in a real forward pass.

**Cheapest falsifier.** Pair every neuron's lens with the steer record of a different neuron. If the overlap with its own record is not above that chance level, the lens says nothing about steering.

**Run.** `exp_steer MODEL 12 12.steer.0.5.bin 12.steer.-0.5.bin` (`src/exp_steer.mojo`) on the engine lane's full-layer steer lens of layer 12 (`engine/.work/bench2`, all 4864 neurons, k=12, contexts: 4 short prompts, last-position mean). The lens is from the q4_K_M file and the steer lens from the bf16 file. Chance: the same count with the record of the neuron n+d for 20 shifts d.

**Result.**

| pairing | neurons with at least one shared token | chance | strongest 5% of lens scores |
|---|---|---|---|
| lens top vs rise at +0.5 | 4.67% | 0.26% (18x) | 15.2% |
| lens bottom vs rise at -0.5 | 4.30% | 0.25% (17x) | 7.4% |
| lens top vs fall at +0.5 (sanity) | 0.06% | 0.21% | 0.4% |
| lens bottom vs fall at +0.5 | 1.73% | 0.25% (7x) | 2.9% |

The lens points the right way: 17 to 18 times chance, and the sanity pairing (promoted tokens that should fall) sits below chance. The effect is small in absolute terms: 95% of neurons share no token at all, at layer 12 of 24 and with the k=8 lens against k=12 steering. The strongest lens scores predict better (15% of the top 5%). Only layer 12 was measured: the other steer files hold 200 neurons or fewer.

## E2. Does a neuron fire when the token it promotes comes next?

**Hypothesis.** If a neuron promotes tokens T, then at the positions where it fires hardest, the next token of the text is in T more often than for a random neuron's T. The effect should grow with depth, because late layers write closer to the output.

**Cheapest falsifier.** Score the same firing positions with the lens of another neuron (10 shifts). Lift = hits / chance; lift near 1 means the lens does not predict firing.

**Run.** `exp_next MODEL.gguf tests/corpus.txt LENSMODEL LAYER` (`src/exp_next.mojo`): CPU forward over 19 windows of 128 tokens (2432 positions, 2413 with a next token in the window), per neuron the 24 positions (1%) with the largest positive activation (next token against lens top) and the 24 most negative (against lens bottom, because a negative activation flips the write direction). About 233,000 firing pairs per layer, 18 s per layer. Sampling noise: the chance counts are about 100 to 190 hits, so a lift under 1.2 is noise.

**Result.**

| layer | next in lens top at positive firings (hits / chance / lift) | next in lens bottom at negative firings | current token in lens top |
|---|---|---|---|
| 4 | 162 / 142 / 1.14 | 195 / 180 / 1.08 | 0.84 |
| 8 | 160 / 191 / 0.84 | 150 / 184 / 0.82 | 1.16 |
| 12 | 178 / 170 / 1.05 | 187 / 175 / 1.07 | 1.11 |
| 16 | 289 / 167 / 1.73 | 273 / 173 / 1.58 | 2.03 |
| 20 | 466 / 154 / 3.03 | 456 / 143 / 3.20 | 2.06 |
| 22 | 325 / 92 / 3.53 | 239 / 77 / 3.11 | 2.55 |

Null for layers 4 to 12 (lift within noise of 1), a clear effect from layer 16, about 3x from layer 20. Negative firings work the same as positive ones through the lens bottom, so the sign flip is real. The lens is a good guide to what a neuron does only in the last third of this model. The current-token column is high from layer 16 too: late neurons both read and write the tokens in their lens, so part of the lift is token repetition in the corpus, which this design does not separate. One model, one 13 KB corpus (the plan document), so topic words recur.

## E3. Does the R-style (ridge) lens predict firing better than the logit lens?

**Hypothesis.** Reading a write direction through a fitted residual-to-final transport map (`rlens`, our R-style lens) predicts the next token at firing positions better than reading it directly, most at middle layers where the residual is still far from the output basis.

**Cheapest falsifier.** The E2 measurement with `rlens.bin` instead of `lens.bin`, same positions, same chance control.

**Run.** `rlens MODEL tests/corpus.txt --layers 12,16,20,22 --windows 36 --holdout 4` (26 s), `lens --rmap` per layer, then `exp_next ... LAYER 40 r`.

**Result.**

| layer | logit lens: next-top hits / chance / lift | R lens: hits / chance / lift | R lens, bottom at negative firings (lift) |
|---|---|---|---|
| 12 | 178 / 170 / 1.05 | 329 / 228 / 1.44 | 1.30 (logit 1.07) |
| 16 | 289 / 167 / 1.73 | 322 / 162 / 1.98 | 2.04 (logit 1.58) |
| 20 | 466 / 154 / 3.03 | 421 / 107 / 3.92 | 3.40 (logit 3.20) |
| 22 | 325 / 92 / 3.53 | 242 / 74 / 3.29 | 2.93 (logit 3.11) |

Mixed. At layer 12 the R lens turns a null into a modest effect (329 hits against 228 chance, about 6 sigma), and the gain shrinks with depth; at layer 22 it is slightly worse than the logit lens, as expected when the transport map approaches the identity and adds fitting noise. At layer 20 the lift is higher but the raw hit count is lower (421 against 466), because the R readout promotes rarer tokens, which lowers the chance level: lift alone overstates that gain. Caveat: the map was fitted on the first 15 windows of the same corpus that the firing positions come from (the last 4 were held out for the fit's own check). The fit uses residuals, not next tokens, so leakage is mild, but a clean test needs a second corpus. Not tested: the LRP-Jacobian R lens (not built).

## E4. Do neurons that are connected by weights fire together?

**Hypothesis.** If the write direction of neuron n (layer L) points at the gate or up row of neuron m in layer L+1 (large weight cosine, `connections`), then a[m] moves with a[n] over text, with the sign of the cosine.

**Cheapest falsifier.** The same correlation against random neurons of layer L+1. A signed correlation near zero and 50% positive pairs means the direct-path cosine says nothing about co-firing.

**Run.** `exp_conn MODEL.gguf tests/corpus.txt LAYER` (`src/exp_conn.mojo`): 2432 corpus positions, every 8th neuron (608), its strongest `downstream` connection (span 1, gate or up), Pearson correlation over positions times the cosine sign, against 20 random targets per neuron. 20 s per layer.

**Result.** Null at every layer measured.

| layer | mean signed correlation (connected / random) | fraction positive | strongest quarter by cosine (cos above about 0.27): mean / fraction positive |
|---|---|---|---|
| 4 | 0.002 / 0.001 | 51.0% | -0.014 / 46.7% |
| 8 | 0.003 / 0.000 | 52.0% | 0.014 / 54.6% |
| 12 | 0.005 / -0.000 | 52.1% | -0.006 / 54.6% |
| 16 | -0.012 / -0.000 | 49.5% | -0.033 / 46.7% |
| 20 | 0.005 / -0.001 | 51.8% | 0.008 / 50.7% |

With 608 pairs a fraction of positives has a standard error of 2 points, so nothing here is outside noise, including the strongest quarter (152 pairs, 4 points). The weight cosine is a path through the residual only, with an RMS norm and the next layer's attention in between; it does not predict that two MLP neurons are active on the same tokens. The `connections` view is a statement about weights, not about behaviour. Not tested: Spearman or a cosine through the norm, and non-adjacent layers.

## E5. Do E2 and E3 hold on a text the R maps never saw?

**Hypothesis.** The depth trend of the logit lens (E2) and the R lens gain at layer 12 (E3) are not artefacts of the corpus: they replicate on a different text, where the ridge maps (fitted on `tests/corpus.txt`) are fully held out.

**Cheapest falsifier.** Rerun `exp_next` with `tests/corpus2.txt` (READMEs of mojo-gguf, mojo-json, mojo-imkit; 2560 positions, 2540 with a next token), same maps, same control.

**Result.** Replicates.

| layer | logit lens: next-top hits / lift, bottom lift | R lens: next-top hits / lift, bottom lift |
|---|---|---|
| 12 | 176 / 0.96, 1.10 | 213 / 1.33, 1.72 |
| 16 | 382 / 2.23, 2.03 | 327 / 2.32, 1.95 |
| 20 | 524 / 3.49, 3.66 | 415 / 4.07, 3.47 |

The logit lens is null at layer 12 and rises to 3.5x at layer 20 on both texts (E2: 1.05 and 3.03). The R lens again turns the layer-12 null into 1.3x (top) and 1.7x (bottom), now with the maps fully held out, so the fit-on-same-text caveat of E3 does not explain it. At layer 20 the R lens has higher lift on top but fewer raw hits (415 against 524) and a lower bottom lift: a lower chance level, not a better predictor. So the supported claim is narrow: the R lens helps in the middle of the model, not at the end. The corpus is technical text about code, so topic words recur; a prose corpus would be a better second test. One model.
