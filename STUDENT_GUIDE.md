# Student guide

Written so that a third-year student can defend this work in a viva or an interview. It
explains the five ideas the project rests on, then gives twenty questions a reviewer would
actually ask, with honest answers drawn from the results and from `LOG.md`.

If you remember one thing: **the hard part of this project was not building the model, it was
noticing when a good-looking number was an artefact.** `LOG.md` records ten such occasions.
Four of them would have produced a publishable-looking but false claim.

---

## 1. The problem, in one paragraph

A drug-resistant cancer cell and a drug-sensitive one differ in which genes they express.
Suppose we could learn a model of how a perturbation (knocking down a gene) moves a cell's
expression state. Then instead of asking "what kills this cell", we could ask "what small
set of knockdowns would move this cell from the resistant expression state toward the
sensitive one". That is a *counterfactual control* question: we are searching over
interventions we have not performed, using a model to predict their effect. The danger is
obvious — if the model is wrong about an intervention we never tried, the search will
happily recommend it. So the model must also say when it does not know.

Formally: a latent state `z`, a perturbation set `S` with `|S| <= k`, a response model
`f(z0, S)` predicting the post-perturbation state, a sensitivity function `s(z)` in [0,1],
and a reversal score

```
R(z0, S) = E[s(z')] - s(z0),    z' ~ f(z0, S)
```

We want `S* = argmax E[R] - lambda * U(S)`, where `U` is calibrated epistemic uncertainty,
and we abstain when `U` is too large.

---

## 2. Flow matching (how `f` works)

**The problem it solves.** Real perturbation experiments give you *unpaired* populations: a
dish of control cells and a dish of perturbed cells. You never see the same cell before and
after, because measuring it destroys it. So you cannot train a model to map cell *i* before
to cell *i* after — there is no such pairing. You need to learn a map between two
*distributions*.

**The idea.** Think of the control distribution `p0` and the perturbed distribution `p1`.
Imagine continuously deforming `p0` into `p1` over a time `t` from 0 to 1. At each point in
space and time there is a velocity `v(z, t)` describing how probability mass is flowing. If
we knew that velocity field, we could take any control cell and transport it to a
perturbed-looking cell by following the flow — solving the ODE `dz/dt = v(z, t)` from `t=0`
to `t=1`.

**Training it.** Pick a control cell `z0` and a perturbed cell `z1` at random,
*independently* (this is "independent coupling"). Draw `t ~ Uniform(0,1)` and put a point on
the straight line between them:

```
z_t = (1 - t) z0 + t z1
```

Along that straight line the velocity is constant and equal to `z1 - z0`. So train a network
to predict `z1 - z0` from `z_t` and `t`:

```
loss = E || v_theta(z_t, t | S) - (z1 - z0) ||^2
```

The remarkable fact is that regressing on these random straight-line pairs recovers the
correct *marginal* velocity field, because the network, being forced to output one value per
`(z_t, t)`, learns the conditional expectation `E[z1 - z0 | z_t, t]` — and that is exactly
the field that transports `p0` to `p1`.

**The mistake I made, which is worth understanding properly.** The project brief writes the
field as `v(z_t, t | z0, S)` — conditioned on the starting cell. I implemented that
literally and it was wrong. Here is why. Under independent coupling, `z1` is drawn
independently of `z0`, so for a *fixed* `z0` the distribution of `z1` is the entire perturbed
population. The network therefore learns `E[z1 - z0 | z_t, z0, S]`, which points at the
*mean* of the perturbed population. Integrating a deterministic ODE from a single starting
point then runs that point to the mean. A deterministic ODE cannot turn one point into a
spread-out distribution. Measured result: the predicted population's per-dimension spread
collapsed to 0.30 against a true value of 0.86, and the energy distance to the truth became
2.91 — *worse than predicting no change at all* (0.87). Removing `z0` from the conditioning
fixed it (spread 0.93, energy distance 0.70). Per-cell variety comes from the ODE's
*starting point*, not from conditioning.

**How `S` gets in.** The perturbation set conditions the field. Sets are unordered and vary
in size, so we use **DeepSets**: embed each perturbed gene, sum the embeddings (summing is
permutation-invariant and works for any set size), then pass through a readout network. This
is what makes an unseen *combination* representable — if the model has seen gene A and gene
B separately, the sum of their embeddings is a well-defined input even if the pair AB never
occurred.

---

## 3. Deep ensembles (where `U` comes from)

Train `M = 5` copies of the same model from different random initialisations. Where they
agree, the data constrained the answer. Where they disagree, it did not. The spread between
members is an estimate of *epistemic* uncertainty — uncertainty from lack of knowledge,
which more data would reduce — as distinct from *aleatoric* uncertainty, the irreducible
noise in the measurement.

Epistemic is the one we want here, because the whole risk is recommending an intervention
the model has no basis for. We use the root-mean-square disagreement between members on the
predicted mean shift.

An ensemble spread is only a *relative* score: "this set is less certain than that one". It
is not an error bar, and nothing guarantees the true value lies within one spread. Turning
it into a guarantee is what conformal prediction is for.

---

## 4. Conformal prediction (turning `U` into a guarantee)

**The promise.** Pick a miscoverage rate `alpha = 0.1`. Split conformal produces intervals
that contain the truth at least 90% of the time — with *no* assumption that the model is
correct, that errors are Gaussian, or anything else about the model. The only assumption is
**exchangeability**.

**The recipe.**
1. Hold out a calibration set the model never trained on.
2. On it, compute a nonconformity score per point. We use the **normalised** score
   `score = |y - yhat| / spread`. Dividing by the ensemble spread is what makes intervals
   *adaptive*: a point the ensemble disagrees about gets a wider interval.
3. Sort the scores and take the `ceil((n+1)(1-alpha))`-th smallest, call it `q`.
4. For a new point, the interval is `yhat ± q * spread`.

**Why that index.** If a new point's score is exchangeable with the `n` calibration scores,
then its rank among all `n+1` scores is uniform. The probability that it falls above the
`ceil((n+1)(1-alpha))`-th is at most `alpha`. That is the entire proof, and it is why the
`+1` matters. Note the consequence: if `n < 1/alpha - 1` there is no valid finite index at
all, and the honest output is an *infinite* interval. Our implementation returns exactly
that rather than a narrow interval that would be silently invalid.

**Where it breaks, in this project, by construction.** Exchangeability fails when
calibration and test points are not drawn from the same distribution. Our unseen-cell-context
split deliberately tests on a context that does not appear in training. The guarantee does
not apply there, so coverage on that split is reported **separately** and is not counted
toward H2. Reporting it pooled with the rest would hide the one place the method is known to
fail.

We group calibration by *perturbation*: a perturbation is wholly in or wholly out. Otherwise
cells from the same perturbation would appear on both sides and the scores would be
correlated through a shared perturbation, inflating apparent coverage.

---

## 5. The optimiser's curse (why `lambda > 0`)

This is the conceptual heart of the project and the thing most worth being able to explain.

Suppose you have 1000 candidate interventions. Each has a true reversal value, and your model
estimates it with some error. You pick the one with the highest *estimate*. That estimate is
high for two possible reasons: the intervention is genuinely good, or the model made an
optimistic error. From inside the model these look identical.

Because you selected on the estimate, you have preferentially selected the optimistic
errors. The winner's estimate is therefore **biased upward**: on average, the thing you pick
performs worse than its estimate predicted. This is not a bug in the model — it happens even
with unbiased, independent noise on every candidate. It is a property of taking a maximum
over noisy estimates. The more candidates you search, the worse it gets.

The fix is to stop maximising the raw estimate and maximise a *penalised* one:

```
J(S) = E[R(z0, S)] - lambda * U(S)
```

You give up some predicted reversal in exchange for picking from the region where the model
is confident — and so where the optimistic-error term is small. H3 is the claim that this
trade is worth making, and it is settled by **verified** reversal from the simulator oracle,
never by the model's own prediction, since that is precisely the quantity being gamed.

We also measure the **optimism gap** = predicted minus verified reversal of the selected set.
The curse predicts it is positive and larger at `lambda = 0`.

---

## 6. Evaluation design (the part reviewers attack)

**Deltas, not raw expression.** All predictive metrics are computed on the *change* from the
matching control, not on raw expression. Raw-expression correlation is dominated by the
average expression profile, which every model gets right — including "predict the control
mean". It flatters everything and compresses the differences between models.

**The shuffled-label control.** We fit the same model with perturbation labels permuted.
Whatever it scores is achievable *without knowing which perturbation was applied*. On an
early version of the simulator it reached Pearson-delta **0.57** on the combination split.
That is the real zero point, and it is why the control appears in every table. If you report
0.65 and the shuffled control gets 0.57, you have reported almost nothing.

**Three split families**, because "generalisation" means different things:
- *unseen single* — a gene held out entirely. Crucially, a held-out gene may not appear
  inside a training *pair* either, or the model has already seen its effect.
- *unseen combination* — two sub-cases: both genes seen individually (does the model
  compose?) and neither gene seen (much harder).
- *unseen cell context* — same perturbations, new cellular background.

**The oracle as a diagnostic, not just a bound.** `oracle_latent` reproduces the observed
post-perturbation population. Under a sound sensitivity readout it should rank interventions
at least as well as any model. If a real model beats it, the *readout* is failing, not the
model. This check caught a genuine bug in my own oracle (see Q13).

---

## 7. Twenty questions a reviewer will ask

**Q1. Your results are entirely synthetic. Why should I care?**
You should care about the *methodological* findings and not at all about the biology. The
synthetic setting is the only one where the true reversal value of every intervention set is
exactly knowable, which is what makes H2-H4 testable — you cannot measure whether abstention
reduces error without knowing the error. The biological claims (H5) are not supported by
anything here and are not made. The honest framing is: this is a methods paper with a
simulator study, and real-data validation is future work that has not been done.

**Q2. Your simulator was built by you, for this project. Isn't that circular?**
Partly, and that is the main threat to validity. I mitigated it in three ways. First, the
parameters were calibrated for *task difficulty* before any model existed — selected so that
most interventions fail, a few succeed, and random sets reverse ~1% of the time. Second,
`check_bistable` and `check_reversible` make a degenerate network raise an error instead of
silently producing an empty experiment. Third, `LOG.md` documents every occasion where a
result turned out to be an artefact. I would not claim the simulator is realistic; I claim it
is a controlled testbed with known answers.

**Q3. What is the single biggest weakness?**
The sensitivity function. It is a learned proxy, trained on unperturbed cells, then applied
to predicted post-perturbation states — which is extrapolation. On a *known true* reversal
(true sensitivity 0.10 to 0.98), the two readouts reported 0.21 to 0.95 and 0.11 to 0.23
respectively. The headline reversal number depends on which proxy you pick. That is why both
are always reported and never averaged.

**Q4. Then why not just use the true sensitivity from the simulator?**
Because then the result would be meaningless for real data, where no such function exists.
The proxy's unreliability is a feature of the real problem, and modelling it is the point.
The true function is used only to *verify* search recommendations, which is legitimate: it
is the held-out ground truth, not an input to the method.

**Q5. Why does `s` have to be trained on a different experiment?**
Otherwise it partly reads back the response model's own training signal and a high reversal
score becomes self-confirming. The price is a domain shift between the two experiments. This
is a real trade-off, not an oversight, and the shift is the thing Q3 describes.

**Q6. H1 says deep models don't beat linear baselines. Isn't that just because your deep
model is bad?**
It is a fair challenge and partly true in the smoke profile, where PertFlow is deliberately
undertrained. The relevant evidence is: (a) the comparison is paired on identical test
points with CIs; (b) the `additive` baseline beating the learned models on combinations is
consistent with the simulator's combination effects being near-additive away from the
switch boundary, which is a property of the data rather than of the model; (c) a published
benchmark reports the same pattern (`TODO[verify]` — I have not verified that citation and
must not cite it until I have). The claim is "does not *reliably* beat", not "is worse".

**Q7. You found your own model had a flaw in the flow formulation. Doesn't that invalidate
the comparison?**
It invalidated the earlier numbers, which is why they are superseded in `LOG.md` rather than
reported. Two bugs mattered: the `z0`-conditioning that collapsed the predicted population,
and an 18x scale mismatch in the gene embeddings that caused the network to ignore
perturbation identity entirely and emit one average shift for every perturbation. The second
also handicapped the ridge baselines, because ridge at fixed `alpha` is not scale-invariant.
Both are fixed and both now have regression tests.

**Q8. Your model's predicted effect magnitudes are only ~60% of the truth. Is that not a
fatal problem?**
It is a real limitation, characterised rather than hidden. More capacity and more training
improve the predicted *direction* on held-out perturbations (cosine 0.78 to 0.90) while the
magnitude stays at ~0.6x and the training-set ratio actually falls. That signature is
shrinkage toward the conditional mean, which is what a squared-error-optimal estimator should
do when genuinely uncertain about an unseen combination. Consequences: reversal estimates are
biased toward zero (so conservative), ranking is largely preserved (which is what the search
consumes), and conformal coverage is unaffected in principle because systematic bias inflates
calibration residuals and therefore interval width.

**Q9. How do you know your splits don't leak?**
Tests, not care. HVG selection, scaling, PCA and the gene embeddings are fitted through a
single `training_cell_mask`, so the decision cannot drift between call sites. A held-out gene
is forbidden from appearing inside a training perturbation of *any* size. And one test
deliberately builds a leaked split and asserts the checker rejects it — without that, the
other leakage tests could all pass vacuously.

**Q10. Your first leakage checker was itself wrong. How?**
It demanded disjoint train/test perturbations for *every* split, which the unseen-context
split violates by design: there the held-out axis is the cell context and the same
perturbations are deliberately re-probed in a new context. The checker is now axis-aware.
Worth stating because it shows a leakage check can be both strict and wrong.

**Q11. Why does conformal coverage hold if your model is biased?**
Because conformal makes no assumption that the model is correct. A systematic bias makes the
calibration residuals larger, which makes `q` larger, which makes the intervals wider. The
coverage guarantee is preserved; what degrades is *efficiency* (wider intervals, so less
useful). The guarantee needs exchangeability, not accuracy.

**Q12. So conformal always works?**
No, and we show where it doesn't. Exchangeability fails under cell-context shift, so coverage
there is reported separately and excluded from the H2 verdict. We also return an *infinite*
interval when the calibration set is too small to certify the requested level, rather than a
narrow invalid one.

**Q13. You flagged your own sensitivity readouts as unreliable and then retracted it. What
happened?**
A diagnostic compared each model's intervention ranking to the latent oracle's. The oracle
ranked 6th of 7 under *both* readouts, which I first read as both readouts being unfit. The
universality was the clue: the fault was my oracle. It had no `sample_latent`, so its
"predicted population" was the control cells shifted by the true *mean* delta — and an
additive shift cannot represent a population moving into a different attractor, which is
exactly what reversal is. The oracle now returns the observed post-perturbation cells. The
lesson: when a diagnostic condemns everything, suspect the diagnostic.

**Q14. Why Spearman correlation and not R-squared for intervention ranking?**
Because the decision is a ranking decision — we pick the top set, so only order matters — and
because predicted and true reversal are on different scales (learned proxy versus the
simulator's exact function). Correlating them as magnitudes would mostly measure the scale
mismatch.

**Q15. Some of your splits have three test perturbations. What can you conclude from that?**
Nothing, and the code enforces it: rank correlations require at least five test
perturbations, and splits below that are excluded from verdicts with the exclusion recorded.
Gene-holdout split sizes are bounded by the number of perturbable genes, so this is a
structural limit, not a sampling choice.

**Q16. Why energy distance rather than just comparing means?**
Because a perturbation can change a population's *shape* without moving its mean much — for
instance splitting it between two attractors. Energy distance is zero if and only if the two
distributions match, so it catches that. It also caught the population-collapse bug in Q7:
the point metrics looked fine while the energy distance was worse than predicting nothing.

**Q17. Isn't greedy search too weak for combinatorial intervention design?**
Possibly, which is why exhaustive search is included as a reference wherever the candidate
set is small enough. That separates two failure modes: *optimisation regret* (a better search
would have found more) from *model error* (no search could, because the objective is wrong).
Reporting only greedy's result would confound them.

**Q18. How do you choose `lambda`?**
We do not tune it to maximise verified reversal — that would be selecting on the test
quantity. We report a sweep over `lambda` and show the trade-off curve. The claim under test
is directional (does `lambda > 0` beat `lambda = 0`?), not that a particular value is
optimal.

**Q19. What would change your mind about this whole approach?**
Several things, each checkable. If uncertainty-penalised search gave no improvement in
verified reversal across seeds with CIs overlapping zero, H3 is dead. If conformal coverage
missed nominal on exchangeable splits, the uncertainty machinery is not working. If a
reversal-aware evaluation ranked models identically to Pearson-delta, the evaluation
contribution is empty. And if the two sensitivity readouts disagree as much on real data as
on synthetic, the reversal score may not be measurable well enough for the method to mean
anything — that is the one I consider most likely.

**Q20. What is the actual contribution, stated without overclaiming?**
A pipeline, and an evaluation design, for uncertainty-aware counterfactual intervention
search over cell states, tested where ground truth is exactly knowable. Concretely: a
reversal-aware evaluation that can reorder models relative to the standard metric; a
demonstration that conformal calibration on held-out perturbations reaches nominal coverage
while failing under context shift, as theory predicts; and a measurement of whether
penalising uncertainty mitigates the optimiser's curse in intervention selection. Plus a
documented account of several ways this kind of pipeline produces convincing artefacts — which
may be the most transferable part.

---

## 8. Equations worth memorising

| | |
|---|---|
| flow-matching objective | `min_theta E_{t, z0, z1} \|\| v_theta(z_t, t \| S) - (z1 - z0) \|\|^2`, `z_t = (1-t)z0 + t z1` |
| reversal score | `R(z0, S) = E_{z' ~ f(z0,S)}[s(z')] - s(z0)` |
| penalised objective | `J(S) = E[R(z0, S)] - lambda * U(S)`, `\|S\| <= k` |
| ensemble uncertainty | `U(S) = sqrt( mean_d Var_m[ delta_m(S)_d ] )` over `M` members |
| conformal score | `score_i = \|y_i - yhat_i\| / spread_i` |
| conformal quantile | `q = ` the `ceil((n+1)(1-alpha))`-th smallest calibration score |
| conformal interval | `[yhat - q * spread, yhat + q * spread]` |
| energy distance | `D(X,Y) = 2 E\|X-Y\| - E\|X-X'\| - E\|Y-Y'\|` |
