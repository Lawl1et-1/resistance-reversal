# PLAN

## Question

Can we learn a model of how perturbations move a cell's transcriptional state, search it
for small interventions that push a drug-resistant state toward a drug-sensitive one, and
know when the model should not be trusted?

Everything here is in silico. There is no wet-lab validation, so the object of study is a
*predicted shift toward a drug-sensitive transcriptional state*, never a treatment effect.

## Scope

Priority follows the milestone tiers. A finished must-tier beats a half-finished everything.

| Tier | Milestones | Content |
|---|---|---|
| must | M0-M5 | simulator, splits/metrics/baselines, PertFlow + ensemble, conformal abstention, uncertainty-penalised search |
| should | M6-M7 | real sensitive/resistant data and the reversal readout, active learning |
| could | M8 | mechanistic anchoring (pathway enrichment, DepMap/PRISM) |

Tier 0 (synthetic) carries the must-tier. It is the only setting with exactly knowable
ground truth -- the true reversal score of every perturbation set of size <= k -- which is
what makes H2-H4 testable at all. Real data (Tier 1-2) is where H1 and H5 live, and where
the honest answer may be "we cannot tell".

## Formal setup

- Latent state `z` (dim 32-64) from an encoder fitted on training data only.
- Perturbation set `S`, `|S| <= k = 3`.
- Response model `f_theta(z0, S)`: predictive distribution over the post-perturbation `z'`.
- Sensitivity functional `s(z)` in [0,1], trained on sensitive-vs-resistant data from
  experiments disjoint from those used to fit `f`.
- Reversal score `R(z0, S) = E[s(z')] - s(z0)`, plus energy distance from `z'` to the
  sensitive-state distribution.
- Objective `S* = argmax E[R] - lambda * U(S)`, abstaining when `U` exceeds a calibrated
  threshold.

## Hypotheses and what would falsify each

- **H1** Deep perturbation models do not reliably beat strong linear baselines under
  unseen-perturbation splits, and reversal-aware evaluation reorders the models relative to
  Pearson-delta. *Falsified if* a deep model wins on both evaluations with non-overlapping CIs.
- **H2** Conditional latent flow + deep ensemble + split conformal reaches target coverage on
  unseen perturbations, and abstention lowers recommendation error. *Falsified if* empirical
  coverage misses the target band, or risk does not fall as coverage drops.
- **H3** Uncertainty-penalised search recommends interventions with higher held-out-verified
  reversal than uncertainty-blind search (the optimiser's curse). *Falsified if* `lambda = 0`
  matches or beats `lambda > 0` on verified reversal.
- **H4** Active selection reaches a given accuracy with fewer perturbations than random.
  *Falsified if* learning curves overlap within CIs.
- **H5** Top-ranked targets are enriched for known resistance biology versus matched random
  controls. *Falsified if* enrichment is indistinguishable from the matched control.

Each gets a verdict with CIs, including a negative one. A rigorous negative result is a
result; H1 in particular is *expected* to come out partly negative.

## Risks

1. **Domain shift between `f` and `s` is the central validity threat.** The perturbation data
   and the resistance data come from different experiments, so `s` is a proxy evaluated
   off-distribution. Mitigation: two independent sensitivity readouts (single-cell classifier
   and bulk pharmacogenomic predictor) and report their disagreement rather than hiding it.
   This caveat belongs in the abstract, not just the limitations section.
2. **A synthetic win proves only that the pipeline works.** Tier 0 results are tagged
   `synthetic` everywhere and are never presented as biological evidence.
3. **Leakage via preprocessing.** HVG selection, scaling, PCA and gene embeddings must be fit
   on train only; conformal calibration must be grouped by perturbation. Enforced by tests,
   not by discipline.
4. **Real-data availability.** Downloads may fail or licences may forbid redistribution. Record
   in `DATA.md`, use the documented fallback, continue. Never ship raw data in the repo.
5. **Compute.** No GPU on the development machine; the full profile must stay CPU-feasible or
   degrade gracefully. Anything over ~10 min runs as a background job with a log.
6. **Conformal exchangeability is assumed and is wrong under cell-context shift.** Report
   coverage under shift separately and say so.

## Simulator design (M1, done)

A toggle switch between a sensitive master regulator and a resistance master regulator, each
self-activating and mutually inhibiting; upstream regulators feed signed drive into the switch;
downstream programs read it out. Bistability is the point: a perturbation that shifts the
basin flips the whole program.

Parameters were calibrated so that the task sits in the interesting regime rather than being
trivially easy or vacuous (`w_self = 3.5`, `w_regulator = 2.5`):

- attractors well separated on every seed (true sensitivity ~0.96 vs ~0.04);
- the best set of size <= 2 reverses on every seed (>= 0.93), so no seed is vacuous;
- the *median* pair barely moves the state (~0.01-0.16), so random sets fail and reversal is
  not handed out for free;
- 0-6 of 20 single knockdowns suffice depending on seed, so combinations genuinely matter.

Resistance-driver magnitudes are drawn near full strength by construction, because with a wide
range unlucky seeds produced networks where nothing reversed at all. `check_bistable` and
`check_reversible` make a degenerate instance fail loudly instead of silently producing an
empty experiment.

## Next

M2: splits (unseen singles, unseen combinations, unseen cell context), metrics, baselines.
