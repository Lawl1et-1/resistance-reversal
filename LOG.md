# LOG

Decisions, failures and open questions, newest milestone last.

## M0 - setup and environment audit

- Environment: Windows 11, Python 3.13.1, torch 2.11.0+cu126, **no CUDA device present**
  (`nvidia-smi` not found). All profiles must therefore be CPU-feasible; `full` is sized for
  CPU rather than assuming a GPU.
- Dependencies already present and pinned in `requirements.txt`: numpy 2.2.4, scipy 1.15.2,
  scikit-learn 1.6.1, pandas 2.2.3, matplotlib 3.10.1, torch 2.11.0, PyYAML 6.0.2,
  pytest 9.1.1.
- No real data downloaded yet, so `DATA.md` is not yet meaningful. Tier 1/2 acquisition is
  M6 work and is tiered *should*; the must-tier runs entirely on the synthetic simulator.

## M1 - synthetic simulator

**Decision: bistable toggle switch rather than a linear response model.** The project's claim
is about moving a cell between discrete transcriptional states, so the simulator needs genuine
attractors. A linear model would make "reversal" a trivially continuous quantity and H2-H4
would not be testing anything.

**Failure 1: the first network was not bistable at all.** Both master regulators saturated
high (0.973 / 0.990) and true sensitivity was 0.49 for *both* phenotypes. Cause: regulator
effects were all non-negative, so ~20 regulators at resting expression contributed a large
positive drive to both masters, swamping the mutual inhibition. Fixes:
  - regulator effects made antagonistic (a resistance driver activates the resistance master
    *and* represses the sensitive one);
  - master bias centred analytically so that net drive is ~0 at the symmetric point given the
    regulators' resting levels.

**Failure 2: after fixing bistability, nothing reversed.** Best reversal over all 1350 sets of
size <= 3 was 0.038. The hysteresis barrier (`w_self = 4`) was far larger than what any three
regulator knockdowns could shift. Resolved by sweeping `(w_self, w_regulator)`; see below.

**Failure 3: unlucky seeds produced vacuous networks.** At `w_self = 3.5` seed 3 had *no*
reversible pair (best pair 0.10) while seed 0 had 59 of them. Per-seed experiments would have
been incomparable and some seeds would have measured nothing. Cause: resistance-driver
magnitudes drawn from `uniform(0.45, 1.0)` meant the two strongest drivers could both land
weak. Fixed by drawing mode-0 (resistance driver) magnitudes from `uniform(0.85, 1.0)`, and by
adding `check_bistable` / `check_reversible` so a degenerate instance raises instead of
silently yielding an empty experiment.

**Calibration.** Two parameter sweeps over `(w_self, w_regulator)`, first at 60 genes then at
the full 200, selecting for: separated attractors, a reversible set within k on *every* seed,
a near-zero median set (random sets must fail), and singles insufficient often enough that
combinations matter. Chosen: `w_self = w_cross = 3.5`, `w_regulator = 2.5`.

Resulting regime, seeds 0-4, both profiles:

| | s(sensitive) | s(resistant) | singles reversing | best pair | median pair |
|---|---|---|---|---|---|
| full (200 genes, 20 regulators) | 0.95-0.97 | 0.02-0.06 | 0-6 of 20 | 0.93-0.96 | 0.01-0.16 |
| smoke (60 genes, 12 regulators) | 0.95-0.97 | 0.03-0.05 | 0-2 of 12 | 0.95-0.96 | 0.01-0.02 |

The spread in "singles reversing" across seeds is kept deliberately: it is realistic
heterogeneity, and averaging over seeds is what the statistics are for.

**Tests:** 29 passing in 5.8 s, covering bistability and attractor stability across 5 seeds,
reversibility within k, the median set *not* reversing, resistance drivers beating bystanders
by the layout's causal truth, knockdowns actually suppressing their target, reversal moving
both programs in opposite directions, determinism, seed-dependence of the network, canonical
and duplicate-free set enumeration, the observation layer's dropout rate, and config validation.

**Open questions**
- Whether the chosen barrier makes the *learning* problem well-scaled is not yet known; it was
  calibrated for the search problem. If baselines saturate in M2, revisit.
- Process noise is currently applied only when an rng is passed, which makes the oracle
  noise-free by construction. That is intended (the oracle is the simulator's exact answer)
  but means the oracle is slightly optimistic relative to any finite-sample estimate.

## M2 - splits, metrics, baselines

**Failure 4: the first leakage check was itself wrong.** `check_no_leakage` demanded that
train and test perturbations be disjoint for *every* split, which fails the unseen-context
split by construction -- there the held-out axis is the cell context and the same
perturbations are deliberately probed in a new context. The check is now axis-aware
(`meta["held_out_axis"]`): perturbation-axis splits require mutual disjointness, the
context-axis split requires disjoint contexts, and train/cal disjointness is required
either way because conformal residuals must come from unfitted perturbations. A test
asserts that a deliberately leaked split actually raises, so the other leakage tests are
not vacuous.

**Failure 5 (the significant one): the unseen-single split was measuring nothing.**
Every model scored ~0.00 Pearson-delta on held-out genes. Before writing that up as a
negative result, I checked whether the split was *solvable in principle*, and it was not:
- correlation between gene-embedding distance and true-effect distance: **-0.11**
- mean |correlation| among regulators in control cells: **0.12** (i.e. noise)

The regulators are upstream and mutually independent, so co-expression carried no
information about what a gene does and no model could have transferred to a held-out gene.
A "deep models fail to generalise to unseen genes" claim from that setup would have been an
artefact of my own simulator, not a finding.

Fix: regulators are now grouped into pathway-like **modules**. Members of a module share an
effect type and magnitude (with per-gene spread, so they are not interchangeable) and
co-express through a shared per-cell latent factor (`sample_cell_bias`). This required
`simulate` to accept a persistent per-cell drive, since initial-state jitter decays during
relaxation. After the fix:
- embedding distance vs effect distance: **0.93**
- mean |correlation| within a module **0.79**, across modules **0.06**
- bistability and reversibility unchanged (0.96 / 0.04, best pair 0.95)

The information needed to transfer now exists, so whether a model exploits it is an
empirical question rather than a foregone conclusion. This is the same class of mistake as
Failure 3: a degenerate setup that silently produces a publishable-looking number.

**First baseline results** (synthetic, 60 genes / 12 regulators / 4 modules, 1 seed, all
singles and pairs, 95% bootstrap CIs over perturbations; `oracle_latent` is the
latent-reconstruction ceiling and `shuffled_labels` the negative control):

| split | additive | ridge_embedding | low_rank_linear | shuffled | oracle |
|---|---|---|---|---|---|
| unseen_single | n/a | 0.30 [0.12, 0.47] | 0.30 | -0.03 | 0.99 |
| unseen_combo_seen_genes | **0.84** [0.77, 0.90] | 0.67 | 0.67 | 0.24 | 1.00 |
| unseen_combo_unseen_genes | n/a | 0.16 [-0.02, 0.35] | 0.16 | -0.09 | 1.00 |
| unseen_context | 0.74 [0.66, 0.80] | 0.73 | 0.72 | 0.21 | 1.00 |

Early observations, all provisional (one seed, one network):
- `additive` beats both learned linear models on unseen combinations. Relevant to H1.
- Transfer to a held-out *gene* is weak but real (0.30, CI excludes 0) and transfer to an
  unseen *pair of unseen genes* is not yet distinguishable from zero.
- `shuffled_labels` reaches 0.24 on the combination split: a fifth of the Pearson-delta
  signal there does not require knowing which perturbation was applied. Under the earlier
  module-free simulator it reached **0.57**. This is direct support for the H1 framing that
  Pearson-delta flatters models, and it is why the control is reported in every table.
- `oracle_latent` is ~0.99 everywhere, so the 32-d latent bottleneck is not the binding
  constraint and does not explain any model's failure.
- `control_mean` yields an undefined Pearson-delta (a constant prediction has no variance);
  it is reported as n/a with the MSE columns ranking it, not silently dropped.
- `ridge_embedding` and `low_rank_linear` agree to ~3 decimals, so the rank-8 truncation is
  not binding at this feature count. Needs either a lower rank or an honest note that the
  two coincide here.

**Open questions**
- Gene-holdout splits are small (3 test genes at 12 regulators). The full profile's 20
  regulators and multiple seeds should fix the perturbation-level sample size.
- The low-rank baseline is described as "in the spirit of" the 2025 benchmark paper's
  model, not a reproduction, and the citation is `TODO[verify]`.

## M3 - PertFlow and deep ensemble

**Failure 6: conditioning the flow on `z0` destroyed the predicted distribution.** The
project brief writes the velocity field as `v_theta(z_t, t | z0, S)`, and I implemented that
literally. It is wrong under independent coupling. For a given `z0` the training target is
the *whole* perturbed marginal, so the network can only learn `E[z1 - z0 | z_t, z0, S]`, and
integrating a deterministic ODE from a single starting point runs it to that conditional
mean. A point cannot spread into a distribution. Measured consequence:

| | per-dimension std | energy distance to truth |
|---|---|---|
| control cells (no prediction) | 0.854 | 0.871 |
| observed perturbed cells | 0.858 | - |
| PertFlow, z0-conditioned | **0.300** | **2.914** (worse than predicting nothing) |
| PertFlow, marginal field | 0.930 | 0.703 |

Standard conditional flow matching uses the marginal field `v(z_t, t | S, context)`, which
provably transports the control marginal to the perturbed marginal; per-cell structure is
carried by the ODE's starting point, not by conditioning. Implemented that way, the
distribution is recovered and the energy distance finally beats the do-nothing control. The
notation in the brief describes the conditional *path construction*, not the network's
inputs, and the paper should say so explicitly since it is an easy mistake to repeat.

**Failure 7 (affected the baselines too): the gene embeddings were ~18x too small.**
After fixing the flow, predicted mean shifts were still only ~40% of the true magnitude --
and crucially they undershot on *training* perturbations too, and 10x more training did not
help, so it was neither underfitting nor a generalisation gap. Isolating the model on a
single perturbation gave near-perfect transport (predicted shift norm 3.33 vs true 3.16,
cosine 0.997), which located the fault in the conditioning pathway rather than the flow.

The cause: `gene_embeddings` normalised by a global maximum, leaving per-dimension spread at
**0.0485** while the latent state concatenated alongside it has spread **0.854**. The
network could minimise its loss by ignoring perturbation identity and emitting one average
shift for every perturbation. Evidence: spread of predicted shifts across perturbations was
**0.066** against a true value of 0.306, with predicted norms sitting in a narrow 0.65-1.11
band while true norms ranged 1.7-3.8.

The same bug silently handicapped `ridge_embedding` and `low_rank_linear`, because ridge at
a fixed `alpha` is not scale-invariant and near-zero features get shrunk almost to nothing.
Both baselines' numbers in the M2 table above are therefore pessimistic and are superseded.

Fix: standardise the embedding per dimension. After it, embedding spread is 1.015,
cross-perturbation spread of predictions rises to 0.214 (true 0.306) and the mean cosine to
the true shift is 0.842 -- the direction is now broadly right.

**Failure 8: scikit-learn plus torch segfaults under pytest on this machine.** Adding the
PertFlow tests crashed the interpreter (exit 139) on even the lightest test, including one
that only touched the set encoder. Isolating it: `torch` with `sklearn.decomposition` or
`sklearn.preprocessing` segfaults *under pytest* while both import fine in plain Python, and
`scipy.stats`, `scipy.spatial.distance` and `pandas` are all unaffected. It is the duplicate
OpenMP runtime interacting with pytest's assertion-rewriting import hook;
`KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` made no difference.

Left alone this would have meant no test coverage at all for any model that touches both
libraries, i.e. the entire proposed method. scikit-learn was used for exactly two things,
`PCA` and `StandardScaler`, both a few lines of numpy, and the encoder's delta helpers
already depend on the transform being exactly linear. Reimplemented on `np.linalg.svd`:
the dependency is dropped from `requirements.txt`, the suite runs clean (94 tests), and the
encoder's identities are now pinned by tests
(`encode_delta(x1 - x2) == transform(x1) - transform(x2)` and its decode counterpart, plus
a test asserting that passing a delta through `inverse_transform` is wrong, which is the
trap those helpers exist to prevent).

**Still open: magnitude shrinkage.** Predicted shift norms remain ~55-60% of the truth
(measured 0.63 train / 0.58 test, cosine 0.83 / 0.74). Some shrinkage is the correct
behaviour of a conditional-mean estimator under uncertainty, but this much will bias every
reversal score downward, so a capacity/training sweep is running before any of H2-H3 is
evaluated. A model that systematically understates the effect of every intervention would
make the uncertainty-penalised search look better or worse than it deserves for reasons
unrelated to uncertainty.

## M4 - conformal calibration, sensitivity functional

Split conformal with the normalised score `|y - yhat| / spread` is implemented with the
standard finite-sample quantile (the `ceil((n+1)(1-alpha))`-th smallest calibration score)
and returns an **infinite** radius when the calibration set is too small to certify the
requested level, rather than a silently invalid interval. 20 tests, including the coverage
guarantee verified on constructed exchangeable data at alpha in {0.05, 0.1, 0.2} over 300
trials, heteroscedastic coverage, interval adaptivity, and selective risk falling only when
uncertainty is actually informative.

The exchangeability assumption is stated in the module and will be **violated by design** on
the unseen-context split, where coverage is reported separately and expected to miss
nominal.

**Failure 9: the per-cell module drive was flipping cells between attractors.** The
sensitivity classifier reached only 0.666 held-out accuracy, which looked like a modelling
failure. It was not: at `module_coexpr = 0.9`, 33.7% of cells relaxed to the attractor
*opposite* their assigned phenotype label, so 0.666 was essentially the Bayes limit
(1 - 0.337 = 0.663). The labels were a third noise.

This is a direct tension between two requirements, measured across the parameter:

| `module_coexpr` | label noise | within-module \|r\| | across-module \|r\| |
|---|---|---|---|
| 0.0 | 0.000 | 0.04 | 0.04 |
| 0.2 | 0.027 | 0.39 | 0.04 |
| **0.3** | **0.096** | **0.53** | **0.04** |
| 0.6 | 0.255 | 0.74 | 0.05 |
| 0.9 | 0.333 | 0.80 | 0.05 |

Co-expression has to be strong enough for held-out-gene transfer (Failure 5) but a large
per-cell drive on the regulators pushes cells across the separatrix. Set to 0.3: pathway
signal unambiguous (13x within/across ratio), label noise 10%. Sensitivity accuracy then
rose to 0.901 (single-cell) and 0.890 (signature), which is at the ~0.90 ceiling the
residual label noise allows. The remaining noise is realistic -- persister/sensitive labels
in real data are not clean -- and makes `s` an explicitly imperfect proxy, which is the
caveat the paper has to carry anyway.

**Failure 10: SVD truncation of the gene embeddings destroyed the signal it existed to
carry.** Correlation between embedding distance and true effect distance (positive means
similar profile implies similar effect):

| representation | 60 genes | 200 genes |
|---|---|---|
| full co-expression profile | **+0.67** | +0.52 |
| top-8 SVD | -0.30 | +0.23 |
| top-16 SVD (the previous default) | **-0.39** | +0.23 |
| top-32 SVD | +0.25 | +0.22 |

The leading components of the gene-gene correlation matrix track global program structure;
module membership lives further down, so truncating to 16 dimensions discarded exactly the
relevant variation. At 60 genes the truncated embedding points the *wrong way*, meaning a
model using it would transfer away from functionally similar genes. Default changed to the
full standardised co-expression profile, with the measurements recorded at the function so
the hyperparameter is not "tuned" back later.

**A real and reportable disagreement between the two sensitivity readouts.** On a true
oracle reversal (true sensitivity 0.101 -> 0.980), the two readouts disagree sharply:

| readout | control | after oracle reversal |
|---|---|---|
| true sensitivity (simulator) | 0.101 | 0.980 |
| signature / "bulk" | 0.205 | 0.950 |
| single-cell latent classifier | 0.107 | **0.225** |

The low-capacity signature readout tracks the reversal; the 32-dimensional latent classifier
barely moves. Both were fitted on *unperturbed* cells, and a knockdown drives its target gene
far outside that training range, so the latent classifier is extrapolating while the
signature (an average over many program genes) is not. This is precisely the domain shift the
brief says to expect, and it is the single most important caveat in the project: the headline
reversal number depends on which proxy is used. Both are reported everywhere, never averaged
into one number. Whether regularising the latent classifier narrows the gap is still to be
checked -- an under-regularised classifier would be a straw man.
