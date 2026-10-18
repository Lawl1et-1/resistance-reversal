# Model card — PertFlow and the reversal search

## Overview

A conditional flow-matching model of how perturbations move a cell's latent transcriptional
state, wrapped in a calibrated search for small intervention sets predicted to shift a
drug-resistant state toward a drug-sensitive one.

- **Version**: pre-publication research code. No release has been tagged.
- **Provenance of every current result**: `synthetic`. See `DATA.md`.
- **Task**: given a control cell population and a candidate set of at most `k = 3` gene
  knockdowns, predict the post-perturbation population, score its predicted shift toward a
  sensitive state, and either recommend a set or abstain.

## Intended use

Research use only, specifically:

1. **Methods research.** Studying how perturbation-response models behave under splits that
   hold out genes, combinations or cell contexts; how deep-ensemble uncertainty plus split
   conformal behaves on such splits; and whether penalising uncertainty mitigates the
   optimiser's curse in intervention search.
2. **Evaluation design.** Demonstrating that a reversal-aware evaluation can rank models
   differently from the conventional Pearson-delta metric, and that a shuffled-label control
   reaches a non-trivial score on the conventional metric.
3. **Hypothesis generation, on real data, as an input to experimental design** — and only
   once Tier 1/2 data is actually in use, which it is not.

## Out-of-scope and prohibited use

**This model must not be used to make or support any clinical, treatment or patient-care
decision.** It is not a medical device, has no regulatory clearance, and has never been
tested in a living system.

Specifically out of scope:

- Any claim that a recommended intervention reverses resistance, treats, or cures anything.
  The output is a *predicted shift toward a drug-sensitive transcriptional state* in a
  model — a prediction about a transcriptional readout, not about disease.
- Ranking therapeutic targets for a real programme without independent experimental
  validation. The model's top-ranked sets have never been tested in cells.
- Any use on a patient sample or in a clinical workflow.
- Transfer to a biological context unlike the training data. Generalisation to an unseen
  cell context is a measured weakness, and the conformal guarantee is void there.

## Training data

- **Response model**: perturbed resistant cells from the synthetic simulator
  (`make_response_data`). Training perturbations and training contexts only.
- **Sensitivity functional**: unperturbed sensitive vs resistant cells from a **separate
  simulation seed** (`make_sensitivity_data`), sharing no cells with the response data. The
  disjointness is deliberate — a sensitivity function fitted on the same experiment would
  partly read back the response model's own training signal, making a high reversal score
  self-confirming. The cost is a domain shift, discussed below.
- No real, human or patient data is used anywhere.

## Evaluation

Three split families, all defined and written to disk before anything is fitted: unseen
single perturbations (gene holdout), unseen combinations (both genes seen, and neither gene
seen), and unseen cell context. A held-out gene may not appear inside a training
perturbation of any size.

Metrics: Pearson-delta and MSE on top-20 DEGs computed on deltas against the matching
control; energy distance and MMD in latent space; reversal-score error and rank correlation;
top-k hit rate; conformal coverage and interval width; calibration error; risk-coverage AUC;
AUROC of using uncertainty to flag large errors.

Controls reported alongside every model:

- `shuffled_labels` — fitted with perturbation labels permuted. Whatever it scores is
  obtainable without knowing which perturbation was applied, so it is the real zero point.
- `control_mean` — predicts no change.
- `oracle_latent` — reproduces the observed delta through the latent basis; an upper bound,
  and also a diagnostic, since a readout under which a real model beats this oracle is a
  readout that cannot score reversal faithfully.

## Known limitations

- **No wet-lab validation whatsoever.**
- **Synthetic data only.** The simulator encodes a hypothesis about resistance states rather
  than a measurement of one. `LOG.md` documents three occasions where a plausible-looking
  result was an artefact of the simulator's construction, including one where an entire
  evaluation split was unsolvable in principle and every model scored zero.
- **The sensitivity proxy is the weakest link.** Its two readouts disagree sharply on
  predicted post-perturbation states (on a known true reversal from 0.10 to 0.98, the
  signature readout gave 0.21 to 0.95 and the latent classifier 0.11 to 0.23). Both readouts
  are fitted on unperturbed cells and applied to perturbed ones, which is extrapolation.
- **Phenotype labels carry about 10% noise** at the chosen co-expression strength, which
  caps achievable sensitivity accuracy near 0.90.
- **Predicted effect magnitudes are shrunk** to roughly 0.6x of true on unseen
  perturbations, biasing reversal estimates toward zero.
- **Conformal coverage is not guaranteed under cell-context shift**, and this is reported
  rather than worked around.
- **Gene-holdout splits are small** and are excluded from rank-correlation verdicts below
  five test perturbations.

## Ethical considerations and dual use

The realistic risk here is not misuse for harm but **overclaiming**. A pipeline that outputs
a ranked list of gene knockdowns labelled with a "reversal score" invites being read as a
therapeutic recommendation. It is not one, and the gap between a predicted transcriptional
shift in a simulator and a clinical effect is enormous and not quantified anywhere in this
work. The project's language rules exist for that reason: no "cure", no "treats", and no
claim that resistance was reversed.

A secondary concern is that the method identifies interventions that move cells *between*
phenotypic states, which is directionally neutral — the same machinery could in principle be
pointed at making cells less drug-sensitive. This is not a meaningful uplift: the underlying
perturbation-response modelling is standard published methodology, the simulator is not a
model of any real organism, and nothing here provides capability that was not already
available to anyone with the public datasets.

## Reporting

Every results file records the git commit, config path, and library versions, and carries a
`provenance` tag of `real` or `synthetic` so the two can never be silently combined in a
table. Negative results are reported; `H1` is expected to come out partly negative, and the
verdict functions are written to read the outcome from the numbers rather than to confirm a
preferred direction.
