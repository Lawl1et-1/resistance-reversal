# DATA

## Status: synthetic only

**Every result currently in `results/` is tagged `synthetic` and was produced by the
simulator in `src/simulator/grn.py`. No real dataset has been downloaded, inspected or
used.** Real-data acquisition is milestone M6, which is `should`-tier, and it has not
started.

This file exists partly to stop a plausible-looking table of accession numbers appearing
here before anything has actually been fetched. Nothing below is an accession ID, a URL or a
licence claim unless it is marked verified, and none of it is yet.

| tier | content | status |
|---|---|---|
| 0 | synthetic GRN simulator | **in use**, provenance `synthetic` |
| 1 | real perturbation response (scPerturb / GEARS loaders) | not acquired |
| 2 | real sensitive vs resistant, bulk pharmacogenomics | not acquired |
| 3 | compound-level response, bulk signatures | not acquired |

## Tier 0 - synthetic (in use)

Generated, not downloaded, so there is no accession or licence. Reproduced exactly by seed.

- **Generator**: `src/simulator/grn.py`, `build_grn(GRNConfig(...))`.
- **Configuration**: `configs/smoke.yaml` (60 genes, 12 regulators, 4 modules) and
  `configs/full.yaml` (200 genes, 20 regulators, 5 modules).
- **Structure**: a bistable toggle switch between a sensitive and a resistance master
  regulator; upstream regulators grouped into co-expressing pathway-like modules feed signed
  drive into the switch; two downstream programmes read it out. Observation adds noise and
  optional dropout and batch offsets.
- **Ground truth available**: the exact reversal score of every perturbation set of size
  <= k, by noise-free relaxation (`ground_truth_reversal`). This is the only setting in the
  project where ground truth is exactly knowable, which is why the must-tier hypotheses are
  tested here.
- **Two datasets, deliberately disjoint**: `make_response_data` (perturbed resistant cells,
  trains the response model) and `make_sensitivity_data` (unperturbed sensitive vs resistant
  cells, different simulation seed, trains the sensitivity functional).
- **Known limitations** recorded in `LOG.md`: ~10% phenotype-label noise at the chosen
  co-expression strength, and the two sensitivity readouts disagree sharply on predicted
  post-perturbation states.

Synthetic results validate the pipeline against known ground truth. They are **not**
biological evidence and are never presented as such.

## Tier 1-3 - real data (not acquired)

Candidate sources, recorded as *leads to verify*, not as citations. Each needs its
accession, licence, redistribution terms and download date confirmed before use, and the
names below are from the project brief rather than from anything checked:

- **Perturbation response**: the scPerturb harmonised collection; the GEARS `PertData`
  loaders for the Norman 2019 (K562 CRISPRa, singles and doubles) and Replogle 2022
  (K562/RPE1 CRISPRi) datasets. `TODO[verify]` for all of: accession, licence, exact
  contents.
- **Sensitive vs resistant, single-cell**: candidate systems are EGFR-TKI persisters in lung
  adenocarcinoma, BRAF-inhibitor-resistant melanoma, venetoclax-resistant AML. No specific
  study has been selected or verified. `TODO[verify]`.
- **Bulk pharmacogenomics**: DepMap/CCLE expression with GDSC2 and/or PRISM response,
  sensitive vs resistant defined by within-drug AUC or IC50 quantiles. `TODO[verify]`,
  including whether the licences permit redistribution (they generally do not, so raw data
  will not be committed).
- **Optional**: sci-Plex 3 for compound-level response; LINCS L1000 for bulk signatures.
  `TODO[verify]`.

## Policy when acquisition begins

1. Record, per dataset: source, accession, version, licence, download date, SHA256 of each
   file, and the exact loader used.
2. Raw data is **not** committed. `.gitignore` already excludes `data/raw/`, `data/cache/`
   and `*.h5ad`.
3. A failed download is recorded here with the error and the documented fallback, and the
   run continues rather than silently substituting something else.
4. Downloads go through the configured proxy. TLS verification is never disabled.
5. Any result computed from real data is tagged `real` in its results file, so a `real` and
   a `synthetic` number can never be confused in a table.

## Domain shift (applies once Tier 1 and Tier 2 are both in use)

The response model and the sensitivity functional will come from different experiments, cell
lines and protocols. The sensitivity functional is therefore a proxy evaluated off its
training distribution. Two independent readouts are used and their disagreement is reported
rather than averaged away. On synthetic data that disagreement is already large (see
`LOG.md`, M4), which is a reason to expect it to be worse on real data, not better.
