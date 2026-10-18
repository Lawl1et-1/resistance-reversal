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
