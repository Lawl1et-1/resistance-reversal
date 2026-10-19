# DECISIONS

Every analysis rule that was changed **after** the full-profile results came back, what it
was before, what it is now, why, and what the verdict would have been under the original
rule on the same data.

This file exists because all three revisions moved a verdict toward a cleaner story. Each is
defensible in isolation; together they are a forking path, and a reader cannot judge them
without seeing the counterfactual. `python -m experiments.decision_audit --profile full`
recomputes the table below from the saved raw rows, so it cannot drift from the code.

**The short version: the H1a verdict is rule-dependent and should not be read as a clean
result.** The statistical correction that was most defensible did not change it; the
redefinition made after seeing the counts did.

---

## 1. Pairing unit for model comparisons (H1)

| | |
|---|---|
| **Original** | Paired on `(seed, perturbation, context)` — about 720 rows per comparison. |
| **Revised** | Aggregate to one mean per seed first — 5 rows per comparison. |
| **Why** | The 720 rows come from 5 fitted models on 5 networks; perturbations within a seed share a model and are not independent. The symptom was tests contradicting their own intervals: `mean_diff 0.0060, CI [-0.0053, 0.0179], p = 0.0000`. |
| **Trigger** | Noticed while reading the full-profile output, i.e. **after** the data. |
| **Effect on the verdict** | **None.** wins=5 / losses=4 under both pairings with the CI-only criterion. |

This is the one change I would defend without reservation, and it is the one that changed
nothing. Worth stating plainly, because it is evidence that the other two did the work.

## 2. Significance criterion (H1)

| | |
|---|---|
| **Original** | "Significant" = bootstrap CI on the mean paired difference excludes zero. |
| **Revised** | "Decisive" = CI excludes zero **and** \|Cohen's d_z\| > 1. |
| **Why** | With 5 seeds the Wilcoxon signed-rank test cannot return p < 0.0625 whatever the data, so p-value significance is unreachable by construction. A percentile bootstrap over 5 points is also fragile on its own. |
| **Trigger** | After the data. |
| **Effect on the verdict** | Changes the counts (5/4 to 4/3 at seed level) but **not** the verdict under either rule. |

The threshold \|d_z\| > 1 is conventional for "large", but it was not pre-registered, and a
different threshold would change the counts.

## 3. The "reliably beats" rule (H1a) — **this is the one that flipped the verdict**

| | |
|---|---|
| **Original** | `H1a = (wins == 0) or (losses >= wins)` |
| **Revised** | `H1a = not (wins > 0 and losses == 0)` |
| **Why** | 4 decisive wins against 3 decisive losses out of 20 comparisons is not "reliably winning" in any ordinary sense, and the original rule called it one. |
| **Trigger** | After seeing wins=4, losses=3. |
| **Effect on the verdict** | **Flips it.** |

Audit output across every combination of the three choices above:

| pairing | criterion | rule | wins | losses | H1a |
|---|---|---|---|---|---|
| row | CI only | original | 5 | 4 | **False** |
| row | CI only | revised | 5 | 4 | True |
| row | CI + effect | original | 0 | 1 | True |
| row | CI + effect | revised | 0 | 1 | True |
| seed | CI only | original | 5 | 4 | **False** |
| seed | CI only | revised | 5 | 4 | True |
| seed | CI + effect | original | 4 | 3 | **False** |
| seed | CI + effect | revised | 4 | 3 | True *(as reported)* |

The revised rule yields `True` in all four configurations; the original yields `False` in
three of four. **The reported H1a verdict is a consequence of the rule change, not of the
data or of the statistical correction.**

### How H1 should therefore be stated

Not as "H1 supported". The defensible claim is the descriptive one, which holds under every
rule above:

> Across 20 paired comparisons on held-out splits, deep models won decisively 4 times and
> lost decisively 3 times. The largest deep advantage was +0.097 Pearson-delta; the largest
> deficit was -0.605, about six times larger. Whether this counts as "failing to reliably
> beat" linear baselines depends on a threshold we set after seeing the counts.

## 4. H3 saturation rule

| | |
|---|---|
| **Original (v1)** | `np.allclose(verified, oracle, atol=1e-6)` |
| **Original (v2)** | Mean regret across **all** lambdas <= 1% of the oracle value. |
| **Revised** | Regret at **lambda = 0 only** <= 1% of the oracle value. |
| **Why** | Lambda = 0 is the baseline H3 proposes to improve on. Averaging over all lambdas mixes in runs whose regret is high *because the penalty over-corrected*, which hides whether the baseline had headroom. |
| **Trigger** | After the data. |
| **Effect on the verdict** | **Flips it**: regret 0.0486 (all lambdas, not saturated, H3 reported `False`) vs 0.0006 (lambda = 0, saturated, H3 reported untestable). |

Unlike change 3, this one is defensible from the definition of the hypothesis rather than
from the outcome: H3 is a claim about improving on uncertainty-blind search, so headroom
must be measured at the uncertainty-blind setting. I would make the same change having seen
no data. But it was in fact made after, and it did change the verdict.

## 5. Optimism gap — a **bug fix**, not a rule change

| | |
|---|---|
| **Original** | `optimism_gap = predicted_reversal - oracle_verified_reversal` |
| **Revised** | `optimism_gap = predicted_reversal - observed_reversal_through_the_same_readout` |
| **Why** | The two terms were on different scales. Predicted reversal passes through the *learned* signature readout; oracle reversal uses the simulator's `true_sensitivity`. Their difference is not optimism, it is mostly the offset between the two scales. |

Decomposition on the smoke profile, which makes the error explicit:

| quantity | value |
|---|---|
| predicted (learned readout) | 0.767 |
| observed, same learned readout | 0.714 |
| **corrected optimism gap** | **+0.054** |
| pure scale offset (readout vs oracle) | **-0.238** |

The originally reported full-profile figure of **-0.707** was therefore not evidence that the
model underestimates reversal. The claim "there is no optimiser's curse to correct because
the model systematically underestimates" was **withdrawn** and recomputed; see `LOG.md`.

The oracle scale is still used for ranking, regret and the lambda comparisons, where only
ordering matters and a constant offset cancels.

## 6. Confidence-interval method — changed after the fact, **widens** every interval

| | |
|---|---|
| **Original** | Percentile bootstrap over all rows in a (model, split) cell — about 720 rows. |
| **Revised** | Hierarchical bootstrap: resample seeds, then perturbations within the drawn seeds. |
| **Why** | Same nesting problem as change 1. Pooling rows treats perturbations as independent replicates and reports intervals far tighter than the evidence supports. Collapsing to 5 seed means is honest but discards within-seed information. |
| **Trigger** | After the data, prompted by review. |
| **Effect** | Every interval widens. On `unseen_single` the widths go from roughly 0.05 to roughly 0.50. |

This change makes every claim **weaker**, which is the opposite direction from changes 2
and 3, and it is the reason change 7 below became visible.

## 7. Comparison against the shuffled-label control — an **added** analysis

| | |
|---|---|
| **Original** | The control was reported in tables; no explicit per-split test of separation. |
| **Revised** | Each split now reports whether any model's interval clears the control's upper bound. |
| **Why** | With honest (hierarchical) intervals, several models' intervals visibly overlapped the control's, which is the one comparison the control was built to support. |
| **Trigger** | After the data. |
| **Effect** | **Adds a negative finding.** |

Result: on `unseen_single` and `unseen_combo_unseen_genes`, **no model is distinguishable
from the shuffled-label control**. Ridge reaches 0.416 [0.157, 0.655] against the control's
0.260 [0.038, 0.474].

This is reported because it is the most defensible finding in the project: it rests on a
control that was built in before any results existed, it needs no threshold chosen after the
fact, and it makes the project's conclusions more conservative rather than less. It
supersedes the gene-holdout columns of the H1 table, which should not be read as evidence of
generalisation to held-out genes in either direction.

---

## Rules fixed before the next run

To avoid repeating this, the following are set now, before any further experiment:

1. A hypothesis verdict is reported as a **descriptive statement with an interval**, and any
   binary call states the rule and threshold inline.
2. Thresholds (`|d_z| > 1`, 1%-of-oracle saturation, 5 test perturbations minimum) are
   frozen as written here. Changing one requires an entry in this file with the
   counterfactual verdict.
3. Any quantity formed as a difference must have both terms on the same measurement scale,
   asserted in code where the quantity is built.
4. If a new regime is built to make H3 testable, the regime's design is written down and
   committed **before** it is run.
