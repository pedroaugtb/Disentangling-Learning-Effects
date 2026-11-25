# Pre-analysis plan

Fixed before any model was fitted on the 2017+ test period or on NDX. Development used pre-2017
SPX data only (`configs/dev.yaml`). The outcome-blind design check used `(X, V)` from the test
period, but no label value. Every output carries the sha256 of the merged config in its
`manifest_*.json`, so later edits are visible.

## Design

| Item | Primary choice |
|---|---|
| Asset / signal | SPX / VIX; NDX / VXN is a prespecified replication |
| Sample | closes 2010-01-04 to 2025-12-31; first OOS origin 2017-01-03; last labelled origin 2025-12-01 |
| Target | `Y30`: sum of `r_s^2` over `t < s <= t + 30` calendar days (Eq. 12) |
| Signal | `V = (I/100)^2 * 30/365` (Eq. 13) |
| History | `rv_d, rv_w, rv_m, rv_neg_w, ret_1, ret_5, dd_22` |
| Walk-forward | expanding window, monthly refit, annual retuning, last 2 years as purged validation; a label trains only if its window has closed by the fit date |
| Families | HAR (identity link), elastic net (log link), XGBoost (`reg:gamma`), all minimising QLIKE with the same floor rule |

## Statistics

All are computed by one code path (`core.py`, `stats.py`) for the observed fit and for every
placebo, refit and variant.

- **Decomposition:** `D = f1 - f0`, `HFS = P1 - f0`, `ISD = f1 - P1`, with `P1` from law A (location-scale, training residuals). Law B (kernel) is the sensitivity law, read only where its Kish `n_eff >= 30`.
- **Response, feature `k` in `{rv_w, rv_m, rv_neg_w}`:** stitched OOS ALE on training-quantile bins.
  - `S_m`: change between the q10 and q90 training edges.
  - `dS = S_1 - S_0`: slope contrast.
  - `T`: count-weighted L2 distance between the centred curves.
- **Matched pairs:** dates with nearby standardised histories, differing IV surprises and disjoint target windows, both forecast by the earlier date's fit, oriented so `hi` has the larger surprise. Caliper and thresholds come from the training segment.
  - `sep_slope_rel`: slope of `(Y-f0)/f0` differences on `D/f0` differences. It equals 1 for ideal predictors by Eq. (8) and 0 if the separation carries no outcome information.
  - `dqlike`: mean loss difference on the matched dates.

## Hierarchy

Confirmatory scope: spec `primary`, asset SPX, three core families. Primary block length: 63
sessions. Primary placebo: `cond_block`, moving blocks of training residuals, which keep the
persistence of the surprises. Assembled by `scripts/11_collect_primary.py` in `hierarchy.csv`.

**P1, falsification gate, per family.** Passes only if both hold:

1. `rms_D_rel`, or `T` for at least one primary feature, exceeds the placebo distribution (`p_upper <= 0.05`, the exact Monte Carlo rule).
2. `rms_D` exceeds the q95 of repeated-fit distance from the refit bootstrap (`rms_self_f0`, `rms_self_f1`).

A family failing the gate is reported as showing no behavior distinguishable from artifacts.
Its P2 and P3 results are still shown, labelled as ungated.

**P2, conditional reorganisation.** Nine (family, feature) cells. A cell is claimed only if both hold:

- `T` beats the placebo with Holm-adjusted `p < 0.05` across the nine cells.
- The block-bootstrap interval of `dS` excludes zero, with Holm-adjusted `p < 0.05` across the nine cells.

**P3, validated state differentiation.** Six tests: `sep_slope_rel` and `dqlike` for each family,
two-sided block bootstrap, Holm across the six. A family is claimed if either holds:

- `sep_slope_rel` is positive and significant, and also above the placebo q95.
- `dqlike` is negative and significant.

P2 and P3 are the two primary empirical targets and are not corrected against each other.

**Reading P3 for HAR.** Linear HAR has no nonlinear history terms, so even a redundant V can lower
its population loss by correcting functional form (seen in the simulation's redundant design). A
HAR `dqlike` claim alone is therefore not evidence of incremental information. Every P3 claim is
read against the false-detection rate of the same family and test in `simulation/summary_main.csv`.

**Reading `dqlike` intervals.** The pilot simulation found them too narrow (about 89% coverage at a
nominal 95%, at every block length tried). They are reported with the coverage measured in the full
simulation, and a borderline `dqlike` result is not treated as settled.

**Secondary (no confirmatory claim).**

- `dSp`, `T_shape`, `S1_high_minus_low`, `conc_diff`, `mean_dlogY`, `calib_slope_rel`.
- Law B statistics, IV-surprise strata.
- Caliper sensitivity `k = 5, 50`; excluding 2020-02-15 to 2020-04-30.
- Accuracy tests (DM, GW), which are validity checks only.

**Exploratory.** WP6 pre-onset paths; the number of independent episodes is small by construction.

**Sensitivity specs (read against the primary, never replacing it).**

- `vlag1`, `rolling5y`, `vol_level`, `weekday_cal`.
- `garch`: a price-only control signal.
- `h5td`, `h10td`: horizon-mismatched targets.
- Log-target learners, read separately because `exp(E[log Y])` is not `E[Y]`.

**Replication.** NDX/VXN runs the identical code and config with the asset swapped, after the SPX
decisions are final. It adds no hypothesis. It is reported as agreement in sign and interval with
SPX, and supports only within-equity-index portability.

## Null results that count

The study reports each of these as a finding:

- unstable displacement;
- changes only in attribution;
- separation that does not track outcomes;
- results that fail replication;
- persistently nonzero HFS without outcome validation, which is algorithmic reorganisation and not new information (Eq. 11).

## Addendum A: small neural forecasters (2026-11-05)

Written after the SPX core results were seen, before any neural model or any NDX model was fitted.
Config `configs/nn.yaml` (profile `nn`); everything not listed is inherited from `primary.yaml`.

- **Families:** `mlp` reads the same seven summaries as the core; `tcn` (causal dilated convolution) and `attn` (one-layer transformer) read the last 66 daily returns. All are 8-member ensembles trained on QLIKE with a log link.
- **Change, 2026-11-11, before any neural result:** `attn` is dropped because it took about 75% of the GPU budget; `tcn` remains the sequence model. Results are under profile `nn2`.
- **Status:** a robustness extension of WP7. For SPX it is read against the core results and never replaces them; for NDX it is declared here in advance.
- **Statistics:** same code path and claim rules as the hierarchy. P2 applies to `mlp` only, because summary-based response functions are undefined for sequence models. For `tcn` and `attn`, P1 integrates V over the law conditional on the seven summaries, a coarsening of what the model reads. Their HFS/ISD split is therefore secondary; `D` and the matched-pair statistics are unaffected.
- **Budgets (deviations):** quarterly refit with an annual retune; 12-candidate grids; 200 `cond_block` placebo draws (no iid draws, displaced lag 252 only); 100 refit-bootstrap draws; reduced multiplicity.

## Addendum B: NDX replication targets (2026-11-05)

Fixed after the SPX results, before any NDX model was fitted. NDX runs `primary.yaml` unchanged
(config hash `b33d50a4c3cb4bcb`), and its full hierarchy is reported with the original rules.
Replication is judged per family on:

1. **P2:** for each of the nine cells claimed on SPX, NDX `dS` has the same sign, its interval excludes zero, and `T` beats the `cond_block` placebo, with Holm across the nine.
2. **Directional state differentiation:** SPX secondary statistics promoted to replication targets. `conc_diff > 0` and `mean_dlogY > 0`, with two-sided block-bootstrap p under Holm across the six (family x statistic) below 0.05, and placebo `p <= 0.05`. They are read as option-specific only if the `garch` spec does not reproduce them.
3. **P1 and the P3 primary tests** keep their original rules. The SPX failures there are reported again on NDX, not re-tested as new hypotheses.

The annual-retune weakness seen on SPX in 2020 (near-constant flexible models) applies equally to
NDX. It is declared, not fixed.

## Addendum C: teacher or input? Decomposing the gain from V (2026-11-17)

Written after the core, sensitivity and nn2-SPX behavioral results had been seen, and **before
any loss of the projection `P1` was computed on any test period** (SPX or NDX). Parameters are
in `configs/teacher.yaml`, kept apart from `primary.yaml` so that the hashes of finished profiles
do not change. Code: `src/ivbehav/teacher.py`, `teacher_sim.py`, scripts 12–14, `scripts/run_final.sh`.

### Question and decomposition

The hierarchy shows that V robustly changes how the forecasters use the history (P2), but not
whether that change is worth anything. With `P1(x) = E[f1(x, V) | x]` under law A:

    L(f0) - L(f1)  =  [L(f0) - L(P1)]  +  [L(P1) - L(f1)]
     gain_total        gain_teacher         gain_input

- **Teacher gain.** `P1` uses the history only, so this is what fitting with V taught the learner
  about the history. It stays available without observing V at forecast time.
- **Input gain.** The value of observing the V surprise in real time; the only part that can be
  incremental information.
- Gains are positive when they favour the second forecast; the identity holds row by row.
- Loss: QLIKE (the training objective), on every common OOS origin, the same set as the
  `validity` `dqlike`. Relative gains divide by `L(f0)`.

**Theory anchor.** For OLS `f0`, `f1` on the same rows and a projection law whose mean is the OLS
fit of V on the same regressors, `P1 = f0` at every x, so the teacher gain is exactly zero
(`test_ols_teacher_gain_is_zero_under_the_linear_law`). A teacher gain of the linear HAR can
therefore come only from its QLIKE weights or from law A; a flexible learner can also learn a
different function of the history. `har_ols` under law A is reported as the anchor.

### Inference

- **Date uncertainty.** Moving-block bootstrap over OOS dates, 2000 draws, blocks 21/42/63/126;
  63 is primary.
- **Combined uncertainty (confirmatory cells).** The refit-bootstrap draws of script 10 (same seeds,
  same training resamples, same combined date weights), scored on the decomposition. When
  present, claims use these intervals and p-values.
- **Placebo.** The `cond_block` draws of script 06 (same seeds, same refits; checked to be identical
  on the dev profile). A persistence-matched fake V tests whether *any* persistent extra input
  produces the same gains, e.g. by regularising the fit. This is the null for both components.
- **Simulation.** Script 13: same DGP and learners as script 02. The ideal teacher gain is zero by
  construction (`E[m1 | Z] = m0`), so the audit separates what a fitted learner learns from V
  from what V adds. It reports bias of law A against the true law, coverage of the intervals for
  the population gains of the fitted pair, and the rate at which the full C1 and C2 rules fire.

### Confirmatory cells and rules

Cells: SPX and NDX × {har, enet, xgb}, spec `primary`, block 63. Both assets enter because no `P1`
loss was seen on either. Holm runs across the six cells, separately for each component.

- **C1, input information.** Claimed if `gain_input > 0`, its Holm-adjusted p < 0.05, and the
  placebo `p_upper <= 0.05`. A family is downgraded to *not claimable* if the C1 rule fires in more
  than 10% of the redundant-design replicates for its simulation proxy (enet: the worse of
  `enet_light`, `enet_heavy`).
- **C2, teaching.** "V teaches beyond a placebo" if `gain_teacher > 0`, Holm p < 0.05 and placebo
  `p_upper <= 0.05`. "Harmful reorganisation" if `gain_teacher < 0` with Holm p < 0.05. Under the
  redundant design a positive teacher gain is a real functional-form effect, not information;
  the simulation's `pop_teacher` measures its size per learner.
- **C3, non-superiority.** The one-sided 95% upper bound (q95 of the combined draws) of
  `gain_input / L(f0)` below 5% "rules out an input gain of 5% of the loss or more". The same
  bound is reported for the total gain. The margin is the paper's statement of a negligible gain.
- **Dissociation (the paper's thesis, per family).** At least one P2 feature claimed in
  `hierarchy.csv`, C1 not claimed, and C3 holds. Reported per asset; no cross-asset pooling.

### Secondary (no confirmatory claim)

- Specs `garch` (option specificity: price-only signal), `h5td`, `h10td` (horizon), with placebo; the
  other sensitivity specs with the date bootstrap only.
- Law B on its supported states, against law A on the same states.
- Real-time level recalibration: each forecast is multiplied by `mean(y/f)` over earlier OOS
  origins whose label had closed, from 250 closed labels on. Reported with the raw gains on the
  same origins. This answers whether a teacher gain is only a level correction.
- Excluding 2020-02-15 to 2020-04-30; `sup_x` only; MSE.
- DM tests of the two differentials; gains by historical-variance tercile × IV-surprise tercile and
  by year (descriptive, the counterpart of the GW result in the core study).
- Neural families (`nn2`: mlp, tcn) with the date bootstrap and placebo, no refit bootstrap. For
  `tcn`, law A conditions on the seven summaries only (Addendum A), so its split is coarser.

### Results that count

Each of these is reported as a finding: a teacher gain that a fake persistent signal reproduces
(regularisation, not options); a teacher gain that vanishes under recalibration (level only); an
input gain on NDX but not on SPX; a significantly negative component offset by a positive one.

## Addendum D: post-hoc confirmation of the short-horizon input gain (2026-11-18)

**Status: post-hoc.** Written after every Addendum C result had been seen, including the
secondary h5td and h10td decompositions (date bootstrap and placebo). Its claims carry less
weight than those of Addendum C and are reported as a confirmation of a pattern found in the
secondary analyses, never as a prespecified test. Parameters: `configs/teacher_short.yaml`.
Code: `scripts/15_collect_short.py`, `scripts/run_short.sh`.

### What had been seen

- At 30 days no rule of Addendum C fired in any confirmatory cell: C1, C2 and C3 all fail, so the
  dissociation is not claimed. With the refit+date intervals, the input gain cannot be bounded
  below 6–16% of `L(f0)`.
- At 5 trading days, on the date bootstrap: the input gain was 7–11% of `L(f0)` in all six
  asset × family cells, with placebo `p_upper` ≤ 0.002 in all six and date p < 0.05 in five.
  The teacher gain was 0–2.5%, not significant. The input gain stayed positive under law B and
  under real-time recalibration.
- At 10 trading days the pattern was weaker and mixed.
- **Not seen:** any refit-bootstrap draw of h5td or h10td, and therefore any combined interval,
  any Holm adjustment across these cells, and the input-minus-teacher contrast.

### Cells and rules

Cells: SPX, NDX × h5td, h10td × har, enet, xgb (12 cells), primary block 63. The 10-day cells are
included so the confirmation is not limited to the best-looking horizon. All p-values come from the
refit+date draws (the refit-bootstrap seeds of script 10, run through script 12). Holm runs across
the 12 cells, separately per rule. A cell without refit draws gets no claim.

- **D0, total gain.** `gain_total > 0` with Holm p < 0.05.
- **D1, input information.** As C1: `gain_input > 0`, Holm p < 0.05, placebo `p_upper` ≤ 0.05, and
  the same simulation downgrade (simulated at the 30-day horizon; stated as a limitation).
- **D2, teaching.** As C2.
- **D3, input exceeds teacher.** `gain_input − gain_teacher > 0` with Holm p < 0.05, computed draw by
  draw on the same refit+date draws. This is the statement "the short-horizon gain is information,
  not a better use of the history".

### Reading

- **Supported.** "At short horizons the value of IV is real-time information" is supported for an
  asset × horizon if D1 and D3 hold in at least two of its three families. The count is reported
  per asset and horizon, with no pooling across assets.
- **Partial.** One family is reported as partial support.
- **Not supported.** Zero families is reported as the short-horizon pattern not surviving the
  refit uncertainty. That is a finding the paper must state.
- **Term structure.** Relative gains at 5, 10 and 30 days with their intervals are descriptive.
  No test across horizons is made, because origin sets differ by horizon (2257, 2252 and 2241
  dates) and the date weights are therefore not paired.

## Addendum E: refit-bootstrap budget of the teacher / input claims (2026-11-21)

**Status: amendment of a budget, written after the Addendum C and D results had been seen.** No
rule, cell, statistic, seed, block length or alpha changes. Parameters: `configs/teacher_amend.yaml`.
Code: `scripts/run_amend.sh` (script 12 `--refit-draws`, then scripts 14 and 15).

- **Defect.** A percentile p-value from B draws cannot fall below 2/(B+1). With the 200 xgb refit
  draws inherited from script 10, the floor is 0.00995, which is 0.060 after Holm over the six
  cells of Addendum C and 0.119 over the twelve cells of Addendum D. Those cells could not be
  claimed whatever the data showed. With 500 draws (har, enet) the Addendum D floor after Holm is
  0.048, which leaves almost no room either.
- **Change.** Every Addendum C and D cell gets 2000 refit+date draws, the same budget as the date
  bootstrap. The floor becomes 0.0010, i.e. 0.006 and 0.012 after Holm. Draws are indexed by seed, so
  the existing draws are kept and new ones appended; the old results are therefore exactly the
  first 200 or 500 draws of the new ones.
- **What had been seen.** Every Addendum C and D result, including that NDX h5td xgb sat exactly at
  the Holm floor. The amendment is motivated by the floor, a property of the design, not by any
  estimate; it applies to all cells and families alike, including those far from significance.
- **Reporting.** The claims with the original budgets stay in `results/primary/pre_addendumE/` and
  are reported next to the amended ones. A claim that appears only under the amended budget is
  labelled as such.
- **Not changed.** The hierarchy (P1–P3) uses the refit draws of script 10 only through the q95 of
  the repeated-fit distance, which has no resolution floor; its budgets stay as they are. Placebo
  p-values (500 draws, floor 0.002) enter the rules as a gate at 0.05 and through Holm over nine
  P2 cells (0.018), so they are not affected.
