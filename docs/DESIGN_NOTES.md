# Design notes

Decisions where the proposal left room, and things found while building the pipeline. None of
these used an outcome from the test period.

## Data

- **Fresh snapshots replace the old `cripto` files.** The old repository only had ETF-paired hourly panels and daily IV through 2026-03; it had no NDX close and no pre-2010 SPX. It is still used as a cross-check in the QC.
- **Yahoo has a wrong SPX print.** On 2021-08-11 it reports 4442.41; the official close is 4447.70. Yahoo and FRED also differ by more than 1 bp on 2019-08-12 (SPX), 2016-04-18 and 2024-10-22 (NDX). The panel therefore takes the issuer's close through FRED wherever it exists: all of NDX and SPX from 2016-11-03. Yahoo supplies the session calendar (FRED lists non-sessions such as 2019-04-19) and SPX before 2016-11.
- **SPX before 2016-11 has no second index source.** It was screened against SPY total returns instead. No gap above 50 bp appears before 2020, and the flagged gaps fall in March 2020 and April 2025, when closes are least synchronous. The ETF enters this QC only.
- **The old `cripto` VXN file has a bad close.** On 2024-02-01 it holds 11.20, which is that day's bad LOW print; the correct close is 17.33. The current Cboe and FRED files are right.
- **Retained oddity.** Exact zero returns on SPX 2017-01-10 and NDX 2012-03-12 appear in both sources, so they stay. `log(rv + 1e-7)` keeps them finite.

## Sensitivity analyses that would be vacuous

Annualised variance and the 21/252 convention are constant rescalings of Eq. (13). All three
families standardise or are invariant to them, so they produce identical forecasts
(`test_constant_rescaling_of_v_is_vacuous`). They are not run. The non-vacuous alternatives kept
are the volatility level (`vol_level`, non-affine) and the weekday-count window (`weekday_cal`,
varying by origin).

## A finite-sample result worth adding to Section 3

Fit `f0` and `f1` by OLS on the same rows, and project with the OLS regression of `V` on the same
regressors. Then HFS is **exactly zero at every x**, in or out of sample: the omitted-variable
identity `b0 = b1 + b_V * gamma` gives the same fitted function
(`test_ols_hfs_vanishes_with_the_matching_linear_law`).

This is a finite-sample analogue of Eq. (6) that does not need correct specification. For the HAR
family, nonzero HFS can only come from the QLIKE weights or from the projection law, and "IV
reorganised the HAR persistence" is an accounting statement about `gamma`. The per-fold accounting
is in `har_coefs_*.csv`.

## Models

- **`reg:gamma` level.** Overfitting noise lowers the mean forecast by construction, because the log-link first-order condition is `mean(y/mu) = 1`. Replacing the observed Hessian with Fisher scoring did not help (0.885 vs 0.875 of the mean at 400 rounds on noise, and no QLIKE gain on pre-2017 SPX), so the standard objective stays. Validation-chosen rounds limit the effect, placebo fits measure what remains, and `level_ratio` is reported for every model.
- **Forecast floor.** `0.25 * q01(training Y)`, identical across families, with the floored share reported.
- **Augmented elastic net.** It sees `log V`, `log V` squared and all `log V * x` interactions, so `f1` can actually condition on IV. Inputs are clipped to the training range, so it never extrapolates.

## Conditional laws and support

- **Why `n_eff` no longer filters law A.** The outcome-blind design check showed the kernel `n_eff >= 30` rule withholding 70% of 2018 and 61% of 2022, while the k-NN rule accepted 98% of origins. The kernel bandwidth is chosen for CRPS, not for coverage of states, and law A does not use neighbourhoods. So `sup_x` is the k-NN rule and governs law A, and `sup_b` restricts only law B quantities. A reader comparing laws should use `rms_HFS_on_sup_b` against `rms_HFS_b`.
- **Same states for observed and placebo fits.** Laws depend only on a fold's training rows and are shared (`test_placebo_and_observed_share_support_states`).

## Matching

Chosen from `(X, V)` alone, using balance and pair counts:

| Metric | k | Pairs | Dates | Fewest pairs in a year | RMS z-gap, `rv_m` |
|---|---|---|---|---|---|
| Whitened, 7 features | 5 | 1,726 | 1,170 | 25 | 0.26 |
| Whitened, 7 features | 50 | 6,480 | 1,911 | 243 | 0.39 |
| Standardised, 7 features | 5 | 1,604 | 1,153 | 21 | 0.24 |
| **Standardised, 7 features** | **20** | **4,297** | **1,704** | **82** | **0.31** |
| Standardised, 7 features | 50 | 6,363 | 1,924 | 227 | 0.36 |
| Standardised, 3 features | 20 | 4,035 | 1,758 | 167 | 0.14, but `rv_d` gap 1.16 and `dd_22` gap 0.51 |

Whitening inflates the near-collinear noise directions of the HAR terms. Dropping features looks
tighter on what remains but leaves the last-day shock and the drawdown unmatched. Standardised
7-feature distance at `k = 20` is primary, with `k = 5, 50` as sensitivity.

## Pair construction beyond the proposal

Both members of a pair are forecast by the earlier date's fit. Its labels closed before that
date, so both forecasts are strictly out of sample and the contrast has no refit drift. Date
bootstrap weights multiply across a pair, so pairs sharing a date stay dependent.

## Placebos

The proposal's `V~ ~ p(V | X)` drawn iid has no persistence, which may make it easier to ignore
than real IV. The primary placebo therefore uses moving blocks of training residuals, with iid
draws and displaced `V_{t-L}` as supplements. The generator uses each fold's training rows only
(`test_placebo_is_built_from_training_only`).

## Simulation

- **DGP.** Calibrated on pre-2017 SPX. `log V` carries the proposal's `a(X) + u`, so V stays positive. Overlapping windows are built so that `Y | X, V` is still exactly `Gamma(k, mu/k)`, which keeps closed-form `m0`, `m1`.
- **Evaluation.** All population identities are also checked against Monte Carlo draws.
- **Redundant is not harmless for a misspecified fit.** In the pilot (64 replicates), a redundant V lowered the linear HAR's population QLIKE by about 0.004, because V proxies for the nonlinearity HAR lacks. The test detecting this is working; the mistake would be reading it as new information. This is the proposal's "artificially weak model" warning, measured.
- **Pilot detection rates, primary block 63 (64 replicates per design, Monte Carlo s.e. about 0.03).** Under the redundant design, a falsely favourable matched slope appeared in 0-8% of replicates; a falsely favourable matched `dqlike` in 0-6%, and 14% for HAR. Against that, `b = 0.25` with state dependence gave 98-100% and 67-81% respectively. The final numbers come from the full run.
- **Known limitation: `dqlike` intervals are too narrow.** Percentile block-bootstrap intervals cover the population `dqlike` of a fitted pair about 89% of the time against a nominal 95% (120 replicates per design, 3 learners, population value from a 120,000-origin path). Block length does not fix it: overall coverage is 0.85, 0.89, 0.89, 0.88, 0.86 at blocks 21, 63, 126, 252, 504, and the bootstrap standard deviation is 0.64-0.97 of the true one at block 63. Block 63 stays primary. Read `dqlike` intervals as somewhat optimistic, and report the coverage from `summary_main.csv` alongside them.

## Addendum C implementation (2026-11-17)

- **Same draws, not new ones.** The decomposition's placebo and refit draws re-run the fits of
  scripts 06 and 10 with their seeds, scoring only the per-origin rows. On the dev profile the
  placebo statistics of both routes agree to 0.0. From 2026-11-17, `run_placebo` also stores the
  `ti:` statistics in every new draw, and script 12 copies them instead of refitting (the nn2 NDX
  placebo, which had not run yet, needs no second GPU pass).
- **Level recalibration is real time.** QLIKE's optimal scalar on past data is `mean(y/f)`; it is
  computed over earlier OOS origins whose window had closed (`test_realtime_scale_uses_only_closed_labels`).
- **No neural refit bootstrap for Addendum C.** It would cost about one GPU-day per asset; the
  neural cells are secondary, so they use the date bootstrap and the placebo.
- **Dev smoke run.** `results/dev/` holds a pre-2017 end-to-end run of scripts 01, 04–07 and 12–14,
  for debugging only.

## Addendum D implementation (2026-11-18)

- **Shared collection module.** The collection helpers of script 14 moved to
  `src/ivbehav/teacher_report.py`. Rerunning script 14 afterwards reproduced `claims.csv`,
  `secondary.csv`, `simulation_oc.csv` and `REPORT.md` byte for byte.
- **Refit draws.** Script 12 `--part refit` is spec-generic. Addendum D runs it on h5td and h10td with
  the primary budgets (har 500, enet 500, xgb 200), using the same seeds as the 30-day draws.
  Draw 0 on NDX h10td was timed at about 25 s per core for har and xgb; the values were not looked
  at.
- **Collector check.** Script 15 was checked on the 30-day cells, which had already been seen. Its
  refit+date intervals equal those of script 14.

## Addendum E implementation (2026-11-21)

- **Override, not a new profile.** `primary.yaml` is untouched (hash `b33d50a4c3cb4bcb`). Script 12
  takes `--refit-draws N`; its manifest records the override. Seeds depend on the draw index only
  (`seed + 100000 + draw` for the fit, `[seed, draw]` for the date weights), so draws 0..N-1 of a
  larger run include every earlier draw unchanged.
- **Check.** Appending draw 500 to SPX primary har moved the combined input interval from
  [-0.5731, 0.0419] to [-0.5729, 0.0418] (n 500 to 501), as expected for one extra draw.
- **Snapshot.** `results/primary/pre_addendumE/` holds the Addendum C and D outputs and the
  `teacher_refit_*.csv` tables as they were before the amendment.
- **Collectors.** Scripts 14 and 15 now report `n_refit_draws` per cell.
