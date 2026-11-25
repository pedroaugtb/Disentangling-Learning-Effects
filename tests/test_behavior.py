"""Conditional laws, response statistics, pairs, inference and the shared evaluation path."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from ivbehav.conditional import KernelLaw, LocationScaleLaw, SupportChecker, fit_laws, weighted_crps
from ivbehav.core import evaluate_fold, select_pairs
from ivbehav.data import HIST_COLS
from ivbehav.explain import conditional_group_shapley, conditional_reliance
from ivbehav.folds import make_folds, oos_positions
from ivbehav.inference import dm_test, gw_test, hac_slope, holm, mbb_indices, mbb_weights, newey_west_lrv, placebo_p
from ivbehav.models import HARModel, fit_pair
from ivbehav.pipeline import displaced_path, placebo_path
from ivbehav.stats import OriginStats, PairStats, ResponseStats, stitched_stats


def _law_sample(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = np.exp(rng.normal(-9, 0.6, (n, len(HIST_COLS))))
    X[:, 4:] = rng.normal(0, 0.01, (n, 3))
    z = (np.log(X[:, 2]) + 9) / 0.6
    V = np.exp(-5 + 0.5 * z + np.exp(-1.5 + 0.3 * z) * rng.standard_normal(n))
    return X, V


def test_location_scale_coverage_and_pit():
    X, V = _law_sample()
    Xt, Vt = _law_sample(seed=1)
    law = LocationScaleLaw(200).fit(X, V)
    pit = law.pit(Xt, Vt)
    assert abs(np.mean(np.abs(pit - 0.5) <= 0.4) - 0.8) < 0.025
    vals, wts = law.draws(Xt[:5])
    np.testing.assert_allclose(wts.sum(axis=1), 1.0)
    assert np.all(vals > 0)
    np.testing.assert_allclose(law.mean(Xt[:5]), (vals * wts).sum(axis=1))


def test_kernel_law_coverage_and_neff():
    X, V = _law_sample(3000)
    Xt, Vt = _law_sample(500, seed=1)
    law = KernelLaw([0.5, 1.0, 2.0, 4.0], top_m=300).fit(X, V)
    pit = law.pit(Xt, Vt)
    assert abs(np.mean(np.abs(pit - 0.5) <= 0.4) - 0.8) < 0.1
    assert law.n_eff(Xt[:20]).min() >= 1


def test_weighted_crps_against_brute_force():
    rng = np.random.default_rng(0)
    values = np.sort(rng.normal(size=(5, 40)), axis=1)
    w = rng.dirichlet(np.ones(40), size=5)
    obs = rng.normal(size=5)
    expected = [(w[i] * np.abs(values[i] - obs[i])).sum()
                - 0.5 * (w[i][:, None] * w[i][None, :] * np.abs(values[i][:, None] - values[i][None, :])).sum()
                for i in range(5)]
    np.testing.assert_allclose(weighted_crps(values, w, obs), expected, atol=1e-10)


def test_support_checker_flags_far_states():
    X, _ = _law_sample()
    sup = SupportChecker(20, 0.99).fit(X)
    assert sup.in_support(X).mean() > 0.97
    far = X[:1].copy()
    far[0, :4] = np.exp(0.0)
    assert not sup.in_support(far)[0]


def test_block_placebo_keeps_residual_order():
    X, V = _law_sample(1000)
    law = LocationScaleLaw(50).fit(X, V)
    rng = np.random.default_rng(0)
    path = law.resample_path(X[:44], rng, block=22)
    eps = law.standardized(X[:44], path)
    series = law.eps_series_
    start = int(np.argmin(np.abs(series[:-21] - eps[0])))
    np.testing.assert_allclose(series[start : start + 22], eps[:22], atol=1e-8)


def test_placebo_is_built_from_training_only(synth, cfg):
    """Changing V after the fit date must not change a fold's placebo generator or draws."""
    import dataclasses
    folds = make_folds(synth, cfg, cfg["walk_forward"])
    fold, oos = folds[2], oos_positions(folds)
    V = synth.V.copy()
    V[synth.dates >= fold.fit_date] *= 3.0
    other = dataclasses.replace(synth, V=V)
    def draw(d):
        law = LocationScaleLaw(50).fit(d.X[fold.train_idx], d.V[fold.train_idx])
        return placebo_path(d, fold, law, oos, "cond_block", np.random.default_rng(5), cfg)
    np.testing.assert_array_equal(draw(synth), draw(other))
    displaced = displaced_path(synth, 63)
    assert displaced[100] == synth.V[37]
    assert np.isnan(displaced[10])


def test_select_pairs_rules():
    Z = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [0.05, 0.05]])
    eps = np.array([0.0, 2.0, 2.0, 0.1])
    allowed = np.ones((4, 4), bool)
    np.fill_diagonal(allowed, False)
    chosen = select_pairs(Z, eps, Z, eps, allowed, caliper=1.0, delta=1.0, max_partners=5)
    pairs = {(a, p) for a, p, _ in chosen}
    assert (0, 1) in pairs and (1, 0) in pairs and (3, 1) in pairs
    assert (0, 3) not in pairs                 # close history, same surprise
    assert (0, 2) not in pairs                 # different surprise, distant history


def _toy_pairs():
    return pd.DataFrame({"hi": [0, 1, 2], "lo": [3, 3, 4], "dist": 0.5, "deps": 1.0,
                         "y_hi": [2.0, 3.0, 1.0], "y_lo": [1.0, 1.0, 2.0],
                         "f0_hi": [1.0, 1.0, 1.0], "f0_lo": [1.0, 1.0, 1.0],
                         "f1_hi": [1.5, 2.0, 1.5], "f1_lo": [1.0, 1.0, 1.0]})


def test_pair_stats_values_and_dependence():
    ps = PairStats(_toy_pairs(), np.arange(5))
    s = ps.compute()
    assert np.isclose(s["pairs:conc1"], 2 / 3)
    assert np.isclose(s["pairs:conc0"], 0.5)           # f0 ties count one half
    assert np.isclose(s["pairs:mean_dY"], (1 + 2 - 1) / 3)
    # Zero weight on date 3 removes both pairs that share it.
    w = np.array([1.0, 1.0, 1.0, 0.0, 1.0])
    assert np.isclose(ps.compute(w)["pairs:mean_dY"], -1.0)


def test_response_stats_recover_linear_slope(synth, cfg):
    folds = make_folds(synth, cfg, cfg["walk_forward"])
    oos = oos_positions(folds)
    results = []
    for fold in folds[:12]:
        tr = fold.train_idx
        m0, m1 = fit_pair("har", synth.X[tr], synth.V[tr], synth.y[tr], {}, {}, cfg, 0)
        results.append(evaluate_fold(synth, fold, m0, m1, cfg, oos[oos <= folds[11].test_idx[-1]],
                                     parts=("origin", "response"), with_kernel=False))
        last = (m0, m1, fold)
    frame = pd.concat([r["response"] for r in results], ignore_index=True)
    oos12 = oos[oos <= folds[11].test_idx[-1]]
    stat = ResponseStats(frame, oos12, cfg)
    res = stat.compute()
    assert res["rv_m:n_valid_bins"] >= 3
    # A linear rule has the same local effect per unit in both LE columns.
    m0, _, _ = last
    g = frame[(frame.feature == "rv_m") & frame.keep & (frame.fold == 11)]
    np.testing.assert_allclose(g["le0"] / g["width"], m0.coef_raw()["rv_m"], rtol=1e-8)
    assert stat.curves()["function"].nunique() == 3


def test_full_fold_evaluation_and_stitching(synth, cfg):
    folds = make_folds(synth, cfg, cfg["walk_forward"])[:3]
    oos = oos_positions(make_folds(synth, cfg, cfg["walk_forward"]))
    results = []
    for fold in folds:
        tr = fold.train_idx
        m0, m1 = fit_pair("har", synth.X[tr], synth.V[tr], synth.y[tr], {}, {}, cfg, 0)
        results.append(evaluate_fold(synth, fold, m0, m1, cfg, oos))
    origin = results[0]["origin"]
    np.testing.assert_allclose(origin["HFS_a"] + origin["ISD_a"], origin["D"], atol=1e-12)
    np.testing.assert_allclose(origin["HFS_b"] + origin["ISD_b"], origin["D"], atol=1e-12)
    stitched = stitched_stats(results, oos, cfg)
    for key in ("rms_D", "rms_HFS", "rms_ISD", "calib_slope_rel", "rv_m:T", "rv_neg_w:dS"):
        assert np.isfinite(stitched[key])
    for r in results:
        if not r["pairs"].empty:
            p = r["pairs"]
            assert (p["dist"] <= p["caliper"]).all() and (p["deps"] >= p["delta"]).all()
            hi_end = synth.label_end[p["hi"]]
            lo_end = synth.label_end[p["lo"]]
            hi_d, lo_d = synth.dates[p["hi"]], synth.dates[p["lo"]]
            assert np.all((hi_d > lo_end) | (lo_d > hi_end))     # target windows are disjoint


def test_origin_stats_under_weights_match_duplicates(synth, cfg):
    folds = make_folds(synth, cfg, cfg["walk_forward"])
    fold = folds[0]
    oos = fold.test_idx
    tr = fold.train_idx
    m0, m1 = fit_pair("har", synth.X[tr], synth.V[tr], synth.y[tr], {}, {}, cfg, 0)
    frame = evaluate_fold(synth, fold, m0, m1, cfg, oos, parts=("origin",), with_kernel=False)["origin"]
    stat = OriginStats(frame, oos)
    w = np.ones(len(oos))
    w[0] = 3.0
    dup = pd.concat([frame, frame.iloc[[0, 0]]], ignore_index=True)
    a, b = stat.compute(w), OriginStats(dup, oos).compute()
    assert np.isclose(a["rms_D"], b["rms_D"])
    assert np.isclose(a["dqlike"], b["dqlike"])


def test_placebo_signal_has_no_outcome_validation(cfg):
    """With V unrelated to Y given X, the matched slope and calibration slope stay near zero."""
    from conftest import make_synth_data
    data = make_synth_data(seed=11, informative=False)
    rng = np.random.default_rng(3)
    folds = make_folds(data, cfg, cfg["walk_forward"])
    oos = oos_positions(folds)
    results = []
    for fold in folds[:12]:
        tr = fold.train_idx
        law = LocationScaleLaw(100).fit(data.X[tr], data.V[tr])
        path = placebo_path(data, fold, law, oos, "cond_block", rng, cfg)
        m0 = HARModel("qlike").fit(data.X[tr], None, data.y[tr], floor=1e-6)
        m1 = HARModel("qlike").fit(data.X[tr], path[tr], data.y[tr], floor=1e-6)
        results.append(evaluate_fold(data, fold, m0, m1, cfg, oos, V_eval=path, parts=("origin",), with_kernel=False))
    frame = pd.concat([r["origin"] for r in results], ignore_index=True)
    assert frame["D"].abs().mean() < 0.3 * frame["f0"].mean()


def test_reliance_and_shapley_efficiency(synth, cfg):
    fold = make_folds(synth, cfg, cfg["walk_forward"])[0]
    tr = fold.train_idx
    m0, m1 = fit_pair("har", synth.X[tr], synth.V[tr], synth.y[tr], {}, {}, cfg, 0)
    rel = conditional_reliance(synth, fold, m0, m1, cfg, np.random.default_rng(0))
    assert set(rel["group"]) == {"persistence", "downside", "returns", "iv"}
    assert rel.loc[rel.group == "iv", "function"].eq("f1").all()
    positions = fold.test_idx[:5]
    for model, iv, tag in ((m0, False, "f0"), (m1, True, "f1")):
        phi = conditional_group_shapley(synth, fold, model, positions, cfg, iv, tag)
        total = phi.groupby("pos").agg(phi=("phi", "sum"), pred=("pred", "first"), base=("baseline", "first"))
        np.testing.assert_allclose(total["phi"] + total["base"], total["pred"], rtol=1e-8)


def test_newey_west_matches_statsmodels():
    x = np.random.default_rng(0).standard_normal(500).cumsum() * 0.01 + np.random.default_rng(1).standard_normal(500)
    res = sm.OLS(x, np.ones(500)).fit(cov_type="HAC", cov_kwds={"maxlags": 10, "use_correction": False})
    assert np.isclose(newey_west_lrv(x, 10) / 500, res.bse[0] ** 2, rtol=1e-8)


def test_hac_slope_matches_statsmodels():
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal(400), rng.standard_normal(400)
    res = sm.OLS(y, x[:, None]).fit(cov_type="HAC", cov_kwds={"maxlags": 5, "use_correction": False})
    out = hac_slope(y, x, 5)
    assert np.isclose(out["slope"], res.params[0])
    assert np.isclose(out["se"], res.bse[0], rtol=1e-8)


def test_dm_size_and_gw():
    rng = np.random.default_rng(0)
    rejections = np.mean([dm_test(rng.standard_normal(300), horizon=1, lag=3)["p"] < 0.05 for _ in range(400)])
    assert rejections < 0.09
    d = rng.standard_normal(500)
    H = np.column_stack([np.ones(500), rng.standard_normal(500)])
    assert 0 <= gw_test(d, H, 5)["p"] <= 1


def test_mbb_and_multiple_testing():
    rng = np.random.default_rng(0)
    assert len(mbb_indices(100, 7, rng)) == 100
    assert mbb_weights(100, 7, rng).sum() == 100
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert placebo_p(5.0, [1, 2, 3, 6]) == 2 / 5


def test_placebo_and_observed_share_support_states(synth, cfg):
    """Observed and placebo summaries must average over the same origins (P1 comparability)."""
    from ivbehav.core import origin_rows
    folds = make_folds(synth, cfg, cfg["walk_forward"])
    fold, oos = folds[0], oos_positions(folds)
    tr = fold.train_idx
    laws = fit_laws(synth.X[tr], synth.V[tr], cfg)
    m0, m1 = fit_pair("har", synth.X[tr], synth.V[tr], synth.y[tr], {}, {}, cfg, 0)
    observed = origin_rows(synth, fold, m0, m1, laws, cfg)
    path = placebo_path(synth, fold, laws[0], oos, "cond_block", np.random.default_rng(0), cfg)
    mp = HARModel("qlike").fit(synth.X[tr], path[tr], synth.y[tr], floor=m0.floor)
    placebo = origin_rows(synth, fold, m0, mp, laws, cfg, V_eval=path, project_b=False)
    np.testing.assert_array_equal(observed["sup_x"], placebo["sup_x"])
    assert "HFS_b" in observed and "HFS_b" not in placebo


def test_fixed_bandwidth_skips_search(synth, cfg):
    fold = make_folds(synth, cfg, cfg["walk_forward"])[0]
    tr = fold.train_idx
    law = KernelLaw(cfg["conditional"]["kernel_bandwidths"]).fit(synth.X[tr], synth.V[tr], bandwidth=0.7)
    assert law.h_ == 0.7 and not hasattr(law, "holdout_crps_")


def test_match_rules_are_training_only_and_fixed_for_refits(synth, cfg):
    from ivbehav.core import match_rules
    fold = make_folds(synth, cfg, cfg["walk_forward"])[0]
    tr = fold.train_idx
    laws = fit_laws(synth.X[tr], synth.V[tr], cfg)
    a = match_rules(synth.X[tr], laws[0], laws[2], cfg)
    assert a["caliper"] > 0 and a["delta"] > 0
    assert match_rules(synth.X[tr], laws[0], laws[2], cfg, k_caliper=5)["caliper"] < a["caliper"]


def test_slopes_are_undefined_when_the_signal_is_unused():
    y = np.random.default_rng(0).standard_normal(200)
    assert np.isnan(hac_slope(y, np.full(200, 1e-14), 5)["slope"])
    from ivbehav.stats import _wslope
    assert np.isnan(_wslope(y, np.full(200, 1e-14), np.ones(200)))
    assert np.isfinite(_wslope(y, np.random.default_rng(1).standard_normal(200), np.ones(200)))
