"""Proposition 1, Eq. (11) and Sec. 3.4 hold numerically; also the finite-sample OLS identity."""
import numpy as np
from scipy.stats import norm

from ivbehav.conditional import project
from ivbehav.folds import make_folds
from ivbehav.models import HARModel, _HAR_IDX, qlike, se
from ivbehav.simulation import OracleModel, TrueLaw, _g, _state_b, simulate


class _Mu:
    """The true m1 as a model, for population checks."""
    uses_v, floor = True, 0.0

    def __init__(self, cal, mode, b_bar):
        self.cal, self.mode, self.b_bar = cal, mode, b_bar

    def predict(self, X, V):
        from ivbehav.conditional import LawBasis
        from ivbehav.simulation import _a_sigma
        Z = (LawBasis._log(X) - self.cal.feat_mean) / self.cal.feat_sd
        a, s = _a_sigma(Z, self.cal)
        eps = (np.log(V) - a) / s
        return np.exp(_g(Z, self.cal) + _state_b(Z, self.mode, self.b_bar, 0.8) * eps)


def _population(cal, mode="state", n=400_000, seed=0):
    return simulate(n, cal, mode, 0.25, 0.8, 5.0, 21, 0.9, np.random.default_rng(seed))


def _assert_mean_close(series, target, block=2000, n_se=4.0):
    """Mean within n_se block-based standard errors (persistent, heavy-tailed draws)."""
    means = series[: len(series) // block * block].reshape(-1, block).mean(axis=1)
    se_mean = means.std(ddof=1) / np.sqrt(len(means))
    assert se_mean < 0.15 * abs(target), "Monte Carlo too noisy for a meaningful check"
    assert abs(series.mean() - target) < n_se * se_mean


def test_tower_property_population_hfs_is_zero(cal):
    """(5)-(6): projecting m1 under the true law recovers m0, with IV fully informative."""
    p = _population(cal, n=4000)
    law = TrueLaw(cal, n_draws=4000)
    vals, wts = law.draws(p["X"])
    p1 = project(_Mu(cal, "state", 0.25), p["X"], vals, wts)
    hfs_rel = (p1 - p["m0"]) / p["m0"]
    assert np.max(np.abs(hfs_rel)) < 2e-2       # quadrature error only
    assert np.mean(np.abs(hfs_rel)) < 2e-3
    assert np.std(p["m1"] - p["m0"]) > 0        # while the information value is positive


def test_isd_moment_identities(cal):
    """(7), (8), (9) by Monte Carlo on the DGP."""
    p = _population(cal)
    isd, y, m0, m1 = p["m1"] - p["m0"], p["Y"], p["m0"], p["m1"]
    scale = np.mean(isd**2)
    assert abs(isd.mean()) / np.sqrt(scale) < 0.02
    _assert_mean_close((y - m0) * isd, scale)
    _assert_mean_close(se(y, m0) - se(y, m1), scale)


def test_qlike_gain_identity(cal):
    """(10) by Monte Carlo."""
    p = _population(cal)
    ratio = p["m1"] / p["m0"]
    gain = qlike(p["Y"], p["m0"]) - qlike(p["Y"], p["m1"])
    assert gain.mean() > 0
    _assert_mean_close(gain, np.mean(ratio - 1 - np.log(ratio)))


def test_redundant_signal_has_zero_isd(cal):
    p = _population(cal, mode="redundant", n=2000)
    np.testing.assert_allclose(p["m1"], p["m0"])


def test_y_is_gamma_with_mean_mu(cal):
    p = _population(cal, n=300_000)
    ratio = p["Y"] / p["m1"]
    assert np.isclose(ratio.mean(), 1.0, rtol=0.01)
    assert np.isclose(ratio.var(), 1 / 5.0, rtol=0.05)


def test_eq11_decomposition_is_exact(cal):
    p = simulate(1200, cal, "state", 0.25, 0.8, 5.0, 21, 0.9, np.random.default_rng(1))
    m0 = OracleModel(cal).fit(p["X"][:900], None, p["Y"][:900], floor=1e-8)
    m1 = OracleModel(cal).fit(p["X"][:900], p["V"][:900], p["Y"][:900], floor=1e-8)
    X = p["X"][900:]
    true_law = TrueLaw(cal, 400)
    vals, wts = true_law.draws(X)
    p_true = project(m1, X, vals, wts)
    # A deliberately wrong law: the true one, shifted.
    p_hat = project(m1, X, vals * 1.2, wts)
    hfs = p_hat - m0.predict(X)
    fitted_error_contrast = p_true - m0.predict(X)
    assert np.max(np.abs(hfs - (fitted_error_contrast + (p_hat - p_true)))) < 1e-12
    assert np.sqrt(np.mean((p_hat - p_true) ** 2)) > 0


def test_common_law_centering_and_orthogonality(synth, cfg):
    """Sec. 3.3: under the projection law itself, E[ISD|x]=0 and E[HFS*ISD]=0 for any fit."""
    fold = make_folds(synth, cfg, cfg["walk_forward"])[0]
    tr = fold.train_idx
    m1 = HARModel("qlike").fit(synth.X[tr], synth.V[tr], synth.y[tr], floor=1e-6)
    from ivbehav.conditional import LocationScaleLaw
    law = LocationScaleLaw(100).fit(synth.X[tr], synth.V[tr])
    X = synth.X[fold.test_idx]
    vals, wts = law.draws(X)
    p1 = project(m1, X, vals, wts)
    n, m = vals.shape
    f1 = m1.predict(np.repeat(X, m, axis=0), vals.ravel()).reshape(n, m)
    np.testing.assert_allclose(((f1 - p1[:, None]) * wts).sum(axis=1), 0.0, atol=1e-12)


def test_affine_rule_makes_isd_proportional_to_surprise(synth, cfg):
    """Sec. 3.4: the |ISD|-|u| link in a linear rule is algebra, not evidence."""
    fold = make_folds(synth, cfg, cfg["walk_forward"])[0]
    tr = fold.train_idx
    m1 = HARModel("qlike").fit(synth.X[tr], synth.V[tr], synth.y[tr], floor=1e-6)
    from ivbehav.conditional import LocationScaleLaw
    law = LocationScaleLaw(100).fit(synth.X[tr], synth.V[tr])
    X, V = synth.X[fold.test_idx], synth.V[fold.test_idx]
    vals, wts = law.draws(X)
    isd = m1.predict(X, V) - project(m1, X, vals, wts)
    b = m1.coef_raw()["V"]
    np.testing.assert_allclose(isd, b * (V - (vals * wts).sum(axis=1)), rtol=1e-8)


def test_ols_hfs_vanishes_with_the_matching_linear_law(synth, cfg):
    """Finite-sample analogue of (6), by the omitted-variable identity.

    OLS f0 and f1 on the same rows, projecting with the OLS fit of V on the same regressors:
    HFS is zero at every x, in or out of sample.  So nonzero HAR HFS can only come from the
    objective (QLIKE weights) or from the projection law, never from "new information".
    """
    fold = make_folds(synth, cfg, cfg["walk_forward"])[0]
    tr, te = fold.train_idx, fold.test_idx
    X, V, y = synth.X[tr], synth.V[tr], synth.y[tr]
    m0 = HARModel("mse").fit(X, None, y, floor=1e-12)
    m1 = HARModel("mse").fit(X, V, y, floor=1e-12)
    A = np.column_stack([np.ones(len(X)), X[:, _HAR_IDX]])
    gamma = np.linalg.lstsq(A, V, rcond=None)[0]
    Xq = synth.X[te]
    gq = np.column_stack([np.ones(len(Xq)), Xq[:, _HAR_IDX]]) @ gamma
    hfs = m1._raw(Xq, gq) - m0._raw(Xq, None)
    assert np.max(np.abs(hfs)) < 1e-10 * np.abs(m0._raw(Xq, None)).max()
    # Coefficient form: b0 = b1 + b_V * gamma, the reallocation of credit between inputs.
    b0, b1 = m0.coef_raw(), m1.coef_raw()
    for j, name in enumerate(b0):
        assert np.isclose(b0[name], b1[name] + b1["V"] * gamma[j + 1], rtol=1e-8)


def test_normal_grid_law_matches_dgp(cal):
    p = simulate(3000, cal, "constant", 0.25, 0.8, 5.0, 21, 0.9, np.random.default_rng(3))
    pit = TrueLaw(cal).pit(p["X"], p["V"])
    np.testing.assert_allclose(pit, norm.cdf(p["eps"]), atol=1e-8)
