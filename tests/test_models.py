"""Forecasting families target E[Y | .] on the level scale, under QLIKE."""
import numpy as np
import statsmodels.api as sm

from ivbehav.data import HIST_COLS
from ivbehav.folds import make_folds
from ivbehav.models import ENetModel, HARModel, LogHARModel, XGBModel, fit_gamma_identity, qlike


def _hp():
    return {"max_depth": 3, "eta": 0.05, "min_child_weight": 5, "subsample": 0.8,
            "colsample_bytree": 1.0, "lambda": 1.0, "rounds": 150}


def test_qlike_minimised_at_conditional_mean():
    y = np.random.default_rng(0).gamma(2.0, 0.5, 200_000)
    grid = np.linspace(0.6, 1.4, 81)
    assert np.isclose(grid[np.argmin([np.mean(qlike(y, q)) for q in grid])], y.mean(), atol=0.02)
    assert qlike(1.0, 1.0) == 0.0


def test_gamma_identity_matches_statsmodels():
    rng = np.random.default_rng(1)
    A = np.column_stack([np.ones(3000), rng.uniform(0, 2, 3000), rng.uniform(0, 1, 3000)])
    mu = A @ np.array([0.5, 1.0, 0.7])
    y = rng.gamma(4.0, mu / 4.0)
    ours = fit_gamma_identity(A, y, floor=1e-6)
    theirs = sm.GLM(y, A, family=sm.families.Gamma(link=sm.families.links.Identity())).fit().params
    np.testing.assert_allclose(ours, theirs, atol=1e-4)


def test_enet_small_penalty_approaches_mle_and_targets_the_mean():
    rng = np.random.default_rng(2)
    n, p = 4000, len(HIST_COLS)
    X = np.exp(rng.normal(-9, 0.5, (n, p)))
    log_mu = -5 + 0.5 * (np.log(X[:, 2]) + 9)
    y = rng.gamma(3.0, np.exp(log_mu) / 3.0)
    model = ENetModel(1e-4, 0.5, 60).fit(X, None, y, floor=1e-8)
    assert np.isclose(model.predict(X).mean(), y.mean(), rtol=0.02)
    assert np.corrcoef(np.log(model.predict(X)), log_mu)[0, 1] > 0.97
    heavy = ENetModel(1.0, 0.9, 60).fit(X, None, y, floor=1e-8)
    assert np.count_nonzero(heavy.coef_) <= np.count_nonzero(model.coef_)


def test_xgb_targets_the_level_mean_not_the_log_mean():
    """Early rounds sit near E[Y], well above exp(E[log Y]).  Overfitting noise lowers the level
    (the log-link first-order condition is mean(y/mu) = 1): validation-chosen rounds limit it,
    and the placebo fits measure whatever remains."""
    rng = np.random.default_rng(3)
    n, p = 3000, len(HIST_COLS)
    X = np.exp(rng.normal(-9, 0.5, (n, p)))
    y = rng.gamma(0.5, 2e-3, n)                # strongly skewed: E[log y] is far below log E[y]
    hp = {**_hp(), "rounds": 40}
    model = XGBModel(hp, 1).fit(X, None, y, floor=1e-9, seed=0)
    assert np.isclose(model.predict(X).mean(), y.mean(), rtol=0.06)
    assert model.predict(X).mean() > 3 * np.exp(np.log(y).mean())
    assert np.isclose(np.mean(y / model.predict(X)), 1.0, rtol=0.03)


def test_log_target_bias_is_visible():
    """Why log fits are only a sensitivity: exp(E[log Y]) understates E[Y]."""
    rng = np.random.default_rng(4)
    n, p = 4000, len(HIST_COLS)
    X = np.exp(rng.normal(-9, 0.5, (n, p)))
    y = rng.gamma(1.0, 1e-3, n)
    plain = LogHARModel(smear=False).fit(X, None, y, floor=1e-9)
    assert plain.predict(X).mean() < 0.7 * y.mean()
    smeared = LogHARModel(smear=True).fit(X, None, y, floor=1e-9)
    assert np.isclose(smeared.predict(X).mean(), y.mean(), rtol=0.05)


def test_constant_rescaling_of_v_is_vacuous(synth, cfg):
    """Annualised variance and 21/252 are constant rescalings of Eq. (13): they need no run."""
    fold = make_folds(synth, cfg, cfg["walk_forward"])[0]
    tr, te = fold.train_idx, fold.test_idx
    X, y, V = synth.X[tr], synth.y[tr], synth.V[tr]
    c = 365.0 / 30.0
    for make in (lambda: HARModel("qlike"), lambda: ENetModel(0.01, 0.5, 40), lambda: XGBModel(_hp(), 1)):
        a = make().fit(X, V, y, floor=1e-6, seed=0).predict(synth.X[te], synth.V[te])
        b = make().fit(X, V * c, y, floor=1e-6, seed=0).predict(synth.X[te], synth.V[te] * c)
        np.testing.assert_allclose(a, b, rtol=1e-5)


def test_forecasts_positive_and_v_required(synth, cfg):
    fold = make_folds(synth, cfg, cfg["walk_forward"])[0]
    tr, te = fold.train_idx, fold.test_idx
    model = HARModel("mse").fit(synth.X[tr], synth.V[tr], synth.y[tr], floor=1e-5)
    assert np.all(model.predict(synth.X[te], synth.V[te]) >= 1e-5)
    try:
        model.predict(synth.X[te])
    except ValueError:
        return
    raise AssertionError("augmented model accepted a missing V")
