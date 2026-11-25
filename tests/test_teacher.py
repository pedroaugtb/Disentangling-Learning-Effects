"""Addendum C: teacher / input decomposition.  Synthetic data only."""
import numpy as np
import pandas as pd

from ivbehav.conditional import project
from ivbehav.core import evaluate_fold
from ivbehav.folds import make_folds, oos_positions
from ivbehav.models import HARModel, _HAR_IDX, fit_pair, qlike
from ivbehav.teacher import TeacherStats, dm_table, realtime_scale, state_table, year_table
from ivbehav.teacher_sim import summarize, teacher_replicate


def _origin(synth, cfg, n_folds=6, family="har"):
    folds = make_folds(synth, cfg, cfg["walk_forward"])[:n_folds]
    oos = oos_positions(folds)
    frames = []
    for fold in folds:
        tr = fold.train_idx
        m0, m1 = fit_pair(family, synth.X[tr], synth.V[tr], synth.y[tr], {}, {}, cfg, 0)
        frames.append(evaluate_fold(synth, fold, m0, m1, cfg, oos, parts=("origin",))["origin"])
    return pd.concat(frames, ignore_index=True), oos


def test_gains_add_up_under_any_weights(synth, cfg):
    origin, oos = _origin(synth, cfg)
    stat = TeacherStats(origin, oos, label_end=synth.label_end[origin["pos"].to_numpy()], min_closed=20)
    rng = np.random.default_rng(1)
    for w in (None, rng.poisson(1.0, len(oos)).astype(float)):
        s = stat.compute(w)
        for tag in ("", "_mse", "_supx", "_excovid", "_lawB", "_recal", "_raw_on_recal"):
            assert np.isclose(s[f"ti:gain_total{tag}"], s[f"ti:gain_teacher{tag}"] + s[f"ti:gain_input{tag}"],
                              rtol=1e-10, atol=1e-14), tag
    s = stat.point()
    q0, q1 = qlike(origin.y, origin.f0).mean(), qlike(origin.y, origin.f1).mean()
    assert np.isclose(s["ti:gain_total"], q0 - q1)
    assert np.isclose(s["ti:rel_gain_input"], s["ti:gain_input"] / s["ti:L0"])


def test_weights_match_duplicated_dates(synth, cfg):
    origin, oos = _origin(synth, cfg, n_folds=3)
    stat = TeacherStats(origin, oos)
    w = np.zeros(len(oos))
    w[:40] = 1.0
    w[5] = 3.0
    dup = pd.concat([origin.iloc[:40], origin.iloc[[5, 5]]], ignore_index=True)
    a = stat.compute(w)
    b = TeacherStats(dup, oos).compute()
    for key in ("ti:gain_teacher", "ti:gain_input", "ti:L0"):
        assert np.isclose(a[key], b[key])


def test_ols_teacher_gain_is_zero_under_the_linear_law(synth, cfg):
    """OLS f0, f1 on the same rows, projecting V with its OLS fit on the same regressors: P1 = f0
    at every x (omitted-variable identity), hence a teacher gain of exactly zero under any loss.
    A teacher gain of the linear HAR can only come from its QLIKE weights or from the law."""
    fold = make_folds(synth, cfg, cfg["walk_forward"])[0]
    tr, te = fold.train_idx, fold.test_idx
    X, V, y = synth.X[tr], synth.V[tr], synth.y[tr]
    m0 = HARModel("mse").fit(X, None, y, floor=1e-12)
    m1 = HARModel("mse").fit(X, V, y, floor=1e-12)
    gamma = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X[:, _HAR_IDX]]), V, rcond=None)[0]
    Xq = synth.X[te]
    gq = np.column_stack([np.ones(len(Xq)), Xq[:, _HAR_IDX]]) @ gamma
    # Linear in V: integrating over any law with mean gq gives f1(x, gq).
    spread = np.array([-1.0, 0.0, 1.0]) * 1e-4
    p1 = project(m1, Xq, gq[:, None] + spread[None, :], np.full((len(Xq), 3), 1 / 3))
    f0 = m0.predict(Xq)
    np.testing.assert_allclose(p1, f0, rtol=1e-8)
    yq = synth.y[te]
    assert abs(np.mean(qlike(yq, f0) - qlike(yq, p1))) < 1e-10


def test_realtime_scale_uses_only_closed_labels():
    dates = pd.bdate_range("2017-01-02", periods=400)
    label_end = dates + pd.Timedelta(days=30)
    rng = np.random.default_rng(0)
    y = rng.gamma(5.0, 0.2, 400)
    f = np.ones(400)
    c = realtime_scale(dates, label_end, y, f, min_closed=50)
    # Changing outcomes whose window has not closed by origin t leaves c_t unchanged.
    t = 300
    y2 = y.copy()
    y2[np.asarray(label_end >= dates[t])] *= 10.0
    c2 = realtime_scale(dates, label_end, y2, f, min_closed=50)
    assert np.isclose(c[t], c2[t])
    closed = np.asarray(label_end < dates[t])
    assert np.isclose(c[t], y[closed].mean())
    assert np.isnan(c[10])


def test_descriptive_tables_and_dm(synth, cfg):
    origin, _ = _origin(synth, cfg, n_folds=3)
    st = state_table(origin)
    assert {"rv_state", "surprise", "gain_teacher", "gain_input"} <= set(st.columns)
    full = st[(st.rv_state == "all")]
    assert full["n"].sum() == len(origin)
    yt = year_table(origin)
    assert np.isclose((yt["gain_total"] * yt["n"]).sum() / yt["n"].sum(),
                      np.mean(qlike(origin.y, origin.f0) - qlike(origin.y, origin.f1)))
    dm = dm_table(origin, horizon=22)
    assert set(dm["component"]) == {"teacher", "input", "total"}


def test_teacher_simulation_replicate_smoke(cal, cfg):
    c = {**cfg, "simulation": {**cfg["simulation"], "n_train": 400, "n_test": 300, "bootstrap_n": 20,
                               "xgb": {**cfg["simulation"]["xgb"], "rounds": 20}}}
    tsim = {"n_pop": 1500, "pop_grid": 20, "placebo_draws": 2, "placebo_learners": ["har", "xgb"]}
    rows = teacher_replicate(0, "redundant", 0.0, cal, c, tsim, 1, True)
    df = pd.DataFrame(rows)
    assert set(df["law"]) == {"true", "A_wellspec"}
    assert np.allclose(df["teacher"] + df["input"], df["total"])
    assert df.loc[(df.learner == "har") & (df.law == "A_wellspec"), "placebo_p_input"].notna().all()
    # Redundant design: V carries no information, so the ideal input gain is zero.
    assert abs(df["ideal_input"].iloc[0]) < 1e-12
    s = summarize(df)
    assert {"detect_C1", "bias_teacher", "input_covers_pop"} <= set(s.columns)
