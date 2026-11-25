"""Labels, predetermined predictors, fold purging and end-to-end absence of look-ahead."""
import dataclasses

import numpy as np
import pandas as pd

from ivbehav.data import calendar_target, history_features, trading_target
from ivbehav.folds import make_folds, oos_positions
from ivbehav.models import fit_pair, make_model, validation_table


def test_calendar_target_matches_brute_force():
    dates = pd.bdate_range("2020-01-01", "2020-06-30").delete([5, 40, 41])   # holidays
    r = np.random.default_rng(1).standard_normal(len(dates)) * 0.01
    y, n_obs, ends = calendar_target(dates, r, 30)
    for i, t in enumerate(dates):
        end = t + pd.Timedelta(days=30)
        mask = (dates > t) & (dates <= end)
        if end <= dates[-1]:
            assert np.isclose(y[i], np.sum(r[mask] ** 2))
            assert n_obs[i] == mask.sum()
            assert ends[i] == end
        else:
            assert np.isnan(y[i])


def test_label_starts_at_next_session():
    dates = pd.bdate_range("2020-01-01", periods=60)
    r = np.zeros(60)
    r[10] = 0.5
    y, _, _ = calendar_target(dates, r, 30)
    assert y[10] == 0.0            # the origin's own return is predetermined, not a target
    assert np.isclose(y[9], 0.25)


def test_trading_target():
    dates = pd.bdate_range("2020-01-01", periods=30)
    r = np.random.default_rng(2).standard_normal(30) * 0.01
    y, _, ends = trading_target(dates, r, 5)
    assert np.isclose(y[3], np.sum(r[4:9] ** 2))
    assert ends[3] == dates[8]
    assert np.isnan(y[-1]) and pd.isna(ends[-1])


def test_features_are_predetermined():
    dates = pd.bdate_range("2020-01-01", periods=120)
    logc = np.cumsum(np.random.default_rng(3).standard_normal(120) * 0.01)
    base = history_features(pd.Series(np.exp(logc), index=dates))
    moved = logc.copy()
    moved[80:] += 0.5
    changed = history_features(pd.Series(np.exp(moved), index=dates))
    pd.testing.assert_frame_equal(base.iloc[:80], changed.iloc[:80])


def test_feature_definitions():
    dates = pd.bdate_range("2020-01-01", periods=40)
    r = np.r_[0.0, np.tile([0.01, -0.02, 0.03, -0.01, 0.02], 8)[:39]]
    f = history_features(pd.Series(np.exp(np.cumsum(r)), index=dates))
    i = 30
    win = r[i - 4 : i + 1]
    assert np.isclose(f["rv_w"].iloc[i], np.mean(win**2))
    assert np.isclose(f["rv_neg_w"].iloc[i], np.mean(win**2 * (win < 0)))
    assert np.isclose(f["ret_5"].iloc[i], win.sum())
    assert f["dd_22"].iloc[i] <= 0


def test_fold_purging(synth, cfg):
    wf = cfg["walk_forward"]
    folds = make_folds(synth, cfg, wf)
    assert len(folds) == 23                     # Dec 2016 has no complete 30-day label
    for f in folds:
        assert np.all(synth.label_end[f.train_idx] <= f.fit_date)
        assert np.all(synth.dates[f.train_idx] < f.fit_date)
        assert np.all(synth.dates[f.test_idx] >= f.fit_date)
        assert synth.dates[f.test_idx[0]] == f.fit_date
        assert np.all(synth.label_end[f.inner_train_idx] < synth.dates[f.val_idx[0]])
        assert not set(f.inner_train_idx) & set(f.val_idx)
        assert np.all(synth.dates[f.train_idx] >= pd.Timestamp(cfg["sample"]["start"]))
    pos = oos_positions(folds)
    assert len(pos) == len(set(pos))
    assert np.all(np.diff(pos) > 0)
    assert sum(f.retune for f in folds) == 2


def test_rolling_window_and_quarterly(synth, cfg):
    wf = {**cfg["walk_forward"], "window": "rolling", "rolling_years": 3}
    for f in make_folds(synth, cfg, wf):
        assert synth.dates[f.train_idx[0]] >= f.fit_date - pd.DateOffset(years=3)
    assert len(make_folds(synth, cfg, {**cfg["walk_forward"], "refit": "quarter"})) == 8   # up to Q4 2016


def test_future_outcomes_and_v_do_not_change_a_fit(synth, cfg):
    """Corrupting every label not yet closed at the fit date must leave forecasts unchanged."""
    fold = make_folds(synth, cfg, cfg["walk_forward"])[3]
    y = synth.y.copy()
    y[np.asarray(synth.label_end > fold.fit_date)] = 1e3
    corrupt = dataclasses.replace(synth, y=y)
    for family in ("har", "enet", "xgb"):
        hp = validation_table(family, synth.X[fold.inner_train_idx], None, synth.y[fold.inner_train_idx],
                              synth.X[fold.val_idx], None, synth.y[fold.val_idx], cfg, 0)[0]["hp"]
        hp1 = validation_table(family, synth.X[fold.inner_train_idx], synth.V[fold.inner_train_idx],
                               synth.y[fold.inner_train_idx], synth.X[fold.val_idx], synth.V[fold.val_idx],
                               synth.y[fold.val_idx], cfg, 0)[0]["hp"]
        tr, te = fold.train_idx, fold.test_idx
        a0, a1 = fit_pair(family, synth.X[tr], synth.V[tr], synth.y[tr], hp, hp1, cfg, 1)
        b0, b1 = fit_pair(family, corrupt.X[tr], corrupt.V[tr], corrupt.y[tr], hp, hp1, cfg, 1)
        np.testing.assert_array_equal(a0.predict(synth.X[te]), b0.predict(synth.X[te]))
        np.testing.assert_array_equal(a1.predict(synth.X[te], synth.V[te]), b1.predict(synth.X[te], synth.V[te]))


def test_pair_is_truly_paired(synth, cfg):
    """Same rows, same floor; the augmented family sees exactly one more input."""
    fold = make_folds(synth, cfg, cfg["walk_forward"])[0]
    tr, te = fold.train_idx, fold.test_idx
    m0, m1 = fit_pair("har", synth.X[tr], synth.V[tr], synth.y[tr], {}, {}, cfg, 0)
    assert m0.floor == m1.floor > 0
    assert len(m1.beta_) == len(m0.beta_) + 1
    assert m1.predict(synth.X[te], synth.V[te]).min() >= m1.floor
    bench = make_model("ivonly", {}, cfg).fit(synth.X[tr], synth.V[tr], synth.y[tr], m0.floor)
    assert np.all(bench.predict(synth.X[te], synth.V[te]) > 0)


def test_config_stamp_refuses_mixed_results(tmp_path, monkeypatch):
    import ivbehav.config as C
    monkeypatch.setattr(C, "RESULTS_DIR", tmp_path)
    monkeypatch.delenv("IVBEHAV_ALLOW_CONFIG_CHANGE", raising=False)
    cfg = {"profile": "x", "seed": 1}
    C.check_config_stamp(cfg)
    C.check_config_stamp(cfg)
    try:
        C.check_config_stamp({"profile": "x", "seed": 2})
    except RuntimeError:
        monkeypatch.setenv("IVBEHAV_ALLOW_CONFIG_CHANGE", "1")
        C.check_config_stamp({"profile": "x", "seed": 2})
        return
    raise AssertionError("a changed config was accepted silently")
