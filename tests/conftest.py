"""Shared synthetic fixtures.  No test reads the real panels or the test period."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ivbehav.config import load_config  # noqa: E402
from ivbehav.data import HIST_COLS, SpecData, calendar_target, history_features  # noqa: E402
from ivbehav.simulation import Calibration  # noqa: E402


def make_synth_data(seed=0, start="2008-10-01", end="2016-12-31", informative=True) -> SpecData:
    """Stochastic-volatility returns, with V a noisy view of the next month's log variance."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    h = np.empty(n)
    h[0] = -9.5
    for t in range(1, n):
        h[t] = -9.5 + 0.985 * (h[t - 1] + 9.5) + 0.12 * rng.standard_normal()
    ret = np.exp(h / 2) * rng.standard_normal(n)
    close = pd.Series(1000 * np.exp(np.cumsum(ret)), index=dates)
    feats = history_features(close)
    r = feats["ret"].to_numpy()
    y, n_obs, ends = calendar_target(dates, r, 30)
    forward_h = pd.Series(h).rolling(21).mean().shift(-21).to_numpy()
    signal = forward_h if informative else h
    V = np.exp(np.nan_to_num(signal, nan=-9.5) + np.log(21) + 0.3 + 0.2 * rng.standard_normal(n))
    V[0] = np.nan
    return SpecData(asset="SYN", spec="primary", dates=dates, X=feats[HIST_COLS].to_numpy(), V=V, y=y,
                    label_end=ends, n_obs=n_obs.astype(float),
                    in_sample=np.asarray(dates >= pd.Timestamp("2010-01-01")), ret=r,
                    log_iv_vol=0.5 * np.log(V * 365 / 30))


@pytest.fixture(scope="session")
def cfg():
    c = load_config("dev.yaml")
    c["sample"] = {"download_start": "2008-10-01", "start": "2010-01-01", "end": "2016-12-31", "oos_start": "2015-01-01"}
    return c


@pytest.fixture(scope="session")
def synth():
    return make_synth_data()


@pytest.fixture(scope="session")
def cal():
    p = len(HIST_COLS)
    corr = 0.4 * np.ones((p, p)) + 0.6 * np.eye(p)
    return Calibration(
        feat_mean=np.array([-10.5, -10.0, -9.8, -10.8, 0.0003, 0.002, -0.02]),
        feat_sd=np.array([1.6, 0.9, 0.7, 1.2, 0.01, 0.02, 0.02]),
        corr=corr, ar=np.array([0.2, 0.8, 0.97, 0.7, 0.0, 0.8, 0.9]),
        beta_a=np.r_[-5.8, 0.05, 0.1, 0.4, 0.05, -0.02, -0.05, -0.05, -0.03],
        gamma_s=np.r_[-2.5, 0.2, -0.05, -0.05], scale_s=1.0,
        g0=-5.8, g1=np.array([0.03, 0.08, 0.4, 0.03, 0.0, -0.03, -0.03]))
