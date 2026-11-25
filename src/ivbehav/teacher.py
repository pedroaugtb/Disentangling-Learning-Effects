"""Teacher / input decomposition of the forecast gain from adding V (PREANALYSIS, Addendum C).

With f0(x), f1(x, v) and the projection P1(x) = E[f1(x, V) | x] under a conditional law of V:

    L(f0) - L(f1)  =  [L(f0) - L(P1)]  +  [L(P1) - L(f1)]
     gain_total        gain_teacher         gain_input

P1 is a function of the history alone, so ``gain_teacher`` is what fitting with V taught the
learner about the history, usable without observing V at forecast time.  ``gain_input`` is the
value of observing the V surprise in real time.  Gains are positive when they favour the
second forecast.  The identity is exact, row by row, for any loss.

Everything here works on the per-origin frames written by ``core.origin_rows`` (columns pos,
date, y, f0, f1, P1_a, sup_x and optionally P1_b, sup_b), so the observed run, placebo refits
and refit bootstraps share one code path, as the rest of the study does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .inference import dm_test
from .models import qlike, se

PREFIX = "ti:"
EX_COVID = ("2020-02-15", "2020-04-30")


def _wmean(values, w):
    total = w.sum()
    return float((values * w).sum() / total) if total > 0 else float("nan")


def realtime_scale(dates, label_end, y, f, min_closed: int) -> np.ndarray:
    """QLIKE-optimal multiplicative level correction known at each origin.

    For a scalar c, mean QLIKE(y, c f) is minimised at c = mean(y / f).  At origin t the mean
    runs over earlier out-of-sample origins whose target window had closed strictly before t,
    so the correction uses no outcome unknown at t.  NaN until ``min_closed`` labels exist.
    """
    dates = np.asarray(dates, "datetime64[ns]")
    label_end = np.asarray(label_end, "datetime64[ns]")
    ratio = np.asarray(y, float) / np.asarray(f, float)
    order = np.argsort(label_end, kind="stable")
    ends = label_end[order]
    csum = np.cumsum(ratio[order])
    k = np.searchsorted(ends, dates, side="left")         # labels with end < date
    out = np.full(len(dates), np.nan)
    ok = k >= max(min_closed, 1)
    out[ok] = csum[k[ok] - 1] / k[ok]
    return out


class TeacherStats:
    """Gain decomposition under date weights (unit weights give the point estimate).

    ``label_end`` enables the real-time level-recalibrated sensitivity; ``ex_covid`` drops the
    PREANALYSIS exclusion window as another sensitivity.  Law B quantities need ``P1_b``.
    """

    def __init__(self, frame: pd.DataFrame, oos_pos: np.ndarray, label_end=None,
                 min_closed: int = 250, ex_covid: tuple[str, str] | None = EX_COVID):
        f = frame.reset_index(drop=True)
        where = {p: i for i, p in enumerate(oos_pos)}
        self.date_i = np.array([where[p] for p in f["pos"]])
        y = f["y"].to_numpy(float)
        F = {"0": f["f0"].to_numpy(float), "P": f["P1_a"].to_numpy(float), "1": f["f1"].to_numpy(float)}
        self.q = {k: qlike(y, v) for k, v in F.items()}
        self.s = {k: se(y, v) for k, v in F.items()}
        self.sup_x = f["sup_x"].to_numpy(bool) if "sup_x" in f else np.ones(len(f), bool)
        dates = pd.DatetimeIndex(f["date"])
        self.ex_covid = (np.ones(len(f), bool) if ex_covid is None
                         else ~((dates >= pd.Timestamp(ex_covid[0])) & (dates <= pd.Timestamp(ex_covid[1]))))
        self.has_b = "P1_b" in f and "sup_b" in f and f["P1_b"].notna().any()
        if self.has_b:
            self.sup_b = f["sup_b"].to_numpy(bool) & np.isfinite(f["P1_b"].to_numpy(float))
            self.qPb = np.where(self.sup_b, qlike(y, np.where(self.sup_b, f["P1_b"].to_numpy(float), 1.0)), 0.0)
        self.has_recal = label_end is not None
        if self.has_recal:
            scales = {k: realtime_scale(dates, label_end, y, v, min_closed) for k, v in F.items()}
            self.recal = np.all([np.isfinite(c) for c in scales.values()], axis=0)
            self.qr = {k: np.where(self.recal, qlike(y, np.where(self.recal, scales[k] * F[k], 1.0)), 0.0)
                       for k in F}

    @staticmethod
    def _gains(L0, LP, L1, tag=""):
        return {f"{PREFIX}gain_total{tag}": L0 - L1, f"{PREFIX}gain_teacher{tag}": L0 - LP,
                f"{PREFIX}gain_input{tag}": LP - L1}

    def compute(self, w_dates=None) -> dict:
        n = len(self.date_i)
        w = np.ones(n) if w_dates is None else np.asarray(w_dates, float)[self.date_i]
        q, s = self.q, self.s
        L0, LP, L1 = (_wmean(q[k], w) for k in ("0", "P", "1"))
        out = {f"{PREFIX}L0": L0, f"{PREFIX}LP": LP, f"{PREFIX}L1": L1, **self._gains(L0, LP, L1)}
        # Relative to L(f0): the scale of the non-superiority margin.
        out[f"{PREFIX}rel_gain_teacher"] = (L0 - LP) / L0
        out[f"{PREFIX}rel_gain_input"] = (LP - L1) / L0
        out[f"{PREFIX}rel_gain_total"] = (L0 - L1) / L0
        out.update(self._gains(*(_wmean(s[k], w) for k in ("0", "P", "1")), tag="_mse"))
        wx = w * self.sup_x
        out.update(self._gains(*(_wmean(q[k], wx) for k in ("0", "P", "1")), tag="_supx"))
        wc = w * self.ex_covid
        out.update(self._gains(*(_wmean(q[k], wc) for k in ("0", "P", "1")), tag="_excovid"))
        if self.has_b:
            wb = w * self.sup_b
            L0b, L1b = _wmean(q["0"], wb), _wmean(q["1"], wb)
            out.update(self._gains(L0b, _wmean(self.qPb, wb), L1b, tag="_lawB"))
            out.update(self._gains(L0b, _wmean(q["P"], wb), L1b, tag="_lawA_on_supB"))
            out[f"{PREFIX}share_sup_b"] = _wmean(self.sup_b.astype(float), w)
        if self.has_recal:
            wr = w * self.recal
            out.update(self._gains(*(_wmean(self.qr[k], wr) for k in ("0", "P", "1")), tag="_recal"))
            # The raw gains on the same (later) origins, so recalibration is not confused with sample.
            out.update(self._gains(*(_wmean(q[k], wr) for k in ("0", "P", "1")), tag="_raw_on_recal"))
            out[f"{PREFIX}share_recal"] = _wmean(self.recal.astype(float), w)
        return out

    def point(self) -> dict:
        return self.compute()


# ------------------------------------------------------------------------------------
# Descriptive tables (no claims)
# ------------------------------------------------------------------------------------
def _decorate(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    f["q0"], f["qP"], f["q1"] = qlike(f.y, f.f0), qlike(f.y, f.P1_a), qlike(f.y, f.f1)
    f["teacher"], f["input"], f["total"] = f.q0 - f.qP, f.qP - f.q1, f.q0 - f.q1
    return f


def _agg(g: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    return g.agg(n=("y", "size"), L0=("q0", "mean"), LP=("qP", "mean"), L1=("q1", "mean"),
                 gain_teacher=("teacher", "mean"), gain_input=("input", "mean"), gain_total=("total", "mean"),
                 mean_rel_HFS=("rel_HFS", "mean"), mean_rel_ISD=("rel_ISD", "mean")).reset_index()


def state_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Gains by historical variance rank and IV-surprise tercile (training-law PIT).

    Cuts come from training quantities only, so no OOS information sets them.  This is the
    descriptive counterpart of the Giacomini-White result: gains that differ in sign across
    states can cancel on average.
    """
    f = _decorate(frame)
    f["rel_HFS"], f["rel_ISD"] = (f.P1_a - f.f0) / f.f0, (f.f1 - f.P1_a) / f.f0
    f["rv_state"] = pd.cut(f["rv_rank"], [-0.001, 1 / 3, 2 / 3, 1.0001], labels=["rv_low", "rv_mid", "rv_high"])
    f["surprise"] = pd.cut(f["pit_a"], [-0.001, 1 / 3, 2 / 3, 1.0001], labels=["iv_low", "iv_mid", "iv_high"])
    parts = [_agg(f.groupby(["rv_state", "surprise"], observed=True)),
             _agg(f.groupby(["rv_state"], observed=True)).assign(surprise="all"),
             _agg(f.groupby(["surprise"], observed=True)).assign(rv_state="all")]
    return pd.concat(parts, ignore_index=True)


def year_table(frame: pd.DataFrame) -> pd.DataFrame:
    f = _decorate(frame)
    f["rel_HFS"], f["rel_ISD"] = (f.P1_a - f.f0) / f.f0, (f.f1 - f.P1_a) / f.f0
    f["year"] = pd.DatetimeIndex(f["date"]).year
    return _agg(f.groupby("year"))


def dm_table(frame: pd.DataFrame, horizon: int, lags=(21, 42, 63)) -> pd.DataFrame:
    """HAC Diebold-Mariano on the two loss differentials (validity checks, as in 03)."""
    f = _decorate(frame.sort_values("pos"))
    rows = []
    for name in ("teacher", "input", "total"):
        d = f[name].to_numpy()
        for lag in lags:
            # dm_test takes L_a - L_b; a positive gain is a positive mean here.
            t = dm_test(d, horizon=horizon, lag=lag)
            rows.append({"component": name, "lag": lag, "mean_gain": t["mean_diff"], "dm": t["dm"], "dm_p": t["p"]})
    return pd.DataFrame(rows)


def horizon_of(target: str) -> int:
    """Forecast horizon in sessions for the HLN correction (Y30 spans about 22 sessions)."""
    return {"Y30": 22, "Y5td": 5, "Y10td": 10}.get(target, 22)
