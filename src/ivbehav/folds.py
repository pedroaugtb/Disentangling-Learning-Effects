"""Walk-forward folds with label-observability purging (proposal, Sec. 6).

A fold fits on the first session tau of a block and forecasts origins in [tau, next tau).
Row t may train a model at tau only if its complete target window had closed by tau, that
is label_end[t] <= tau.  The nested validation block is purged in the same way: an inner
training label must close before the first validation origin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import SpecData


@dataclass(frozen=True)
class Fold:
    k: int
    fit_date: pd.Timestamp
    test_idx: np.ndarray
    train_idx: np.ndarray
    inner_train_idx: np.ndarray
    val_idx: np.ndarray
    retune: bool


def _block_starts(dates: pd.DatetimeIndex, refit: str) -> np.ndarray:
    if refit == "month":
        key = dates.year * 100 + dates.month
    elif refit == "quarter":
        key = dates.year * 10 + dates.quarter
    else:
        raise ValueError(f"unknown refit frequency {refit!r}")
    return np.flatnonzero(np.r_[True, key[1:] != key[:-1]])


def make_folds(data: SpecData, cfg: dict, wf: dict) -> list[Fold]:
    """Folds for one spec.  ``wf`` is the resolved walk-forward block (see resolve_spec)."""
    dates = data.dates
    usable = data.usable
    labeled = usable & np.isfinite(data.y) & (data.y > 0)
    oos_start = pd.Timestamp(cfg["sample"]["oos_start"])

    oos_pos = np.flatnonzero(labeled & (dates >= oos_start))
    if len(oos_pos) == 0:
        raise ValueError("no labeled out-of-sample origins")
    oos_dates = dates[oos_pos]
    starts = _block_starts(oos_dates, wf["refit"])
    ends = np.r_[starts[1:], len(oos_pos)]

    folds = []
    for k, (a, b) in enumerate(zip(starts, ends)):
        test_idx = oos_pos[a:b]
        tau = dates[test_idx[0]]
        # NaT label ends compare False, so unlabeled rows drop out here as well.
        eligible = labeled & np.asarray(data.label_end <= tau) & (dates < tau)
        if wf["window"] == "rolling":
            eligible &= dates >= tau - pd.DateOffset(years=wf["rolling_years"])
        elif wf["window"] != "expanding":
            raise ValueError(f"unknown window {wf['window']!r}")
        train_idx = np.flatnonzero(eligible)

        val_start = tau - pd.DateOffset(years=wf["val_years"])
        val_idx = train_idx[dates[train_idx] >= val_start]
        inner_mask = np.asarray(data.label_end[train_idx] < dates[val_idx[0]])
        inner_train_idx = train_idx[inner_mask]
        folds.append(Fold(k=k, fit_date=tau, test_idx=test_idx, train_idx=train_idx,
                          inner_train_idx=inner_train_idx, val_idx=val_idx,
                          retune=k % wf["retune_every"] == 0))
    return folds


def oos_positions(folds: list[Fold]) -> np.ndarray:
    return np.concatenate([f.test_idx for f in folds])
