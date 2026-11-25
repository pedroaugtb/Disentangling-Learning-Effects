"""Complementary reliance diagnostics that respect feature dependence (WP3).

* Conditional group reliance (Fisher et al., 2019): replace a group by values taken from
  training states resembling the rest of the input, and record the increase in OOS loss.
* Group Shapley with conditional value functions (Aas et al., 2021): v(S) averages the model
  over training states close to x on S, instead of independent perturbations.
"""
from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .conditional import LawBasis
from .data import HIST_COLS
from .models import qlike, se


def _whitened_subset(Z_tr, Z_q):
    cov = np.atleast_2d(np.cov(Z_tr, rowvar=False))
    w, U = np.linalg.eigh(cov)
    W = U @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-8))) @ U.T
    return Z_tr @ W, Z_q @ W


def conditional_reliance(data, fold, m0, m1, cfg, rng, k_neighbors: int = 50) -> pd.DataFrame:
    r = cfg["response"]
    te, tr = fold.test_idx, fold.train_idx
    X, V, y = data.X[te], data.V[te], data.y[te]
    X_tr, V_tr = data.X[tr], data.V[tr]
    basis = LawBasis().fit(X_tr)
    Z, Z_tr = basis.z(X), basis.z(X_tr)
    logv, logv_tr = np.log(V)[:, None], np.log(V_tr)[:, None]

    def neighbours(Q, Ref, k):
        ref_w, q_w = _whitened_subset(Ref, Q)
        return np.argsort(cdist(q_w, ref_w), axis=1)[:, :k]

    rows = []
    base = {"f0": m0.predict(X), "f1": m1.predict(X, V)}
    groups = {name: [HIST_COLS.index(c) for c in cols] for name, cols in r["reliance_groups"].items()}
    n, draws = len(te), r["reliance_draws"]

    def record(function, group, preds):
        rows.append(pd.DataFrame({
            "pos": te, "fold": fold.k, "function": function, "group": group,
            "dqlike": np.mean([qlike(y, p) for p in preds], axis=0) - qlike(y, base[function]),
            "dse": np.mean([se(y, p) for p in preds], axis=0) - se(y, base[function]),
        }))

    for group, cols in groups.items():
        rest = [j for j in range(Z.shape[1]) if j not in cols]       # summaries only, not return lags
        # f0 conditions on the remaining history; f1 conditions on it and on the observed IV.
        nn0 = neighbours(Z[:, rest], Z_tr[:, rest], k_neighbors)
        nn1 = neighbours(np.column_stack([Z[:, rest], logv]), np.column_stack([Z_tr[:, rest], logv_tr]), k_neighbors)
        preds0, preds1 = [], []
        for _ in range(draws):
            pick = rng.integers(0, k_neighbors, size=n)
            X_new = X.copy()
            X_new[:, cols] = X_tr[nn0[np.arange(n), pick]][:, cols]
            preds0.append(m0.predict(X_new))
            X_new = X.copy()
            X_new[:, cols] = X_tr[nn1[np.arange(n), pick]][:, cols]
            preds1.append(m1.predict(X_new, V))
        record("f0", group, preds0)
        record("f1", group, preds1)

    nn = neighbours(Z, Z_tr, k_neighbors)
    record("f1", "iv", [m1.predict(X, V_tr[nn[np.arange(n), rng.integers(0, k_neighbors, size=n)]])
                        for _ in range(draws)])
    return pd.concat(rows, ignore_index=True)


def _shapley_weights(n_groups):
    return {s: factorial(s) * factorial(n_groups - s - 1) / factorial(n_groups) for s in range(n_groups)}


def conditional_group_shapley(data, fold, model, positions, cfg, include_iv: bool, function: str) -> pd.DataFrame:
    """Group Shapley values per origin; efficiency holds up to the training mean baseline."""
    r = cfg["response"]
    if len(positions) == 0:
        return pd.DataFrame()
    tr = fold.train_idx
    n_hist = len(HIST_COLS)
    X_tr, V_tr = data.X[tr][:, :n_hist], data.V[tr]
    basis = LawBasis().fit(X_tr)
    Z_tr = basis.z(X_tr)
    if include_iv:
        Z_tr = np.column_stack([Z_tr, (np.log(V_tr) - np.log(V_tr).mean()) / np.log(V_tr).std()])
    names = list(r["reliance_groups"])
    groups = [[HIST_COLS.index(c) for c in r["reliance_groups"][g]] for g in names]
    if include_iv:
        names, groups = names + ["iv"], groups + [[len(HIST_COLS)]]
    p = len(groups)
    top_m = min(r["shapley_top_m"], len(tr))
    weight = _shapley_weights(p)

    X_q = data.X[positions][:, :n_hist]
    Q_all = np.column_stack([X_q, data.V[positions]]) if include_iv else X_q
    Z_q = basis.z(X_q)
    if include_iv:
        logv = np.log(V_tr)
        Z_q = np.column_stack([Z_q, (np.log(data.V[positions]) - logv.mean()) / logv.std()])
    Ref = np.column_stack([X_tr, V_tr]) if include_iv else X_tr

    def predict(F):
        return model.predict(F[:, :-1], F[:, -1]) if include_iv else model.predict(F)

    n = len(positions)
    full = predict(Q_all)
    baseline = float(np.mean(predict(Ref)))
    values = {(): np.full(n, baseline), tuple(range(p)): full}
    for size in range(1, p):
        for coalition in combinations(range(p), size):
            cols = sum((groups[g] for g in coalition), [])
            ref_w, q_w = _whitened_subset(Z_tr[:, cols], Z_q[:, cols])
            dist = cdist(q_w, ref_w)
            idx = np.argsort(dist, axis=1)[:, :top_m]
            near = np.take_along_axis(dist, idx, axis=1)
            h = np.maximum(near[:, -1:] / 2.0, 1e-8)
            w = np.exp(-0.5 * (near / h) ** 2)
            w = w / w.sum(axis=1, keepdims=True)
            filled = Ref[idx]                                # (n, top_m, p)
            filled[:, :, cols] = Q_all[:, None, cols]
            preds = predict(filled.reshape(-1, Ref.shape[1])).reshape(n, top_m)
            values[coalition] = (preds * w).sum(axis=1)

    rows = []
    for g in range(p):
        phi = np.zeros(n)
        others = [o for o in range(p) if o != g]
        for size in range(p):
            for coalition in combinations(others, size):
                with_g = tuple(sorted(coalition + (g,)))
                phi += weight[size] * (values[with_g] - values[coalition])
        rows.append(pd.DataFrame({"pos": positions, "fold": fold.k, "function": function,
                                  "group": names[g], "phi": phi, "pred": full, "baseline": baseline}))
    return pd.concat(rows, ignore_index=True)
