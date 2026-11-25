"""One code path for every behavioral evaluation of a fitted (f0, f1) pair.

The observed run, placebo fits, refit bootstraps and multiplicity variants all go through
``evaluate_fold`` and the statistics in ``stats.py``, so they measure the same quantities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from .conditional import fit_laws, project
from .data import HIST_COLS, SpecData
from .folds import Fold


# ------------------------------------------------------------------------------------
# WP2: per-origin decomposition
# ------------------------------------------------------------------------------------
def origin_rows(data: SpecData, fold: Fold, m0, m1, laws, cfg: dict, V_eval=None,
                project_b: bool = True) -> pd.DataFrame:
    """Decomposition per origin.  ``sup_x`` is the k-NN support of the history and governs law A;
    the kernel law's own quantities also need an adequate neighbourhood (``sup_b``).
    ``project_b`` adds the second projection, which costs one more model pass."""
    law_a, law_b, support = laws
    idx = fold.test_idx
    X, y = data.X[idx], data.y[idx]
    V = (data.V if V_eval is None else V_eval)[idx]
    f0, f1 = m0.predict(X), m1.predict(X, V)

    values_a, weights_a = law_a.draws(X)
    p_a = project(m1, X, values_a, weights_a)
    g_a = (values_a * weights_a).sum(axis=1)
    pit_a = law_a.pit(X, V)
    c = cfg["conditional"]
    sup_x = support.in_support(X)
    rvm = HIST_COLS.index("rv_m")
    train_rv = np.sort(data.X[fold.train_idx, rvm])

    row = {
        "pos": idx, "date": data.dates[idx], "fold": fold.k, "y": y, "n_obs": data.n_obs[idx],
        "f0": f0, "f1": f1, "D": f1 - f0,
        "P1_a": p_a, "HFS_a": p_a - f0, "ISD_a": f1 - p_a,
        "V": V, "g_a": g_a, "u_a": V - g_a, "eps_a": law_a.standardized(X, V), "pit_a": pit_a,
        "rv_m": X[:, rvm], "rv_rank": np.searchsorted(train_rv, X[:, rvm]) / len(train_rv),
        "ret_5": X[:, HIST_COLS.index("ret_5")],
        "floored0": f0 <= m0.floor, "floored1": f1 <= m1.floor,
        "fit_floor": m0.floor,
    }
    if law_b is not None:
        values_b, weights_b = law_b.draws(X)
        neff = 1.0 / (weights_b**2).sum(axis=1)
        row.update({"neff_b": neff, "bandwidth_b": law_b.h_, "sup_b": sup_x & (neff >= c["min_neff"])})
        if project_b:
            p_b = project(m1, X, values_b, weights_b)
            row.update({"P1_b": p_b, "HFS_b": p_b - f0, "ISD_b": f1 - p_b,
                        "g_b": (values_b * weights_b).sum(axis=1),
                        "pit_b": (weights_b * (values_b <= V[:, None])).sum(axis=1)})
    row["sup_x"] = sup_x
    row["sup_xv"] = sup_x & (pit_a >= c["pit_lo"]) & (pit_a <= c["pit_hi"])
    return pd.DataFrame(row)


# ------------------------------------------------------------------------------------
# WP1 / WP3: stitched out-of-sample accumulated local effects
# ------------------------------------------------------------------------------------
def response_rows(data: SpecData, fold: Fold, m0, m1, laws, cfg: dict, features, V_eval=None,
                  with_projection: bool = True) -> pd.DataFrame:
    """Local effects at this fold's OOS states over bins of THIS fold's training quantiles.

    Bin edges are explanation references, so they come from the training window; a bin index
    therefore means a rank in the then-available history and is comparable across folds.
    The augmented rule is differenced at the observed V (f1) and through its projection (P1),
    never at independent (x, v) combinations.  States outside the training range or whose
    perturbed points leave the training support are flagged and later withheld.
    """
    law_a, _, support = laws
    r = cfg["response"]
    n_bins = r["n_bins"]
    idx = fold.test_idx
    X = data.X[idx]
    V = (data.V if V_eval is None else V_eval)[idx]
    X_tr = data.X[fold.train_idx]
    rvm = HIST_COLS.index("rv_m")
    rv_rank = np.searchsorted(np.sort(X_tr[:, rvm]), X[:, rvm]) / len(X_tr)
    rows = []
    for feat in features:
        j = HIST_COLS.index(feat)
        edges = np.quantile(X_tr[:, j], np.linspace(0.0, 1.0, n_bins + 1))
        xj = X[:, j]
        b = np.searchsorted(edges, xj, side="right") - 1
        b = np.where(xj == edges[-1], n_bins - 1, b)
        in_range = (b >= 0) & (b < n_bins)
        b_safe = np.clip(b, 0, n_bins - 1)

        X_lo, X_hi = X.copy(), X.copy()
        X_lo[:, j], X_hi[:, j] = edges[b_safe], edges[b_safe + 1]
        le0 = m0.predict(X_hi) - m0.predict(X_lo)
        le1 = m1.predict(X_hi, V) - m1.predict(X_lo, V)
        pit = law_a.pit(X, V)
        c = cfg["conditional"]
        frame = {
            "pos": idx, "fold": fold.k, "feature": feat, "bin": b_safe,
            "le0": le0, "le1": le1, "width": edges[b_safe + 1] - edges[b_safe],
            "keep": in_range & support.in_support(X_lo) & support.in_support(X_hi)
                    & (pit >= c["pit_lo"]) & (pit <= c["pit_hi"]),
            "pit_a": pit, "rv_rank": rv_rank,
        }
        if with_projection:
            hi_vals, hi_w = law_a.draws(X_hi)
            lo_vals, lo_w = law_a.draws(X_lo)
            frame["le1p"] = project(m1, X_hi, hi_vals, hi_w) - project(m1, X_lo, lo_vals, lo_w)
        rows.append(pd.DataFrame(frame))
    return pd.concat(rows, ignore_index=True)


def ale2d_rows(data: SpecData, fold: Fold, m0, m1, cfg: dict) -> pd.DataFrame:
    """Second differences for two-way ALE on training-rank cells; 'V' may be one coordinate."""
    r = cfg["response"]
    n_bins = r["ale2d_bins"]
    idx = fold.test_idx
    X, V = data.X[idx], data.V[idx]
    X_tr, V_tr = data.X[fold.train_idx], data.V[fold.train_idx]
    probs = np.linspace(0.0, 1.0, n_bins + 1)

    def coord(name):
        if name == "V":
            return V, np.quantile(V_tr, probs)
        j = HIST_COLS.index(name)
        return X[:, j], np.quantile(X_tr[:, j], probs)

    def evaluate(model, name_a, a_val, name_b, b_val):
        Xq, Vq = X.copy(), V.copy()
        for name, val in ((name_a, a_val), (name_b, b_val)):
            if name == "V":
                Vq = val
            else:
                Xq[:, HIST_COLS.index(name)] = val
        return model.predict(Xq) if not model.uses_v else model.predict(Xq, Vq)

    rows = []
    for name_a, name_b in r["ale2d_pairs"]:
        xa, ea = coord(name_a)
        xb, eb = coord(name_b)
        ba = np.searchsorted(ea, xa, side="right") - 1
        bb = np.searchsorted(eb, xb, side="right") - 1
        keep = (ba >= 0) & (ba < n_bins) & (bb >= 0) & (bb < n_bins)
        ba, bb = np.clip(ba, 0, n_bins - 1), np.clip(bb, 0, n_bins - 1)
        frame = {"pos": idx, "fold": fold.k, "pair": f"{name_a}|{name_b}", "bin_a": ba, "bin_b": bb, "keep": keep}
        models = [("m1", m1)] + ([("m0", m0)] if "V" not in (name_a, name_b) else [])
        for tag, model in models:
            frame[f"d2_{tag}"] = (evaluate(model, name_a, ea[ba + 1], name_b, eb[bb + 1])
                                  - evaluate(model, name_a, ea[ba + 1], name_b, eb[bb])
                                  - evaluate(model, name_a, ea[ba], name_b, eb[bb + 1])
                                  + evaluate(model, name_a, ea[ba], name_b, eb[bb]))
        rows.append(pd.DataFrame(frame))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------------------------
# WP5: matched-history pairs
# ------------------------------------------------------------------------------------
def match_rules(X_tr: np.ndarray, law_a, support, cfg: dict, k_caliper: int | None = None) -> dict:
    """Standardised-distance caliper and surprise-gap threshold, from the training segment only.

    The metric is the plain standardised Euclidean one.  Whitening was rejected at the design
    stage (docs/DESIGN_NOTES.md): it inflates the near-collinear noise directions of the HAR
    terms and gave worse balance per retained pair.
    """
    m = cfg["matching"]
    Z = support.basis_.z(X_tr)
    k = m["k_caliper"] if k_caliper is None else k_caliper
    caliper = float(np.median(NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)[0][:, k]))
    rng = np.random.default_rng(0)
    eps = law_a.eps_series_
    i, j = rng.integers(0, len(eps), 200_000), rng.integers(0, len(eps), 200_000)
    delta = float(np.quantile(np.abs(eps[i] - eps[j]), m["delta_quantile"]))
    return {"caliper": caliper, "delta": delta}


def select_pairs(Z_anchor, eps_anchor, Z_partner, eps_partner, allowed, caliper, delta, max_partners):
    """(anchor row, partner row, distance) for close histories with different surprises."""
    dist = cdist(Z_anchor, Z_partner)
    ok = allowed & (dist <= caliper) & (np.abs(eps_anchor[:, None] - eps_partner[None, :]) >= delta)
    out = []
    for a in range(len(Z_anchor)):
        cand = np.flatnonzero(ok[a])
        if cand.size:
            for p in cand[np.argsort(dist[a, cand], kind="stable")[:max_partners]]:
                out.append((a, p, dist[a, p]))
    return out


def pair_rows(data: SpecData, fold: Fold, m0, m1, laws, cfg: dict, oos_pos: np.ndarray,
              V_eval=None, k_caliper: int | None = None, rules: dict | None = None) -> pd.DataFrame:
    """Pairs anchored in this fold, both members forecast by this fold's fit.

    Later partners are strictly out-of-sample for the anchor's fit (its labels closed before
    the fit date), so a pair's contrast contains no drift from refitting.  Target windows do
    not overlap: a partner starts after the anchor's window has closed.
    """
    law_a, _, support = laws
    m = cfg["matching"]
    V_all = data.V if V_eval is None else V_eval
    if rules is None:
        rules = match_rules(data.X[fold.train_idx], law_a, support, cfg, k_caliper)
    anchors = fold.test_idx
    max_date = data.dates[anchors[-1]] + pd.DateOffset(years=m["max_gap_years"])
    partners = oos_pos[(data.dates[oos_pos] > data.dates[anchors[0]]) & (data.dates[oos_pos] <= max_date)]
    if len(partners) == 0:
        return pd.DataFrame()

    c = cfg["conditional"]
    def usable(pos):
        x, v = data.X[pos], V_all[pos]
        pit = law_a.pit(x, v)
        return support.in_support(x) & (pit >= c["pit_lo"]) & (pit <= c["pit_hi"])

    anchors = anchors[usable(anchors)]
    partners = partners[usable(partners)]
    if len(anchors) == 0 or len(partners) == 0:
        return pd.DataFrame()

    Z_a, Z_p = support.basis_.z(data.X[anchors]), support.basis_.z(data.X[partners])
    eps_a = law_a.standardized(data.X[anchors], V_all[anchors])
    eps_p = law_a.standardized(data.X[partners], V_all[partners])
    label_end = data.label_end[anchors].values
    max_dates = data.dates[anchors].values + np.timedelta64(int(m["max_gap_years"] * 366), "D")
    partner_dates = data.dates[partners].values
    allowed = (partner_dates[None, :] > label_end[:, None]) & (partner_dates[None, :] <= max_dates[:, None])
    chosen = select_pairs(Z_a, eps_a, Z_p, eps_p, allowed, rules["caliper"], rules["delta"], m["max_partners"])
    if not chosen:
        return pd.DataFrame()

    ai, pi, dist = map(np.asarray, zip(*chosen))
    pos_a, pos_p = anchors[ai.astype(int)], partners[pi.astype(int)]
    eps_av, eps_pv = eps_a[ai.astype(int)], eps_p[pi.astype(int)]
    hi_is_a = eps_av >= eps_pv
    hi, lo = np.where(hi_is_a, pos_a, pos_p), np.where(hi_is_a, pos_p, pos_a)

    def fcast(pos):
        x, v = data.X[pos], V_all[pos]
        return m0.predict(x), m1.predict(x, v)

    f0_hi, f1_hi = fcast(hi)
    f0_lo, f1_lo = fcast(lo)
    zdiff = support.basis_.z(data.X[hi]) - support.basis_.z(data.X[lo])
    frame = {
        "fold": fold.k, "hi": hi, "lo": lo, "dist": dist.astype(float),
        "deps": np.abs(eps_av - eps_pv),
        "y_hi": data.y[hi], "y_lo": data.y[lo],
        "f0_hi": f0_hi, "f0_lo": f0_lo, "f1_hi": f1_hi, "f1_lo": f1_lo,
        "caliper": rules["caliper"], "delta": rules["delta"],
    }
    for j, col in enumerate(HIST_COLS):
        frame[f"zd_{col}"] = zdiff[:, j]
    return pd.DataFrame(frame)


# ------------------------------------------------------------------------------------
# Everything for one fold
# ------------------------------------------------------------------------------------
def evaluate_fold(data: SpecData, fold: Fold, m0, m1, cfg: dict, oos_pos: np.ndarray, laws=None,
                  V_eval=None, parts=("origin", "response", "pairs"), features=None,
                  train_rows=None, with_kernel: bool = True, rules: dict | None = None) -> dict:
    """``laws`` defaults to laws fitted on this fold's training rows (or ``train_rows``).
    ``with_kernel`` controls the projection under law B."""
    if laws is None:
        rows = fold.train_idx if train_rows is None else train_rows
        laws = fit_laws(data.X[rows], data.V[rows], cfg)
    out = {}
    if "origin" in parts:
        out["origin"] = origin_rows(data, fold, m0, m1, laws, cfg, V_eval, project_b=with_kernel)
    if "response" in parts:
        feats = features or cfg["response"]["primary_features"]
        out["response"] = response_rows(data, fold, m0, m1, laws, cfg, feats, V_eval)
    if "pairs" in parts:
        out["pairs"] = pair_rows(data, fold, m0, m1, laws, cfg, oos_pos, V_eval, rules=rules)
    return out
