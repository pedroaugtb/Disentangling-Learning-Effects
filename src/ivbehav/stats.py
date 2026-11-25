"""Scalar statistics of a stitched run.  Each class evaluates under date weights, so the same
code gives point estimates (unit weights) and moving-block bootstrap replicates.

Weights live on the ordered list of OOS positions; a pair is weighted by the product of its
two dates' weights, which keeps pairs that share dates dependent under resampling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .inference import block_bootstrap, boot_p_two_sided, percentile_ci
from .models import qlike, se


def _wmean(values, w):
    total = w.sum()
    return float((values * w).sum() / total) if total > 0 else float("nan")


def _wrms(values, w):
    return float(np.sqrt(_wmean(values**2, w)))


def _wslope(y, x, w):
    """Weighted slope through the origin; undefined when x is numerically zero (V unused)."""
    denom = (w * x * x).sum()
    if denom <= 1e-16 * max(w.sum(), 1e-300):
        return float("nan")
    return float((w * x * y).sum() / denom)


class OriginStats:
    """D / HFS / ISD summaries on supported states (WP2, WP4)."""

    def __init__(self, frame: pd.DataFrame, oos_pos: np.ndarray):
        self.f = frame
        where = {p: i for i, p in enumerate(oos_pos)}
        self.date_i = np.array([where[p] for p in frame["pos"]])
        self.n_dates = len(oos_pos)
        self.has_b = "HFS_b" in frame
        f = frame
        self.loss = {"qlike0": qlike(f.y, f.f0), "qlike1": qlike(f.y, f.f1),
                     "se0": se(f.y, f.f0), "se1": se(f.y, f.f1)}
        self.sup_x = f["sup_x"].to_numpy(bool)
        self.sup_xv = f["sup_xv"].to_numpy(bool)

    def compute(self, w_dates=None) -> dict:
        f = self.f
        w = np.ones(len(f)) if w_dates is None else w_dates[self.date_i]
        wx, wv = w * self.sup_x, w * self.sup_xv
        f0, y = f["f0"].to_numpy(), f["y"].to_numpy()
        D, H, I = (f[c].to_numpy() for c in ("D", "HFS_a", "ISD_a"))
        out = {
            "share_sup_x": _wmean(self.sup_x.astype(float), w),
            "share_sup_xv": _wmean(self.sup_xv.astype(float), w),
            # Validity checks: full common origin set, negative favours the augmented model.
            "dqlike": _wmean(self.loss["qlike1"] - self.loss["qlike0"], w),
            "dmse": _wmean(self.loss["se1"] - self.loss["se0"], w),
            "qlike0": _wmean(self.loss["qlike0"], w),
            "qlike1": _wmean(self.loss["qlike1"], w),
            # Level scale, the proposal's default for behavioral comparisons.
            "rms_D": _wrms(D, wx), "mean_D": _wmean(D, wx),
            "rms_HFS": _wrms(H, wx), "mean_HFS": _wmean(H, wx),
            "rms_ISD": _wrms(I, wv), "mean_ISD": _wmean(I, wv),
            "cov_HFS_ISD": _wmean(H * I, wv),
            # Relative to f0: stops a few 2020 dates from carrying the whole statistic.
            "rms_D_rel": _wrms(D / f0, wx), "rms_HFS_rel": _wrms(H / f0, wx), "rms_ISD_rel": _wrms(I / f0, wv),
            # Eq. (8): slope of the f0 forecast error on ISD equals one for ideal predictors.
            "calib_slope_level": _wslope(y - f0, I, wv),
            "calib_slope_rel": _wslope((y - f0) / f0, I / f0, wv),
        }
        total = out["rms_HFS"] ** 2 + out["rms_ISD"] ** 2
        out["share_ISD"] = out["rms_ISD"] ** 2 / total if total > 0 else float("nan")
        if self.has_b:
            # Law B quantities, only where its neighbourhood is large enough to be credible.
            wb = w * f["sup_b"].to_numpy(bool)
            wbv = wb * self.sup_xv
            Hb, Ib = f["HFS_b"].to_numpy(), f["ISD_b"].to_numpy()
            out["share_sup_b"] = _wmean(f["sup_b"].to_numpy(float), w)
            out["rms_HFS_b"] = _wrms(Hb, wb)
            out["rms_HFS_on_sup_b"] = _wrms(H, wb)
            out["rms_ISD_b"] = _wrms(Ib, wbv)
            out["rms_law_gap"] = _wrms(H - Hb, wb)
            out["rms_law_gap_rel"] = _wrms((H - Hb) / f0, wb)
            out["calib_slope_rel_b"] = _wslope((y - f0) / f0, Ib / f0, wbv)
        return out

    def unweighted_extras(self) -> dict:
        """Rank diagnostics; these stay out of the bootstrap."""
        f = self.f[self.sup_xv]
        if len(f) < 10:
            return {}
        return {
            # Sec. 3.4: mechanical in an affine rule, reported so it is not read as evidence.
            "spearman_absu_absISD": float(spearmanr(np.abs(f["u_a"]), np.abs(f["ISD_a"])).statistic),
            "spearman_ISD_error": float(spearmanr(f["ISD_a"], f["y"] - f["f0"]).statistic),
            "spearman_ISDrel_errorrel": float(spearmanr(f["ISD_a"] / f["f0"], (f["y"] - f["f0"]) / f["f0"]).statistic),
            "share_floored_f0": float(self.f["floored0"].mean()),
            "share_floored_f1": float(self.f["floored1"].mean()),
        }


class ResponseStats:
    """P2 statistics from stitched local effects.

    For feature k, A_m(b) accumulates mean local effects of function m up to bin b.
    S_m    : change between the q10 and q90 training edges, in variance units.
    dS     : S_1 - S_0 (attenuation if negative for a positive response).
    T      : count-weighted L2 distance between centred curves of f1 and f0.
    T_shape: the part of T that a proportional rescaling of f0's curve cannot explain.
    The ``p`` suffix replaces f1 at observed V by its projection P1(x).
    """

    def __init__(self, frame: pd.DataFrame, oos_pos: np.ndarray, cfg: dict):
        r = cfg["response"]
        self.n_bins = r["n_bins"]
        self.a, self.b = r["slope_from_bin"], r["slope_to_bin"]
        self.min_count = r["min_bin_count"]
        where = {p: i for i, p in enumerate(oos_pos)}
        kept = frame[frame["keep"]]
        self.features = list(frame["feature"].unique())
        self.withheld = {k: float(1 - frame.loc[frame["feature"] == k, "keep"].mean()) for k in self.features}
        self.data = {}
        for k, g in kept.groupby("feature"):
            self.data[k] = {
                "bin": g["bin"].to_numpy(int),
                "date_i": np.array([where[p] for p in g["pos"]]),
                "le0": g["le0"].to_numpy(), "le1": g["le1"].to_numpy(),
                "le1p": g["le1p"].to_numpy() if "le1p" in g else None,
                "pit": g["pit_a"].to_numpy(),
            }
        self.base = {k: self._bin_means(d, np.ones(len(d["bin"]))) for k, d in self.data.items()}

    def _bin_means(self, d, w):
        count = np.bincount(d["bin"], weights=w, minlength=self.n_bins)
        means = {}
        for m in ("le0", "le1", "le1p"):
            if d[m] is None:
                continue
            total = np.bincount(d["bin"], weights=w * d[m], minlength=self.n_bins)
            with np.errstate(invalid="ignore", divide="ignore"):
                means[m] = total / count
        return count, means

    def _curve_stats(self, k, means, count, base_count):
        valid = base_count >= self.min_count
        out = {}
        curves = {}
        for m, vals in means.items():
            vals = np.where(np.isfinite(vals), vals, self.base[k][1][m])
            inc = np.where(valid, vals, 0.0)
            curve = np.concatenate([[0.0], np.cumsum(inc)])[:-1]   # value at each bin's lower edge
            weight = np.maximum(count, 0.0) * valid
            curve = curve - (weight * curve).sum() / max(weight.sum(), 1e-12)
            curves[m] = curve
            out[f"S_{m}"] = float(inc[self.a : self.b].sum())
        weight = np.maximum(count, 0.0) * valid
        weight = weight / max(weight.sum(), 1e-12)

        def l2(c):
            return float(np.sqrt((weight * c**2).sum()))

        result = {f"{k}:S0": out["S_le0"], f"{k}:S1": out["S_le1"], f"{k}:dS": out["S_le1"] - out["S_le0"],
                  f"{k}:T": l2(curves["le1"] - curves["le0"]),
                  f"{k}:n_valid_bins": float(valid.sum())}
        lam = (weight * curves["le1"] * curves["le0"]).sum() / max((weight * curves["le0"] ** 2).sum(), 1e-300)
        result[f"{k}:T_shape"] = l2(curves["le1"] - lam * curves["le0"])
        result[f"{k}:ratio"] = out["S_le1"] / out["S_le0"] if out["S_le0"] != 0 else float("nan")
        if "le1p" in curves:
            result[f"{k}:S1p"] = out["S_le1p"]
            result[f"{k}:dSp"] = out["S_le1p"] - out["S_le0"]
            result[f"{k}:Tp"] = l2(curves["le1p"] - curves["le0"])
        return result

    def compute(self, w_dates=None) -> dict:
        out = {}
        for k, d in self.data.items():
            w = np.ones(len(d["bin"])) if w_dates is None else w_dates[d["date_i"]]
            count, means = self._bin_means(d, w)
            out.update(self._curve_stats(k, means, count, self.base[k][0]))
            # Does the response itself depend on the IV state?  Terciles of the conditional PIT.
            for label, lo, hi in (("low", 0.0, 1 / 3), ("high", 2 / 3, 1.0 + 1e-9)):
                wk = w * ((d["pit"] >= lo) & (d["pit"] < hi))
                c_s, m_s = self._bin_means(d, wk)
                vals = np.where(np.isfinite(m_s["le1"]), m_s["le1"], self.base[k][1]["le1"])
                inc = np.where(self.base[k][0] >= self.min_count, vals, 0.0)
                out[f"{k}:S1_{label}IV"] = float(inc[self.a : self.b].sum())
            out[f"{k}:S1_high_minus_low"] = out[f"{k}:S1_highIV"] - out[f"{k}:S1_lowIV"]
        return out

    def curves(self) -> pd.DataFrame:
        """Accumulated curves for the descriptive figures."""
        rows = []
        for k, d in self.data.items():
            count, means = self.base[k]
            valid = count >= self.min_count
            for m, vals in means.items():
                inc = np.where(valid & np.isfinite(vals), vals, 0.0)
                curve = np.concatenate([[0.0], np.cumsum(inc)])
                for b, value in enumerate(curve):
                    rows.append({"feature": k, "function": m, "edge": b, "value": float(value),
                                 "bin_count": float(count[b]) if b < len(count) else float("nan")})
        return pd.DataFrame(rows)


class PairStats:
    """P3 statistics on matched-history pairs, oriented so 'hi' has the larger IV surprise."""

    def __init__(self, frame: pd.DataFrame, oos_pos: np.ndarray):
        self.f = frame
        self.empty = frame is None or frame.empty
        if self.empty:
            return
        where = {p: i for i, p in enumerate(oos_pos)}
        self.hi_i = np.array([where[p] for p in frame["hi"]])
        self.lo_i = np.array([where[p] for p in frame["lo"]])
        f = frame
        self.dY = (f.y_hi - f.y_lo).to_numpy()
        self.df0 = (f.f0_hi - f.f0_lo).to_numpy()
        self.df1 = (f.f1_hi - f.f1_lo).to_numpy()
        self.dD = self.df1 - self.df0
        self.dY_rel = ((f.y_hi - f.f0_hi) / f.f0_hi - (f.y_lo - f.f0_lo) / f.f0_lo).to_numpy()
        self.dD_rel = ((f.f1_hi - f.f0_hi) / f.f0_hi - (f.f1_lo - f.f0_lo) / f.f0_lo).to_numpy()
        self.dlogY = (np.log(f.y_hi) - np.log(f.y_lo)).to_numpy()
        self.c0 = 0.5 * (1 + np.sign(self.df0) * np.sign(self.dY))
        self.c1 = 0.5 * (1 + np.sign(self.df1) * np.sign(self.dY))
        self.dql = 0.5 * ((qlike(f.y_hi, f.f1_hi) - qlike(f.y_hi, f.f0_hi))
                          + (qlike(f.y_lo, f.f1_lo) - qlike(f.y_lo, f.f0_lo)))
        self.dse = 0.5 * ((se(f.y_hi, f.f1_hi) - se(f.y_hi, f.f0_hi))
                          + (se(f.y_lo, f.f1_lo) - se(f.y_lo, f.f0_lo)))

    def compute(self, w_dates=None) -> dict:
        if self.empty:
            return {"pairs:n": 0.0}
        w = np.ones(len(self.dY)) if w_dates is None else w_dates[self.hi_i] * w_dates[self.lo_i]
        conc0, conc1 = _wmean(self.c0, w), _wmean(self.c1, w)
        return {
            "pairs:conc0": conc0, "pairs:conc1": conc1, "pairs:conc_diff": conc1 - conc0,
            "pairs:sep_slope": _wslope(self.dY - self.df0, self.dD, w),
            "pairs:sep_slope_rel": _wslope(self.dY_rel, self.dD_rel, w),
            "pairs:mean_dY": _wmean(self.dY, w), "pairs:mean_df0": _wmean(self.df0, w),
            "pairs:mean_df1": _wmean(self.df1, w), "pairs:mean_dlogY": _wmean(self.dlogY, w),
            "pairs:p_hi_exceeds": _wmean((self.dY > 0).astype(float), w),
            "pairs:dqlike": _wmean(self.dql, w), "pairs:dmse": _wmean(self.dse, w),
        }

    def design(self) -> dict:
        """Match quality; no outcomes involved."""
        if self.empty:
            return {"pairs:n": 0.0}
        f = self.f
        z = f[[c for c in f.columns if c.startswith("zd_")]]
        return {
            "pairs:n": float(len(f)),
            "pairs:n_dates": float(len(set(f["hi"]) | set(f["lo"]))),
            "pairs:mean_dist": float(f["dist"].mean()),
            "pairs:mean_deps": float(f["deps"].mean()),
            "pairs:max_abs_mean_zdiff": float(z.mean().abs().max()),
        }


class RunStats:
    """All statistics of one stitched run, under common date weights."""

    def __init__(self, oos_pos: np.ndarray, cfg: dict, origin=None, response=None, pairs=None):
        self.n_dates = len(oos_pos)
        self.origin = None if origin is None else OriginStats(origin, oos_pos)
        self.response = None if response is None else ResponseStats(response, oos_pos, cfg)
        self.pairs = None if pairs is None else PairStats(pairs, oos_pos)

    def compute(self, w_dates=None) -> dict:
        out = {}
        for part in (self.origin, self.response, self.pairs):
            if part is not None:
                out.update(part.compute(w_dates))
        return out

    def point(self) -> dict:
        out = self.compute()
        if self.origin is not None:
            out.update(self.origin.unweighted_extras())
        if self.pairs is not None:
            out.update(self.pairs.design())
        return out


def stitch(results: list[dict], key: str) -> pd.DataFrame | None:
    if key not in results[0]:
        return None
    return pd.concat([r[key] for r in results], ignore_index=True)


def run_stats(results: list[dict], oos_pos: np.ndarray, cfg: dict) -> RunStats:
    return RunStats(oos_pos, cfg, stitch(results, "origin"), stitch(results, "response"), stitch(results, "pairs"))


def stitched_stats(results: list[dict], oos_pos: np.ndarray, cfg: dict) -> dict:
    """Point estimates for one complete run."""
    return run_stats(results, oos_pos, cfg).point()


def bootstrap_table(stat, n_dates: int, cfg: dict, seed: int, base: dict | None = None) -> pd.DataFrame:
    """Percentile CIs and p-values for H0: statistic = 0, at every configured block length."""
    boot = cfg["bootstrap"]
    base = stat.compute() if base is None else base
    rows = []
    for block in boot["block_lengths"]:
        samples = block_bootstrap(stat.compute, n_dates, block, boot["n"], seed + block)
        for key, values in samples.items():
            lo, hi = percentile_ci(values)
            rows.append({"statistic": key, "block": block, "estimate": base.get(key, float("nan")),
                         "boot_mean": float(np.nanmean(values)), "boot_sd": float(np.nanstd(values)),
                         "ci_lo": lo, "ci_hi": hi, "p_zero": boot_p_two_sided(values)})
    return pd.DataFrame(rows)
