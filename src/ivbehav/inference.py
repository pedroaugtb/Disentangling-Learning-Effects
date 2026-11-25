"""Dependence-aware inference: HAC long-run variance, DM/GW, moving-block bootstrap, Holm."""
from __future__ import annotations

import numpy as np
from scipy import stats


def newey_west_lrv(d: np.ndarray, lag: int) -> float:
    d = np.asarray(d, float)
    n = d.size
    e = d - d.mean()
    lrv = float(e @ e) / n
    for k in range(1, lag + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * float(e[k:] @ e[:-k]) / n
    return max(lrv, 1e-300)


def dm_test(diff: np.ndarray, horizon: int, lag: int | None = None) -> dict:
    """Diebold-Mariano with Harvey-Leybourne-Newbold correction.  diff = L_a - L_b."""
    d = np.asarray(diff, float)
    n = d.size
    k = horizon - 1 if lag is None else lag
    stat = d.mean() / np.sqrt(newey_west_lrv(d, k) / n)
    correction = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    stat *= np.sqrt(max(correction, 1e-6))
    p = 2 * stats.t.sf(abs(stat), n - 1)
    return {"n": n, "lag": k, "mean_diff": float(d.mean()), "dm": float(stat), "p": float(p)}


def gw_test(diff: np.ndarray, instruments: np.ndarray, lag: int) -> dict:
    """Giacomini-White Wald test that the loss differential is unpredictable from instruments."""
    d = np.asarray(diff, float)
    H = np.asarray(instruments, float)
    n, q = H.shape
    g = H * d[:, None]
    gbar = g.mean(axis=0)
    e = g - gbar
    S = e.T @ e / n
    for k in range(1, lag + 1):
        G = e[k:].T @ e[:-k] / n
        S += (1.0 - k / (lag + 1.0)) * (G + G.T)
    wald = float(n * gbar @ np.linalg.pinv(S) @ gbar)
    return {"n": n, "q": q, "lag": lag, "wald": wald, "p": float(stats.chi2.sf(wald, q))}


def hac_slope(y: np.ndarray, x: np.ndarray, lag: int, intercept: bool = False) -> dict:
    """OLS slope of y on x with a Newey-West standard error."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    A = np.column_stack([np.ones_like(x), x]) if intercept else x[:, None]
    AtA = A.T @ A
    # x numerically zero (the signal went unused) leaves the slope undefined, not huge.
    if np.linalg.cond(AtA) > 1e14 or x @ x <= 1e-16 * len(x):
        return {"slope": float("nan"), "se": float("nan"), "t": float("nan")}
    invA = np.linalg.inv(AtA)
    beta = invA @ (A.T @ y)
    score = A * (y - A @ beta)[:, None]
    S = score.T @ score
    for k in range(1, lag + 1):
        G = score[k:].T @ score[:-k]
        S += (1.0 - k / (lag + 1.0)) * (G + G.T)
    V = invA @ S @ invA
    slope, se = float(beta[-1]), float(np.sqrt(max(V[-1, -1], 0.0)))
    return {"slope": slope, "se": se, "t": slope / se if se > 0 else float("nan")}


def mbb_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """One moving-block bootstrap resample of positions 0..n-1 (circular)."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n


def mbb_weights(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Multiplicity of each position in one moving-block resample."""
    return np.bincount(mbb_indices(n, block, rng), minlength=n).astype(float)


def percentile_ci(samples, level: float = 0.95) -> tuple[float, float]:
    samples = np.asarray(samples, float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    lo, hi = np.quantile(samples, [a, 1 - a])
    return float(lo), float(hi)


def boot_p_two_sided(samples, null: float = 0.0) -> float:
    """Percentile-inversion p-value for H0: statistic = null."""
    samples = np.asarray(samples, float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return float("nan")
    low = (np.sum(samples <= null) + 1) / (samples.size + 1)
    high = (np.sum(samples >= null) + 1) / (samples.size + 1)
    return float(min(1.0, 2 * min(low, high)))


def block_bootstrap(fn, n_dates: int, block: int, n_boot: int, seed: int) -> dict[str, np.ndarray]:
    """Evaluate ``fn(weights) -> dict`` over moving-block date weights."""
    rng = np.random.default_rng(seed)
    out: dict[str, list] = {}
    for _ in range(n_boot):
        stats_b = fn(mbb_weights(n_dates, block, rng))
        for key, val in stats_b.items():
            out.setdefault(key, []).append(val)
    return {k: np.asarray(v, float) for k, v in out.items()}


def placebo_p(observed: float, placebo, tail: str = "upper") -> float:
    placebo = np.asarray(placebo, float)
    placebo = placebo[np.isfinite(placebo)]
    if placebo.size == 0 or not np.isfinite(observed):
        return float("nan")
    if tail == "upper":
        count = np.sum(placebo >= observed)
    elif tail == "lower":
        count = np.sum(placebo <= observed)
    else:
        count = np.sum(np.abs(placebo) >= abs(observed))
    return float((count + 1) / (placebo.size + 1))


def holm(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, float)
    adj = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(order) - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj
