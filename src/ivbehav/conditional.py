"""Training-only conditional laws p(V | X) and support diagnostics (WP2).

Law A : parsimonious location-scale regression of log V with resampled training residuals.
Law B : substantively different nonparametric kernel-weighted empirical conditional law.

A law returns (values, weights) with shape (n_origins, n_atoms), each row's weights summing
to one, so projection is the same weighted average whichever law is used.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from .data import HIST_COLS, VARIANCE_COLS

_VAR_IDX = [HIST_COLS.index(c) for c in VARIANCE_COLS]
_RVM = HIST_COLS.index("rv_m")
_RET5 = HIST_COLS.index("ret_5")
_DD = HIST_COLS.index("dd_22")
_LOG_EPS = 1e-7


class LawBasis:
    """Transforms and whitens X with training moments; Euclidean here is Mahalanobis there."""

    def fit(self, X):
        T = self._log(X)
        self.m_, self.s_ = T.mean(0), T.std(0)
        Z = (T - self.m_) / self.s_
        w, U = np.linalg.eigh(np.cov(Z, rowvar=False))
        self.W_ = U @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-8))) @ U.T
        return self

    @staticmethod
    def _log(X):
        T = np.asarray(X, float)[:, : len(HIST_COLS)].copy()     # ignore return lags, if any
        T[:, _VAR_IDX] = np.log(np.maximum(T[:, _VAR_IDX], 0.0) + _LOG_EPS)
        return T

    def z(self, X):
        return (self._log(X) - self.m_) / self.s_

    def whiten(self, X):
        return self.z(X) @ self.W_


class LocationScaleLaw:
    """log V = mu(x) + sigma(x) eps, with eps drawn from the training residual distribution.

    mu is linear in the standardised predictors plus the squared log 22-day variance; log
    sigma is linear in the state variables that drive IV dispersion.
    """
    name = "A_locscale"

    def __init__(self, n_draws=200):
        self.n_draws = n_draws

    def _mu_design(self, Z):
        return np.column_stack([np.ones(len(Z)), Z, Z[:, _RVM] ** 2])

    def _sigma_design(self, Z):
        return np.column_stack([np.ones(len(Z)), Z[:, _RVM], Z[:, _RET5], Z[:, _DD]])

    def fit(self, X, V):
        self.basis_ = LawBasis().fit(X)
        Z = self.basis_.z(X)
        logv = np.log(V)
        self.beta_ = np.linalg.lstsq(self._mu_design(Z), logv, rcond=None)[0]
        resid = logv - self._mu_design(Z) @ self.beta_
        c = 1e-3 * resid.var()
        self.gamma_ = np.linalg.lstsq(self._sigma_design(Z), np.log(resid**2 + c), rcond=None)[0]
        sigma = np.exp(0.5 * self._sigma_design(Z) @ self.gamma_)
        eps = resid / sigma
        self.scale_ = float(eps.std())
        self.eps_series_ = eps / self.scale_         # training time order, for block placebos
        self.eps_sorted_ = np.sort(self.eps_series_)
        probs = (np.arange(self.n_draws) + 0.5) / self.n_draws
        self.grid_ = np.quantile(self.eps_series_, probs)
        return self

    def mu_sigma(self, X):
        Z = self.basis_.z(X)
        return self._mu_design(Z) @ self.beta_, self.scale_ * np.exp(0.5 * self._sigma_design(Z) @ self.gamma_)

    def draws(self, X):
        mu, sigma = self.mu_sigma(X)
        values = np.exp(mu[:, None] + sigma[:, None] * self.grid_[None, :])
        return values, np.full(values.shape, 1.0 / self.n_draws)

    def mean(self, X):
        mu, sigma = self.mu_sigma(X)
        return np.exp(mu) * np.mean(np.exp(sigma[:, None] * self.grid_[None, :]), axis=1)

    def standardized(self, X, V):
        mu, sigma = self.mu_sigma(X)
        return (np.log(V) - mu) / sigma

    def pit(self, X, V):
        return np.searchsorted(self.eps_sorted_, self.standardized(X, V), side="right") / len(self.eps_sorted_)

    def resample_path(self, X, rng, block: int | None):
        """Placebo V for rows X: residuals resampled from THIS law's training residuals.

        block=None draws residuals iid; otherwise moving blocks along the training series,
        keeping the persistence of the actual surprises but no link to the future of Y.
        """
        n, T = len(X), len(self.eps_series_)
        if block is None:
            eps = rng.choice(self.eps_series_, size=n, replace=True)
        else:
            starts = rng.integers(0, T - block + 1, size=n // block + 1)
            eps = np.concatenate([self.eps_series_[s : s + block] for s in starts])[:n]
        mu, sigma = self.mu_sigma(X)
        return np.exp(mu + sigma * eps)


class KernelLaw:
    """Gaussian-kernel empirical conditional law of V on whitened X.

    The bandwidth is chosen on a purged chronological hold-out inside the training window by
    CRPS on log V.  Kish's n_eff at each origin tells how much data backs the local law.
    """
    name = "B_kernel"

    def __init__(self, bandwidths, top_m=300, holdout=0.2, gap=22):
        self.bandwidths, self.top_m, self.holdout, self.gap = list(bandwidths), top_m, holdout, gap

    def _weights(self, sqdist, Vtr, h):
        d2 = sqdist / (2 * h**2)
        d2 = d2 - d2.min(axis=1, keepdims=True)             # numerically safe far from the data
        m = min(self.top_m, d2.shape[1])
        idx = np.argpartition(d2, m - 1, axis=1)[:, :m]
        w = np.exp(-np.take_along_axis(d2, idx, axis=1))
        return Vtr[idx], w / w.sum(axis=1, keepdims=True)

    def fit(self, X, V, bandwidth=None):
        self.basis_ = LawBasis().fit(X)
        Z = self.basis_.whiten(X)
        self.Z_, self.V_ = Z, np.asarray(V, float)
        if bandwidth is not None:         # refits reuse the bandwidth of the observed fold
            self.h_ = float(bandwidth)
            return self
        n = len(Z)
        split = int(n * (1 - self.holdout))
        fit_end = split - self.gap
        sqdist = cdist(Z[split:], Z[:fit_end], "sqeuclidean")
        scores = []
        for h in self.bandwidths:
            values, weights = self._weights(sqdist, V[:fit_end], h)
            scores.append(np.mean(weighted_crps(np.log(values), weights, np.log(V[split:]))))
        self.h_ = float(self.bandwidths[int(np.argmin(scores))])
        self.holdout_crps_ = dict(zip(self.bandwidths, map(float, scores)))
        return self

    def draws(self, X):
        return self._weights(cdist(self.basis_.whiten(X), self.Z_, "sqeuclidean"), self.V_, self.h_)

    def mean(self, X):
        values, weights = self.draws(X)
        return (values * weights).sum(axis=1)

    def n_eff(self, X):
        _, weights = self.draws(X)
        return 1.0 / (weights**2).sum(axis=1)

    def pit(self, X, V):
        values, weights = self.draws(X)
        return (weights * (values <= V[:, None])).sum(axis=1)


def weighted_crps(values, weights, obs):
    """CRPS of discrete laws (rows) at obs, E|A - y| - 0.5 E|A - A'|."""
    order = np.argsort(values, axis=1)
    v = np.take_along_axis(values, order, axis=1)
    w = np.take_along_axis(weights, order, axis=1)
    first = (w * np.abs(v - obs[:, None])).sum(axis=1)
    cum = np.cumsum(w, axis=1)
    gaps = v[:, 1:] - v[:, :-1]
    second = 2.0 * (gaps * cum[:, :-1] * (1.0 - cum[:, :-1])).sum(axis=1)
    return first - 0.5 * second


class SupportChecker:
    """x is supported if its k-NN distance to training states is within the training q-th
    percentile of leave-one-out k-NN distances."""

    def __init__(self, k=20, q=0.99):
        self.k, self.q = k, q

    def fit(self, X):
        self.basis_ = LawBasis().fit(X)
        Z = self.basis_.whiten(X)
        self.nn_ = NearestNeighbors(n_neighbors=self.k).fit(Z)
        loo = NearestNeighbors(n_neighbors=self.k + 1).fit(Z).kneighbors(Z)[0][:, -1]
        self.threshold_ = float(np.quantile(loo, self.q))
        return self

    def distance(self, X):
        return self.nn_.kneighbors(self.basis_.whiten(X))[0][:, -1]

    def in_support(self, X):
        return self.distance(X) <= self.threshold_


def project(model, X, values, weights) -> np.ndarray:
    """P1(x) = sum_j w_j f1(x, v_j), the conditional projection of Eq. (3)."""
    n, m = values.shape
    preds = model.predict(np.repeat(X, m, axis=0), values.ravel()).reshape(n, m)
    return (preds * weights).sum(axis=1)


def fit_laws(X, V, cfg, bandwidth=None):
    c = cfg["conditional"]
    law_a = LocationScaleLaw(c["n_draws"]).fit(X, V)
    law_b = KernelLaw(c["kernel_bandwidths"], c["kernel_top_m"]).fit(X, V, bandwidth)
    support = SupportChecker(c["support_k"], c["support_q"]).fit(X)
    return law_a, law_b, support
