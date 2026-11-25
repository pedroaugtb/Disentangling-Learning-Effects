"""Monte Carlo audit of the diagnostics (proposal, Sec. 8).

DGP, calibrated on pre-OOS SPX data only (cross-correlations and persistence of the
transformed predictors, and the law of V | X):

    Z_t        stationary Gaussian VAR(1) with diagonal persistence
    log V_t    a(Z_t) + sigma(Z_t) eps_t,   eps_t stationary AR(1), N(0,1) marginals, independent of Z
    mu_t       exp{ g(Z_t) + b(Z_t) eps_t }
    Y_t        mu_t * mean_{s=1..21} xi_{t+s},   xi ~ Gamma(k/21, 21/k)

Y | Z, V is then exactly Gamma(k, mu/k) with mean mu, while overlapping windows reproduce the
dependence of the 30-day target.  Because eps is N(0,1) given Z, every population object has a
closed form: m1 = mu and m0 = exp{g + b^2/2}.  The proposal writes V = a(X) + u; V is modelled on
the log scale here so that it stays positive, as the implied variance does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from .conditional import _LOG_EPS, KernelLaw, LawBasis, LocationScaleLaw, SupportChecker, project
from .core import match_rules, select_pairs
from .data import HIST_COLS, SpecData
from .inference import block_bootstrap, boot_p_two_sided, hac_slope, percentile_ci
from .models import ENetModel, HARModel, XGBModel, _Base, forecast_floor, qlike, se
from .stats import PairStats, _wslope

_RVM = HIST_COLS.index("rv_m")
_RVW = HIST_COLS.index("rv_w")
_NEG = HIST_COLS.index("rv_neg_w")
_VAR = [HIST_COLS.index(c) for c in ("rv_d", "rv_w", "rv_m", "rv_neg_w")]


@dataclass
class Calibration:
    feat_mean: np.ndarray
    feat_sd: np.ndarray
    corr: np.ndarray
    ar: np.ndarray
    beta_a: np.ndarray
    gamma_s: np.ndarray
    scale_s: float
    g0: float
    g1: np.ndarray
    g_interaction: float = 0.06


def calibrate(data: SpecData, oos_start: str) -> Calibration:
    """Moments from origins before the out-of-sample start; no test-period data is used."""
    mask = data.usable & np.isfinite(data.y) & np.asarray(data.label_end < np.datetime64(oos_start))
    X, V, y = data.X[mask], data.V[mask], data.y[mask]
    T = LawBasis._log(X)
    mean, sd = T.mean(0), T.std(0)
    Z = (T - mean) / sd
    ar = np.array([np.corrcoef(Z[1:, j], Z[:-1, j])[0, 1] for j in range(Z.shape[1])])
    law = LocationScaleLaw(50).fit(X, V)
    A = np.column_stack([np.ones(len(Z)), Z])
    coef = np.linalg.lstsq(A, np.log(y), rcond=None)[0]
    # Regress on log Y, then re-level so E[Y] matches the training mean.
    g1 = coef[1:]
    g0 = float(np.log(y.mean()) - np.log(np.mean(np.exp(Z @ g1))))
    return Calibration(mean, sd, np.corrcoef(Z, rowvar=False), np.clip(ar, 0.0, 0.995),
                       law.beta_, law.gamma_, law.scale_, g0, g1)


def _state_b(Z, mode, b_bar, kappa):
    if mode == "redundant":
        return np.zeros(len(Z))
    if mode == "constant":
        return np.full(len(Z), b_bar)
    if mode == "state":
        return b_bar * (1.0 + kappa * np.tanh(Z[:, _RVM]))
    raise ValueError(mode)


def _a_sigma(Z, cal: Calibration):
    A = np.column_stack([np.ones(len(Z)), Z, Z[:, _RVM] ** 2])
    S = np.column_stack([np.ones(len(Z)), Z[:, _RVM], Z[:, HIST_COLS.index("ret_5")], Z[:, HIST_COLS.index("dd_22")]])
    return A @ cal.beta_a, cal.scale_s * np.exp(0.5 * S @ cal.gamma_s)


def _g(Z, cal: Calibration):
    return cal.g0 + Z @ cal.g1 + cal.g_interaction * Z[:, _RVM] * Z[:, _NEG]


def simulate(n: int, cal: Calibration, mode: str, b_bar: float, kappa: float, shape: float, overlap: int,
             rho_eps: float, rng: np.random.Generator) -> dict:
    p = len(cal.ar)
    A = np.diag(cal.ar)
    Q = cal.corr - A @ cal.corr @ A
    w, U = np.linalg.eigh((Q + Q.T) / 2)
    L = U @ np.diag(np.sqrt(np.maximum(w, 1e-8)))
    L0 = np.linalg.cholesky(cal.corr + 1e-8 * np.eye(p))
    z = L0 @ rng.standard_normal(p)
    Z = np.empty((n, p))
    for t in range(n):
        z = A @ z + L @ rng.standard_normal(p)
        Z[t] = z

    eps = np.empty(n)
    e = rng.standard_normal()
    scale = np.sqrt(1 - rho_eps**2)
    for t in range(n):
        e = rho_eps * e + scale * rng.standard_normal()
        eps[t] = e

    T = cal.feat_mean + cal.feat_sd * Z
    X = T.copy()
    X[:, _VAR] = np.maximum(np.exp(T[:, _VAR]) - _LOG_EPS, 0.0)   # inverse of LawBasis._log
    a, sigma = _a_sigma(Z, cal)
    V = np.exp(a + sigma * eps)
    b = _state_b(Z, mode, b_bar, kappa)
    g = _g(Z, cal)
    m1 = np.exp(g + b * eps)
    m0 = np.exp(g + 0.5 * b**2)
    xi = rng.gamma(shape / overlap, overlap / shape, size=n + overlap)
    window = np.convolve(xi, np.ones(overlap) / overlap, mode="valid")[1 : n + 1]
    return {"Z": Z, "X": X, "V": V, "eps": eps, "a": a, "sigma": sigma, "b": b, "m0": m0, "m1": m1,
            "Y": m1 * window}


# ------------------------------------------------------------------------------------
# Oracle learner and known / misspecified laws
# ------------------------------------------------------------------------------------
def gamma_log_glm(A, y, max_iter=100):
    beta = np.zeros(A.shape[1])
    beta[0] = np.log(y.mean())
    AtA_inv = np.linalg.pinv(A.T @ A)
    obj = np.mean(y * np.exp(-A @ beta) + A @ beta)
    for _ in range(max_iter):
        eta = np.clip(A @ beta, -30, 30)
        direction = AtA_inv @ (A.T @ (y * np.exp(-eta) - 1.0))
        step = 1.0
        while step > 1e-5:
            trial = beta + step * direction
            eta_t = np.clip(A @ trial, -30, 30)
            cand = np.mean(y * np.exp(-eta_t) + eta_t)
            if cand < obj:
                break
            step /= 2
        else:
            break
        done = obj - cand < 1e-10
        beta, obj = trial, cand
        if done:
            break
    return beta


class OracleModel(_Base):
    """Correctly specified functional forms with estimated coefficients."""

    def __init__(self, cal: Calibration):
        self.cal = cal

    def _design(self, X, V):
        T = LawBasis._log(X)
        Z = (T - self.cal.feat_mean) / self.cal.feat_sd
        t = np.tanh(Z[:, _RVM])
        base = [np.ones(len(Z)), *Z.T, Z[:, _RVM] * Z[:, _NEG], t, t**2]
        if V is not None:
            a, sigma = _a_sigma(Z, self.cal)
            e = (np.log(V) - a) / sigma
            base += [e, e * t]
        return np.column_stack(base)

    def fit(self, X, V, y, floor, **_):
        self.uses_v, self.floor = V is not None, floor
        self.beta_ = gamma_log_glm(self._design(X, V), y)
        return self

    def _raw(self, X, V):
        return np.exp(np.clip(self._design(X, V) @ self.beta_, -30, 30))


class TrueLaw:
    """The DGP's V | X with a deterministic standard-normal grid."""
    name = "true"

    def __init__(self, cal, n_draws=200):
        self.cal, self.grid_ = cal, norm.ppf((np.arange(n_draws) + 0.5) / n_draws)

    def _as(self, X):
        Z = (LawBasis._log(X) - self.cal.feat_mean) / self.cal.feat_sd
        return _a_sigma(Z, self.cal)

    def draws(self, X):
        a, s = self._as(X)
        v = np.exp(a[:, None] + s[:, None] * self.grid_[None, :])
        return v, np.full(v.shape, 1.0 / len(self.grid_))

    def standardized(self, X, V):
        a, s = self._as(X)
        return (np.log(V) - a) / s

    def pit(self, X, V):
        return norm.cdf(self.standardized(X, V))


class MisspecifiedLaw(LocationScaleLaw):
    """Linear mean without the curvature and a constant scale, against a richer DGP."""
    name = "misspecified"

    def _mu_design(self, Z):
        return np.column_stack([np.ones(len(Z)), Z[:, _RVM]])

    def _sigma_design(self, Z):
        return np.ones((len(Z), 1))


def make_learners(cal, cfg, floor):
    s = cfg["simulation"]
    xgb_hp = {k: v for k, v in s["xgb"].items() if k != "n_bag"}
    return {
        "oracle": lambda: OracleModel(cal),
        "har": lambda: HARModel("qlike"),
        "enet_light": lambda: ENetModel(s["enet"]["alpha_light"], s["enet"]["l1_ratio"], 40),
        "enet_heavy": lambda: ENetModel(s["enet"]["alpha_heavy"], s["enet"]["l1_ratio"], 40),
        "xgb": lambda: XGBModel(xgb_hp, s["xgb"]["n_bag"]),
    }


# ------------------------------------------------------------------------------------
# One replicate
# ------------------------------------------------------------------------------------
def _pairs_frame(test, f0, f1, law, support, rules, overlap, max_partners, max_rows=1200):
    n = min(len(test["Y"]), max_rows)
    X, V = test["X"][:n], test["V"][:n]
    Zw = support.basis_.z(X)
    eps = law.standardized(X, V)
    keep = support.in_support(X)
    positions = np.arange(n)
    allowed = (positions[None, :] > positions[:, None] + overlap) & keep[:, None] & keep[None, :]
    chosen = select_pairs(Zw, eps, Zw, eps, allowed, rules["caliper"], rules["delta"], max_partners)
    if not chosen:
        return pd.DataFrame()
    ai, pi, dist = map(np.asarray, zip(*chosen))
    ai, pi = ai.astype(int), pi.astype(int)
    hi_is_a = eps[ai] >= eps[pi]
    hi, lo = np.where(hi_is_a, ai, pi), np.where(hi_is_a, pi, ai)
    y = test["Y"]
    return pd.DataFrame({"hi": hi, "lo": lo, "dist": dist.astype(float), "deps": np.abs(eps[ai] - eps[pi]),
                         "y_hi": y[hi], "y_lo": y[lo], "f0_hi": f0[hi], "f0_lo": f0[lo],
                         "f1_hi": f1[hi], "f1_lo": f1[lo]})


def replicate(rep: int, mode: str, b_bar: float, cal: Calibration, cfg: dict, seed: int) -> list[dict]:
    s = cfg["simulation"]
    c = cfg["conditional"]
    rng = np.random.default_rng([seed, rep, int(b_bar * 1000), {"redundant": 0, "constant": 1, "state": 2}[mode]])
    args = (cal, mode, b_bar, s["state_kappa"], s["gamma_shape"], s["overlap"], s["rho_eps"])
    # Train and test come from one path, as in the real design, with a purge gap in between.
    n_total = s["n_train"] + s["overlap"] + s["n_test"]
    path = simulate(n_total, *args, rng)
    tr_slice, te_slice = slice(0, s["n_train"]), slice(s["n_train"] + s["overlap"], n_total)
    train = {k: v[tr_slice] for k, v in path.items()}
    test = {k: v[te_slice] for k, v in path.items()}
    long_path = simulate(s["n_long"], *args, rng)

    floor = forecast_floor(train["Y"], cfg)
    laws = {
        "true": TrueLaw(cal, c["n_draws"]),
        "A_wellspec": LocationScaleLaw(c["n_draws"]).fit(train["X"], train["V"]),
        "B_kernel": KernelLaw(c["kernel_bandwidths"], c["kernel_top_m"]).fit(train["X"], train["V"]),
        "misspecified": MisspecifiedLaw(c["n_draws"]).fit(train["X"], train["V"]),
    }
    support = SupportChecker(c["support_k"], c["support_q"]).fit(train["X"])
    rules = match_rules(train["X"], laws["A_wellspec"], support, cfg)

    y, m0_t, m1_t = test["Y"], test["m0"], test["m1"]
    n_test = len(y)
    # Population checks on the long path; (9) and (10) should hold up to Monte Carlo error.
    ideal = {
        "ideal_dmse": float(np.mean(se(long_path["Y"], long_path["m1"]) - se(long_path["Y"], long_path["m0"]))),
        "ideal_dmse_identity": float(-np.mean((long_path["m1"] - long_path["m0"]) ** 2)),
        "ideal_dqlike": float(np.mean(qlike(long_path["Y"], long_path["m1"]) - qlike(long_path["Y"], long_path["m0"]))),
        "ideal_dqlike_identity": float(-np.mean(long_path["m1"] / long_path["m0"] - 1 - np.log(long_path["m1"] / long_path["m0"]))),
        "ideal_rms_isd": float(np.sqrt(np.mean((long_path["m1"] - long_path["m0"]) ** 2))),
    }
    rows = []
    for name, factory in make_learners(cal, cfg, floor).items():
        m0 = factory().fit(train["X"], None, train["Y"], floor, seed=rep)
        m1 = factory().fit(train["X"], train["V"], train["Y"], floor, seed=rep)
        f0, f1 = m0.predict(test["X"]), m1.predict(test["X"], test["V"])
        lf0, lf1 = m0.predict(long_path["X"]), m1.predict(long_path["X"], long_path["V"])
        pop_dq = float(np.mean(qlike(long_path["Y"], lf1) - qlike(long_path["Y"], lf0)))
        dq = qlike(y, f1) - qlike(y, f0)
        boot_dq = block_bootstrap(lambda w: {"v": float((w * dq).sum() / w.sum())}, n_test,
                                  s["bootstrap_block"], s["bootstrap_n"], rep + 11)["v"]
        ci_dq = percentile_ci(boot_dq)

        tv, tw = laws["true"].draws(test["X"])
        p_true = project(m1, test["X"], tv, tw)
        support_flag = support.in_support(test["X"])
        for law_name, law in laws.items():
            vals, wts = law.draws(test["X"])
            p1 = project(m1, test["X"], vals, wts)
            hfs, isd = p1 - f0, f1 - p1
            u = test["V"] - (vals * wts).sum(axis=1)
            # Eq. (11): the two components sum to the estimated HFS by construction.
            fit_contrast, proj_err = p_true - f0, p1 - p_true
            rel_y, rel_x = (y - f0) / f0, isd / f0
            slope = hac_slope(rel_y, rel_x, lag=s["overlap"] - 1)
            boot_slope = block_bootstrap(lambda w: {"v": _wslope(rel_y, rel_x, w)},
                                         n_test, s["bootstrap_block"], s["bootstrap_n"], rep + 13)["v"]
            pit = law.pit(test["X"], test["V"])
            row = {
                "rep": rep, "mode": mode, "b_bar": b_bar, "learner": name, "law": law_name,
                "rms_D": float(np.sqrt(np.mean((f1 - f0) ** 2))),
                "rms_HFS": float(np.sqrt(np.mean(hfs**2))), "mean_HFS": float(hfs.mean()),
                "rms_fit_contrast": float(np.sqrt(np.mean(fit_contrast**2))),
                "rms_proj_err": float(np.sqrt(np.mean(proj_err**2))),
                "identity_residual": float(np.abs(hfs - fit_contrast - proj_err).max()),
                "rms_ISD": float(np.sqrt(np.mean(isd**2))), "mean_ISD": float(isd.mean()),
                "rmse_ISD_vs_true": float(np.sqrt(np.mean((isd - (m1_t - m0_t)) ** 2))),
                "true_rms_ISD": float(np.sqrt(np.mean((m1_t - m0_t) ** 2))),
                "spearman_absu_absISD": float(spearmanr(np.abs(u), np.abs(isd)).statistic) if isd.std() > 0 else float("nan"),
                "calib_slope_rel": slope["slope"], "calib_t": slope["t"],
                "calib_ci_lo": percentile_ci(boot_slope)[0], "calib_ci_hi": percentile_ci(boot_slope)[1],
                "calib_p_zero": boot_p_two_sided(boot_slope),
                "dqlike": float(dq.mean()), "dmse": float((se(y, f1) - se(y, f0)).mean()),
                "pop_dqlike_fit": pop_dq, "dq_ci_covers_pop": float(ci_dq[0] <= pop_dq <= ci_dq[1]),
                "dq_p_zero": boot_p_two_sided(boot_dq),
                "share_supported": float(support_flag.mean()),
                "law_cov80": float(np.mean(np.abs(pit - 0.5) <= 0.4)),
                "share_floored_f1": float(np.mean(f1 <= floor)),
                **ideal,
            }
            if law_name in ("A_wellspec", "true"):
                pairs = _pairs_frame(test, f0, f1, law, support, rules, s["overlap"], cfg["matching"]["max_partners"])
                row["pairs_n"] = float(len(pairs))
                if len(pairs) > 20:
                    ps = PairStats(pairs, np.arange(n_test))
                    base = ps.compute()
                    boot = block_bootstrap(ps.compute, n_test, s["bootstrap_block"], s["bootstrap_n"], rep + 17)
                    for key in ("pairs:conc_diff", "pairs:sep_slope_rel", "pairs:dqlike", "pairs:mean_dlogY", "pairs:conc1", "pairs:conc0"):
                        row[key] = base[key]
                    for key in ("pairs:conc_diff", "pairs:sep_slope_rel", "pairs:dqlike"):
                        row[f"{key}_p"] = boot_p_two_sided(boot[key])
                        lo, hi = percentile_ci(boot[key])
                        row[f"{key}_lo"], row[f"{key}_hi"] = lo, hi
            rows.append(row)
    return rows


def placebo_gate_replicate(rep: int, mode: str, b_bar: float, cal: Calibration, cfg: dict, seed: int) -> list[dict]:
    """Rejection frequency of the placebo gate; expensive, hence a smaller design."""
    s, c = cfg["simulation"], cfg["conditional"]
    rng = np.random.default_rng([seed, rep, 991, int(b_bar * 1000), {"redundant": 0, "constant": 1, "state": 2}[mode]])
    args = (cal, mode, b_bar, s["state_kappa"], s["gamma_shape"], s["overlap"], s["rho_eps"])
    path = simulate(s["n_train"] + s["overlap"] + s["n_test"], *args, rng)
    tr_slice = slice(0, s["n_train"])
    te_slice = slice(s["n_train"] + s["overlap"], None)
    train = {k: v[tr_slice] for k, v in path.items()}
    test = {k: v[te_slice] for k, v in path.items()}
    floor = forecast_floor(train["Y"], cfg)
    law = LocationScaleLaw(c["n_draws"]).fit(train["X"], train["V"])
    all_learners = make_learners(cal, cfg, floor)

    def statistics(model0, model1, V_eval):
        f0, f1 = model0.predict(test["X"]), model1.predict(test["X"], V_eval)
        vals, wts = law.draws(test["X"])
        p1 = project(model1, test["X"], vals, wts)
        isd = f1 - p1
        rel_y, rel_x = (test["Y"] - f0) / f0, isd / f0
        return {"rms_D": float(np.sqrt(np.mean((f1 - f0) ** 2))),
                "rms_HFS": float(np.sqrt(np.mean((p1 - f0) ** 2))),
                "rms_ISD": float(np.sqrt(np.mean(isd**2))),
                "calib_slope_rel": _wslope(rel_y, rel_x, np.ones(len(rel_y)))}

    rows = []
    for name in ("har", "enet_light", "xgb"):
        factory = all_learners[name]
        m0 = factory().fit(train["X"], None, train["Y"], floor, seed=rep)
        m1 = factory().fit(train["X"], train["V"], train["Y"], floor, seed=rep)
        observed = statistics(m0, m1, test["V"])
        draws = {k: [] for k in observed}
        for d in range(s["placebo_draws"]):
            V_tr = law.resample_path(train["X"], rng, cfg["placebo"]["block"])
            V_te = law.resample_path(test["X"], rng, cfg["placebo"]["block"])
            mp = factory().fit(train["X"], V_tr, train["Y"], floor, seed=rep + 1000 + d)
            for key, val in statistics(m0, mp, V_te).items():
                draws[key].append(val)
        for key, val in observed.items():
            arr = np.asarray(draws[key])
            rows.append({"rep": rep, "mode": mode, "b_bar": b_bar, "learner": name, "statistic": key,
                         "observed": val, "placebo_q95": float(np.quantile(arr, 0.95)),
                         "placebo_mean": float(arr.mean()),
                         "p_upper": float((np.sum(arr >= val) + 1) / (np.isfinite(arr).sum() + 1))})
    return rows
