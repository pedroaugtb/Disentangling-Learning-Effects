"""Forecasting families.  Every f0/f1 pair shares objective, candidates and seeds.

Core families all target E[Y | information] on the level scale by minimising QLIKE (the
Gamma deviance up to terms free of the forecast):

* ``har``  : linear variance forecast with a leverage control; Gamma quasi-likelihood,
             identity link.
* ``enet`` : log-link Gamma elastic net on a log-variance dictionary with modest
             nonlinearity.
* ``xgb``  : gradient-boosted trees, ``reg:gamma`` (log link), bagged over seeds.

``predict(X, V)`` takes raw predictors, applies the training-only transforms, and returns
forecasts floored at a training-only positive constant, the same rule in all families.
``har_ols`` (MSE objective) and the log-target learners are sensitivity analyses.
"""
from __future__ import annotations

import itertools

import numpy as np
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet

from .data import HIST_COLS, VARIANCE_COLS

HAR_COLS = ["rv_d", "rv_w", "rv_m", "rv_neg_w"]
N_HIST = len(HIST_COLS)    # X may carry return lags after the summaries (sequence models only)
NEURAL_FAMILIES = ("mlp", "tcn", "attn")
SEQUENCE_FAMILIES = ("tcn", "attn")
_HAR_IDX = [HIST_COLS.index(c) for c in HAR_COLS]
_VAR_IDX = [HIST_COLS.index(c) for c in VARIANCE_COLS]
_LOG_EPS = 1e-7            # about (3 bp)^2; keeps log(rv) finite on zero-return days


def qlike(y, q):
    """QLIKE normalised to zero at q = y; differences equal those of y/q + log q."""
    ratio = np.asarray(y) / np.asarray(q)
    return ratio - np.log(ratio) - 1.0


def se(y, q):
    return (np.asarray(y) - np.asarray(q)) ** 2


def forecast_floor(y_train: np.ndarray, cfg: dict) -> float:
    return float(cfg["floor_frac"] * np.quantile(y_train, 0.01))


def _with_v(X, V):
    return X if V is None else np.column_stack([X, V])


class _Base:
    """Common floor handling.  Subclasses implement ``_raw(X, V)``."""
    floor: float
    uses_v: bool

    def predict(self, X, V=None) -> np.ndarray:
        if self.uses_v and V is None:
            raise ValueError("augmented model needs V")
        return np.maximum(self._raw(X, V if self.uses_v else None), self.floor)


# ------------------------------------------------------------------------------------
# HAR: identity link
# ------------------------------------------------------------------------------------
def fit_gamma_identity(A, y, floor, max_iter=200, tol=1e-10):
    """Minimise mean(y/mu + log mu), mu = A beta floored, by IRLS with step halving."""
    def objective(b):
        mu = np.maximum(A @ b, floor)
        return np.mean(y / mu + np.log(mu))

    beta = np.zeros(A.shape[1])
    beta[0] = y.mean()                  # first column is the intercept: mu > 0 everywhere
    obj = objective(beta)
    for _ in range(max_iter):
        w = 1.0 / np.maximum(A @ beta, floor) ** 2
        AtW = A.T * w
        new = np.linalg.solve(AtW @ A + 1e-12 * np.eye(A.shape[1]), AtW @ y)
        step, improved = 1.0, False
        while step > 1e-5:
            trial = beta + step * (new - beta)
            cand = objective(trial)
            if cand < obj:
                improved = True
                break
            step /= 2
        if not improved:
            break
        done = obj - cand < tol * (abs(obj) + 1e-12)
        beta, obj = trial, cand
        if done:
            break
    return beta


class HARModel(_Base):
    """Linear HAR with leverage control.  ``objective``: 'qlike' (primary) or 'mse'."""

    def __init__(self, objective="qlike"):
        self.objective = objective

    def _design(self, X, V):
        F = _with_v(X[:, _HAR_IDX], V)
        return np.column_stack([np.ones(len(F)), (F - self.m_) / self.s_])

    def fit(self, X, V, y, floor, **_):
        self.uses_v, self.floor = V is not None, floor
        F = _with_v(X[:, _HAR_IDX], V)
        self.m_, self.s_ = F.mean(0), F.std(0)
        A = self._design(X, V)
        if self.objective == "qlike":
            self.beta_ = fit_gamma_identity(A, y, floor)
        elif self.objective == "mse":
            self.beta_ = np.linalg.lstsq(A, y, rcond=None)[0]
        else:
            raise ValueError(self.objective)
        return self

    def _raw(self, X, V):
        return self._design(X, V) @ self.beta_

    def coef_raw(self):
        """Slopes per unit of each raw predictor, as a dict."""
        names = HAR_COLS + (["V"] if self.uses_v else [])
        return dict(zip(names, self.beta_[1:] / self.s_))

    def coef_std(self):
        """Forecast change per training standard deviation of each predictor."""
        names = HAR_COLS + (["V"] if self.uses_v else [])
        return dict(zip(names, self.beta_[1:]))


# ------------------------------------------------------------------------------------
# Elastic net: log link
# ------------------------------------------------------------------------------------
def _log_features(X):
    T = np.asarray(X, float)[:, :N_HIST].copy()
    T[:, _VAR_IDX] = np.log(np.maximum(T[:, _VAR_IDX], 0.0) + _LOG_EPS)
    return T


class ENetModel(_Base):
    """Gamma elastic net with log link.

    Fisher scoring with step halving.  The Gamma/log Fisher weights are constant, so each
    step is one unweighted elastic-net regression on the working response
    eta + (y - mu) / mu.
    """

    def __init__(self, alpha, l1_ratio, max_outer=60):
        self.alpha, self.l1_ratio, self.max_outer = alpha, l1_ratio, max_outer

    def _base(self, X, V):
        T = _log_features(X)
        return T if V is None else np.column_stack([T, np.log(V)])

    def _expand(self, X, V):
        Z = (np.clip(self._base(X, V), self.lo_, self.hi_) - self.m_) / self.s_
        p = Z.shape[1]
        pairs = [Z[:, i] * Z[:, j] for i, j in itertools.combinations_with_replacement(range(p), 2)]
        return np.column_stack([Z, *pairs])

    def fit(self, X, V, y, floor, **_):
        self.uses_v, self.floor = V is not None, floor
        T = self._base(X, V)
        self.lo_, self.hi_ = T.min(0), T.max(0)      # no extrapolation beyond the training range
        self.m_, self.s_ = T.mean(0), T.std(0)
        E = self._expand(X, V)
        self.em_, self.es_ = E.mean(0), np.maximum(E.std(0), 1e-12)
        D = (E - self.em_) / self.es_

        n, p = D.shape
        coef, b0 = np.zeros(p), float(np.log(y.mean()))
        penalty = lambda c: self.alpha * (self.l1_ratio * np.abs(c).sum()
                                          + 0.5 * (1 - self.l1_ratio) * (c @ c))

        def objective(b0_, c):
            eta = np.clip(b0_ + D @ c, -30, 30)
            return np.mean(y * np.exp(-eta) + eta) + penalty(c)

        obj = objective(b0, coef)
        solver = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, fit_intercept=True,
                            warm_start=True, max_iter=2000, tol=1e-5, selection="cyclic")
        for _ in range(self.max_outer):
            eta = np.clip(b0 + D @ coef, -30, 30)
            solver.fit(D, eta + y * np.exp(-eta) - 1.0)
            new_c, new_b = solver.coef_.copy(), float(solver.intercept_)
            step, improved = 1.0, False
            while step > 1e-4:
                c_try = coef + step * (new_c - coef)
                b_try = b0 + step * (new_b - b0)
                cand = objective(b_try, c_try)
                if cand < obj:
                    improved = True
                    break
                step /= 2
            if not improved:
                break
            done = obj - cand < 1e-7
            coef, b0, obj = c_try, b_try, cand
            if done:
                break
        self.coef_, self.intercept_ = coef, b0
        return self

    def _raw(self, X, V):
        D = (self._expand(X, V) - self.em_) / self.es_
        return np.exp(np.clip(self.intercept_ + D @ self.coef_, -30, 30))


# ------------------------------------------------------------------------------------
# Boosted trees
# ------------------------------------------------------------------------------------
_XGB_SCALE = 1e4     # variances near 1e-4 sit awkwardly close to float32 quantisation of the sketch


def _gbt_matrix(X, V):
    F = np.asarray(X, float)[:, :N_HIST].copy()
    F[:, _VAR_IDX] = F[:, _VAR_IDX] * _XGB_SCALE
    return F if V is None else np.column_stack([F, V * _XGB_SCALE])


def _xgb_params(hp, seed):
    return {"objective": "reg:gamma", "tree_method": "hist", "nthread": 1, "seed": int(seed),
            "max_depth": int(hp["max_depth"]), "eta": float(hp["eta"]),
            "min_child_weight": float(hp["min_child_weight"]), "subsample": float(hp["subsample"]),
            "colsample_bytree": float(hp["colsample_bytree"]), "lambda": float(hp["lambda"]),
            "verbosity": 0}


class XGBModel(_Base):
    """Bagged reg:gamma boosters.  The forecast is the arithmetic mean over the bag."""

    def __init__(self, hp, n_bag=5):
        self.hp, self.n_bag = dict(hp), n_bag

    def fit(self, X, V, y, floor, seed=0, **_):
        self.uses_v, self.floor = V is not None, floor
        dtrain = xgb.DMatrix(_gbt_matrix(X, V), label=y)
        params = {**self.hp}
        rounds = int(params.pop("rounds"))
        self.boosters_ = []
        for b in range(self.n_bag):
            base = {**_xgb_params(params, seed + b), "base_score": float(y.mean())}
            self.boosters_.append(xgb.train(base, dtrain, num_boost_round=rounds))
        return self

    def _raw(self, X, V):
        F = _gbt_matrix(X, V)
        return np.mean([b.inplace_predict(F) for b in self.boosters_], axis=0).astype(float)


def xgb_validate(X_tr, V_tr, y_tr, X_va, V_va, y_va, hp, max_rounds, patience, floor, seed):
    """Early-stopped validation fit: returns (validation QLIKE, best rounds)."""
    dtr = xgb.DMatrix(_gbt_matrix(X_tr, V_tr), label=y_tr)
    dva = xgb.DMatrix(_gbt_matrix(X_va, V_va), label=y_va)
    params = {**_xgb_params(hp, seed), "base_score": float(y_tr.mean()), "eval_metric": "gamma-nloglik"}
    booster = xgb.train(params, dtr, num_boost_round=max_rounds, evals=[(dva, "val")],
                        early_stopping_rounds=patience, verbose_eval=False)
    rounds = booster.best_iteration + 1
    pred = booster.predict(dva, iteration_range=(0, rounds))
    return float(np.mean(qlike(y_va, np.maximum(pred, floor)))), rounds


class HGBModel(_Base):
    """scikit-learn histogram boosting with Gamma loss; the alternative tree learner in WP7."""

    def __init__(self, hp, n_bag=1):
        self.hp, self.n_bag = dict(hp), n_bag

    def fit(self, X, V, y, floor, seed=0, **_):
        self.uses_v, self.floor = V is not None, floor
        hp = {**self.hp}
        rounds = int(hp.pop("rounds"))
        F = _gbt_matrix(X, V)
        self.models_ = [HistGradientBoostingRegressor(loss="gamma", max_iter=rounds, early_stopping=False,
                                                      random_state=seed + b, **hp).fit(F, y)
                        for b in range(self.n_bag)]
        return self

    def _raw(self, X, V):
        F = _gbt_matrix(X, V)
        return np.mean([m.predict(F) for m in self.models_], axis=0)


# ------------------------------------------------------------------------------------
# Log-target sensitivity learners (read separately: exp(E[log Y]) is not E[Y])
# ------------------------------------------------------------------------------------
class LogHARModel(_Base):
    def __init__(self, smear=False):
        self.smear = smear

    def _design(self, X, V):
        F = np.log(np.maximum(X[:, _HAR_IDX], 0.0) + _LOG_EPS)
        F = F if V is None else np.column_stack([F, np.log(V)])
        return np.column_stack([np.ones(len(F)), F])

    def fit(self, X, V, y, floor, **_):
        self.uses_v, self.floor = V is not None, floor
        A = self._design(X, V)
        self.beta_ = np.linalg.lstsq(A, np.log(y), rcond=None)[0]
        self.smear_ = float(np.mean(np.exp(np.log(y) - A @ self.beta_))) if self.smear else 1.0
        return self

    def _raw(self, X, V):
        return self.smear_ * np.exp(self._design(X, V) @ self.beta_)


class LogXGBModel(_Base):
    def __init__(self, hp, n_bag=5):
        self.hp, self.n_bag = dict(hp), n_bag

    def fit(self, X, V, y, floor, seed=0, **_):
        self.uses_v, self.floor = V is not None, floor
        params = {**self.hp}
        rounds = int(params.pop("rounds"))
        dtrain = xgb.DMatrix(_gbt_matrix(X, V), label=np.log(y))
        self.boosters_ = []
        for b in range(self.n_bag):
            base = {**_xgb_params(params, seed + b), "objective": "reg:squarederror"}
            self.boosters_.append(xgb.train(base, dtrain, num_boost_round=rounds))
        return self

    def _raw(self, X, V):
        F = _gbt_matrix(X, V)
        return np.exp(np.mean([b.inplace_predict(F) for b in self.boosters_], axis=0).astype(float))


# ------------------------------------------------------------------------------------
# Benchmarks (not paired information-set comparisons)
# ------------------------------------------------------------------------------------
class IVOnlyModel(_Base):
    """E[Y] = a + b V, the informational benchmark."""
    def fit(self, X, V, y, floor, **_):
        self.uses_v, self.floor = True, floor
        self.m_, self.s_ = V.mean(), V.std()
        self.beta_ = fit_gamma_identity(np.column_stack([np.ones(len(V)), (V - self.m_) / self.s_]), y, floor)
        return self

    def _raw(self, X, V):
        return self.beta_[0] + self.beta_[1] * (V - self.m_) / self.s_


class RollingRVModel(_Base):
    """E[Y] = c * rv_m, with c the training ratio of means (persistence benchmark)."""
    def fit(self, X, V, y, floor, **_):
        self.uses_v, self.floor = False, floor
        self.c_ = y.mean() / X[:, HIST_COLS.index("rv_m")].mean()
        return self

    def _raw(self, X, V):
        return self.c_ * X[:, HIST_COLS.index("rv_m")]


class HistMeanModel(_Base):
    def fit(self, X, V, y, floor, **_):
        self.uses_v, self.floor = False, floor
        self.mean_ = y.mean()
        return self

    def _raw(self, X, V):
        return np.full(len(X), self.mean_)


# ------------------------------------------------------------------------------------
# Tuning
# ------------------------------------------------------------------------------------
def xgb_candidates(cfg: dict, seed: int, family: str = "xgb") -> list[dict]:
    """Fixed candidate list drawn once; both information sets see exactly this list."""
    fam = cfg["families"][family]
    keys = sorted(fam["space"])
    grid = list(itertools.product(*(fam["space"][k] for k in keys)))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(grid))[: fam["n_candidates"]]
    return [dict(zip(keys, grid[i])) for i in order]


def _hgb_validate(X_tr, V_tr, y_tr, X_va, V_va, y_va, hp, max_rounds, floor, seed):
    """Validation curve over rounds through staged predictions; budget matches xgb's."""
    F_tr, F_va = _gbt_matrix(X_tr, V_tr), _gbt_matrix(X_va, V_va)
    model = HistGradientBoostingRegressor(loss="gamma", max_iter=max_rounds, early_stopping=False,
                                          random_state=seed, **hp).fit(F_tr, y_tr)
    scores = [np.mean(qlike(y_va, np.maximum(p, floor))) for p in model.staged_predict(F_va)]
    best = int(np.argmin(scores))
    return float(scores[best]), best + 1


def validation_table(family: str, X_tr, V_tr, y_tr, X_va, V_va, y_va, cfg: dict, seed: int) -> list[dict]:
    """Score every candidate on the validation block, best first (ties keep list order)."""
    floor = forecast_floor(y_tr, cfg)
    fam = cfg["families"].get(family, {})
    rows = []
    if family == "enet":
        for alpha, l1 in itertools.product(fam["alphas"], fam["l1_ratios"]):
            hp = {"alpha": alpha, "l1_ratio": l1}
            model = ENetModel(alpha, l1, fam["max_outer"]).fit(X_tr, V_tr, y_tr, floor)
            rows.append({"hp": hp, "val_qlike": float(np.mean(qlike(y_va, model.predict(X_va, V_va))))})
    elif family == "xgb":
        for hp in xgb_candidates(cfg, cfg["seed"]):
            score, rounds = xgb_validate(X_tr, V_tr, y_tr, X_va, V_va, y_va, hp, fam["max_rounds"],
                                         fam["patience"], floor, seed)
            rows.append({"hp": {**hp, "rounds": rounds}, "val_qlike": score})
    elif family == "hgb":
        for hp in xgb_candidates(cfg, cfg["seed"], "hgb"):
            score, rounds = _hgb_validate(X_tr, V_tr, y_tr, X_va, V_va, y_va, hp, fam["max_rounds"], floor, seed)
            rows.append({"hp": {**hp, "rounds": rounds}, "val_qlike": score})
    elif family in NEURAL_FAMILIES:
        from .neural import NeuralModel
        for hp in xgb_candidates(cfg, cfg["seed"], family):
            model = NeuralModel(family, hp, cfg["neural"]).fit(X_tr, V_tr, y_tr, floor, seed=seed,
                                                                X_val=X_va, V_val=V_va, y_val=y_va)
            rows.append({"hp": {**hp, "epochs": model.best_epochs_}, "val_qlike": model.best_val_})
    elif family == "xgb_log":
        for hp in xgb_candidates(cfg, cfg["seed"]):
            model = LogXGBModel({**hp, "rounds": 300}, n_bag=1).fit(X_tr, V_tr, y_tr, floor, seed=seed)
            rows.append({"hp": {**hp, "rounds": 300},
                         "val_qlike": float(np.mean(qlike(y_va, model.predict(X_va, V_va))))})
    else:
        rows.append({"hp": {}, "val_qlike": float("nan")})
    return sorted(rows, key=lambda r: (np.isnan(r["val_qlike"]), r["val_qlike"]))


def make_model(family: str, hp: dict, cfg: dict):
    fam = cfg["families"].get(family.replace("_log", ""), {})
    if family in NEURAL_FAMILIES:
        from .neural import NeuralModel
        return NeuralModel(family, hp, cfg["neural"])
    if family == "har":
        return HARModel("qlike")
    if family == "har_ols":
        return HARModel("mse")
    if family == "enet":
        return ENetModel(hp["alpha"], hp["l1_ratio"], fam["max_outer"])
    if family == "xgb":
        return XGBModel(hp, fam["n_bag"])
    if family == "hgb":
        return HGBModel(hp, fam["n_bag"])
    if family == "har_log":
        return LogHARModel()
    if family == "xgb_log":
        return LogXGBModel(hp, cfg["families"]["xgb"]["n_bag"])
    if family == "ivonly":
        return IVOnlyModel()
    if family == "rwrv":
        return RollingRVModel()
    if family == "histmean":
        return HistMeanModel()
    raise KeyError(family)


def fit_pair(family: str, X, V, y, hp0: dict, hp1: dict, cfg: dict, seed: int):
    """Fit f0(X) and f1(X, V) on identical rows under the same floor, objective and seeds."""
    floor = forecast_floor(y, cfg)
    m0 = make_model(family, hp0, cfg).fit(X, None, y, floor, seed=seed)
    m1 = make_model(family, hp1, cfg).fit(X, V, y, floor, seed=seed)
    return m0, m1
