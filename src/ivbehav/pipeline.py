"""Walk-forward drivers: tuning, fitting, and refit-based runs (placebo, bootstrap, variants)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .conditional import LocationScaleLaw, fit_laws
from .config import atomic_write, check_config_stamp, out_dir, resolve_spec, write_json
from .core import evaluate_fold, match_rules
from .data import SpecData, load_spec_data
from .folds import Fold, make_folds, oos_positions
from .inference import mbb_indices
from .models import SEQUENCE_FAMILIES, fit_pair, forecast_floor, make_model, validation_table
from .stats import stitch, stitched_stats
from .teacher import TeacherStats


def parts_for(family: str) -> tuple:
    """Sequence models read the return path, not the summaries, so summary-based response
    functions (P2) are undefined for them; displacement and matched pairs still apply."""
    if family in SEQUENCE_FAMILIES:
        return ("origin", "pairs")
    return ("origin", "response", "pairs")


def setup(cfg: dict, asset: str, spec: str = "primary", wf_override: dict | None = None):
    check_config_stamp(cfg)
    data = load_spec_data(cfg, asset, spec)
    wf = {**resolve_spec(cfg, spec), **(wf_override or {})}
    folds = make_folds(data, cfg, wf)
    return data, folds, oos_positions(folds)


# ------------------------------------------------------------------------------------
# Tuning
# ------------------------------------------------------------------------------------
def _tune_fold(data: SpecData, fold: Fold, family: str, cfg: dict) -> dict:
    tr, va = fold.inner_train_idx, fold.val_idx
    X_tr, V_tr, y_tr = data.X[tr], data.V[tr], data.y[tr]
    X_va, V_va, y_va = data.X[va], data.V[va], data.y[va]
    seed = cfg["seed"]
    return {
        "fold": fold.k, "fit_date": str(fold.fit_date.date()),
        "n_inner": int(len(tr)), "n_val": int(len(va)),
        "f0": validation_table(family, X_tr, None, y_tr, X_va, None, y_va, cfg, seed),
        "f1": validation_table(family, X_tr, V_tr, y_tr, X_va, V_va, y_va, cfg, seed),
    }


def tune_family(data, folds, family, cfg, n_jobs) -> dict[int, dict]:
    """Validation tables at each retuning fold, identical budget for both information sets."""
    jobs = [f for f in folds if f.retune]
    tables = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_tune_fold)(data, f, family, cfg) for f in jobs)
    return {t["fold"]: t for t in tables}


def hp_for_folds(folds, tables: dict[int, dict], rank: int = 0, tolerance: float | None = None):
    """Hyperparameters per fold, carried forward between retunes.

    ``rank`` selects the k-th best validation configuration (near-equivalent fits); with
    ``tolerance`` a configuration further than that from the best falls back to the best.
    """
    def pick(rows):
        i = min(rank, len(rows) - 1)
        if tolerance is not None and rows[i]["val_qlike"] > (1 + tolerance) * rows[0]["val_qlike"]:
            i = 0
        return rows[i]["hp"]

    out, current = {}, None
    for f in folds:
        if f.k in tables:
            current = (pick(tables[f.k]["f0"]), pick(tables[f.k]["f1"]))
        out[f.k] = current
    return out


# ------------------------------------------------------------------------------------
# Primary fit store
# ------------------------------------------------------------------------------------
def fit_path(cfg, asset, spec, family, k) -> Path:
    return out_dir(cfg, asset, spec, "fits", family) / f"fold_{k:03d}.joblib"


def load_pair(cfg, asset, spec, family, k):
    return joblib.load(fit_path(cfg, asset, spec, family, k))


def load_tables(cfg, asset, spec, family) -> dict[int, dict]:
    path = out_dir(cfg, asset, spec, "fits", family) / "tuning.json"
    with open(path) as fh:
        return {int(k): v for k, v in json.load(fh).items()}


def _fit_store(data, fold, family, hp0, hp1, cfg, path, force):
    if force or not path.exists():
        tr = fold.train_idx
        m0, m1 = fit_pair(family, data.X[tr], data.V[tr], data.y[tr], hp0, hp1, cfg, cfg["seed"])
        atomic_write(path, lambda tmp: joblib.dump({"m0": m0, "m1": m1, "hp0": hp0, "hp1": hp1}, tmp, compress=3))
    pair = joblib.load(path)
    te = fold.test_idx
    return pd.DataFrame({
        "pos": te, "date": data.dates[te], "fold": fold.k, "y": data.y[te], "n_obs": data.n_obs[te],
        "n_train": len(fold.train_idx),
        "f0": pair["m0"].predict(data.X[te]), "f1": pair["m1"].predict(data.X[te], data.V[te]),
        "floor": pair["m0"].floor,
    })


def run_forecasts(cfg, asset, spec, family, n_jobs, force=False) -> pd.DataFrame:
    data, folds, _ = setup(cfg, asset, spec)
    fit_dir = out_dir(cfg, asset, spec, "fits", family)
    tuning_path = fit_dir / "tuning.json"
    if force or not tuning_path.exists():
        write_json(tuning_path, tune_family(data, folds, family, cfg, n_jobs))
    tables = load_tables(cfg, asset, spec, family)
    hp = hp_for_folds(folds, tables)
    rows = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_fit_store)(data, f, family, *hp[f.k], cfg, fit_path(cfg, asset, spec, family, f.k), force)
        for f in folds)
    return pd.concat(rows, ignore_index=True)


def _fit_benchmark(data, fold, family, cfg):
    tr, te = fold.train_idx, fold.test_idx
    floor = forecast_floor(data.y[tr], cfg)
    model = make_model(family, {}, cfg).fit(data.X[tr], data.V[tr], data.y[tr], floor)
    return pd.DataFrame({"pos": te, "date": data.dates[te], "fold": fold.k, "y": data.y[te],
                         "f": model.predict(data.X[te], data.V[te])})


def run_benchmark(cfg, asset, spec, family, n_jobs) -> pd.DataFrame:
    data, folds, _ = setup(cfg, asset, spec)
    rows = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_fit_benchmark)(data, f, family, cfg) for f in folds)
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------------------------
# Placebo signals
# ------------------------------------------------------------------------------------
def placebo_path(data: SpecData, fold: Fold, law_a: LocationScaleLaw, oos_pos, kind, rng, cfg) -> np.ndarray:
    """Outcome-independent V path for one fold's fit: training rows and all later OOS origins.

    The generator is estimated on this fold's training rows only.  Residuals never touch the
    future of Y given X, so any behavior the refit shows on them is an artifact.
    """
    rows = np.unique(np.concatenate([fold.train_idx, oos_pos[oos_pos >= fold.test_idx[0]]]))
    path = np.full(data.V.shape, np.nan)
    block = cfg["placebo"]["block"] if kind == "cond_block" else None
    path[rows] = law_a.resample_path(data.X[rows], rng, block)
    return path


def displaced_path(data: SpecData, lag: int) -> np.ndarray:
    """V_{t-L}: strictly past, with the actual serial structure of IV but misaligned to t."""
    path = np.full(data.V.shape, np.nan)
    path[lag:] = data.V[:-lag]
    return path


# ------------------------------------------------------------------------------------
# Refit-based runs sharing one evaluation path
# ------------------------------------------------------------------------------------
_LAW_CACHE: dict = {}


def fold_laws(data: SpecData, fold: Fold, cfg: dict):
    """Observed-V laws of a fold.  They depend only on its training rows, so a worker running
    many refits over the same folds fits them once."""
    tr = fold.train_idx
    key = (cfg["profile"], data.asset, data.spec, fold.k, int(tr[0]), int(tr[-1]), len(tr))
    if key not in _LAW_CACHE:
        _LAW_CACHE[key] = fit_laws(data.X[tr], data.V[tr], cfg)
    return _LAW_CACHE[key]


_RULE_CACHE: dict = {}


def fold_rules(data: SpecData, fold: Fold, cfg: dict) -> dict:
    """Matching thresholds of the observed fold.  Refits keep them fixed: duplicates in a block
    resample would otherwise shrink the k-NN caliper, and pairs are still rebuilt each time."""
    tr = fold.train_idx
    key = (cfg["profile"], data.asset, data.spec, fold.k, int(tr[0]), int(tr[-1]), len(tr))
    if key not in _RULE_CACHE:
        law_a, _, support = fold_laws(data, fold, cfg)
        _RULE_CACHE[key] = match_rules(data.X[tr], law_a, support, cfg)
    return _RULE_CACHE[key]


def _placebo_fold(data, fold, family, hp1, cfg, asset, spec, oos_pos, kind, draw, parts):
    pair = load_pair(cfg, asset, spec, family, fold.k)
    tr = fold.train_idx
    seed = int(cfg["seed"])
    observed = fold_laws(data, fold, cfg)
    if kind.startswith("displaced_"):
        path = displaced_path(data, int(kind.split("_")[1]))
        train_rows = tr[np.isfinite(path[tr])]
        law_a = LocationScaleLaw(cfg["conditional"]["n_draws"]).fit(data.X[train_rows], path[train_rows])
        laws = (law_a, observed[1], observed[2])
    else:
        rng = np.random.default_rng([seed, draw, fold.k])
        laws = observed
        path = placebo_path(data, fold, laws[0], oos_pos, kind, rng, cfg)
        train_rows = tr
    floor = forecast_floor(data.y[tr], cfg)
    m1 = make_model(family, hp1, cfg).fit(data.X[train_rows], path[train_rows], data.y[train_rows],
                                          floor, seed=seed + 7919 * (draw + 1))
    return evaluate_fold(data, fold, pair["m0"], m1, cfg, oos_pos, laws=laws, V_eval=path, parts=parts,
                         with_kernel=False, rules=fold_rules(data, fold, cfg))


def run_placebo(cfg, asset, spec, family, kind, draw, parts=None) -> dict:
    """Refit f1 on a placebo signal with the observed f1 complexity; f0 is unchanged."""
    parts = parts or parts_for(family)
    data, folds, oos_pos = setup(cfg, asset, spec)
    tables = load_tables(cfg, asset, spec, family)
    hp = hp_for_folds(folds, tables)
    results = [_placebo_fold(data, f, family, hp[f.k][1], cfg, asset, spec, oos_pos, kind, draw, parts)
               for f in folds]
    out = stitched_stats(results, oos_pos, cfg)
    if "origin" in parts:
        # Addendum C statistics ride along, so later placebo runs need no second refit for them.
        out.update(TeacherStats(stitch(results, "origin"), oos_pos).point())
    return out


def run_placebo_teacher(cfg, asset, spec, family, kind, draw) -> dict:
    """Addendum C on one placebo draw: the same placebo signal and f1 refit as ``run_placebo``
    (identical seeds), evaluated on the per-origin decomposition only."""
    data, folds, oos_pos = setup(cfg, asset, spec)
    hp = hp_for_folds(folds, load_tables(cfg, asset, spec, family))
    results = [_placebo_fold(data, f, family, hp[f.k][1], cfg, asset, spec, oos_pos, kind, draw, ("origin",))
               for f in folds]
    return TeacherStats(stitch(results, "origin"), oos_pos).point()


def _refit_fold(data, fold, family, hp0, hp1, cfg, oos_pos, seed, block, parts, with_kernel):
    tr = fold.train_idx
    observed = fold_laws(data, fold, cfg)
    if block is None:
        rows, laws = tr, observed
    else:
        rng = np.random.default_rng([seed, fold.k])
        rows = tr[mbb_indices(len(tr), block, rng)]
        laws = fit_laws(data.X[rows], data.V[rows], cfg, bandwidth=observed[1].h_)
    m0, m1 = fit_pair(family, data.X[rows], data.V[rows], data.y[rows], hp0, hp1, cfg, seed)
    res = evaluate_fold(data, fold, m0, m1, cfg, oos_pos, laws=laws, parts=parts, with_kernel=with_kernel,
                        rules=fold_rules(data, fold, cfg))
    if family == "har":
        res["coef"] = {"fold": fold.k, **{f"b0_{k}": v for k, v in m0.coef_std().items()},
                       **{f"b1_{k}": v for k, v in m1.coef_std().items()}}
    return res


def run_refit(cfg, data, folds, oos_pos, family, hp, seed, block=None, parts=None,
              with_kernel=False, n_jobs=1) -> tuple[dict, list[dict]]:
    """Refit f0, f1 and the laws at every fold; block resamples the training rows when given."""
    parts = parts or parts_for(family)
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_refit_fold)(data, f, family, *hp[f.k], cfg, oos_pos, seed, block, parts, with_kernel)
        for f in folds)
    return stitched_stats(results, oos_pos, cfg), results


def run_refit_teacher(cfg, asset, spec, family, draw, block) -> dict:
    """Addendum C on one refit-bootstrap draw.  Seeds, training resamples and the combined date
    weights are those of ``scripts/10_refit_bootstrap.py``, so the draw is the same refit."""
    from .inference import mbb_weights
    data, folds, oos_pos = setup(cfg, asset, spec)
    hp = hp_for_folds(folds, load_tables(cfg, asset, spec, family))
    seed = cfg["seed"] + 100_000 + draw
    _, results = run_refit(cfg, data, folds, oos_pos, family, hp, seed, block=block, parts=("origin",))
    stat = TeacherStats(stitch(results, "origin"), oos_pos)
    rng = np.random.default_rng([cfg["seed"], draw])
    combined = stat.compute(mbb_weights(len(oos_pos), cfg["bootstrap"]["primary_block"], rng))
    return {"draw": draw, "fit_only": stat.compute(), "combined": combined}
