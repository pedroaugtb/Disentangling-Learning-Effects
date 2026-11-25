"""Monte Carlo audit of the teacher / input decomposition (PREANALYSIS, Addendum C).

Same DGP, learners and laws as ``simulation.py``.  In the DGP the ideal projection is exact:
E[m1 | Z] = m0, so the ideal teacher gain is zero and the ideal input gain is the whole
population gain of observing V.  A fitted learner can still show a teacher gain, because V
changes what it learns about the history (the redundant design measures this channel alone).

Per replicate and learner the audit reports, for the true law and the estimated law A:
  - the sample gains on the test segment and their block-bootstrap intervals;
  - the population gains of the same fitted pair, from an independent long path projected
    under the true law (the target the intervals should cover);
  - for law A, the placebo p-values of both gains, so the full C1 / C2 rules can be scored.
"""
from __future__ import annotations

import numpy as np

from .conditional import LocationScaleLaw, project
from .inference import block_bootstrap, boot_p_two_sided, percentile_ci
from .models import forecast_floor, qlike
from .simulation import Calibration, TrueLaw, make_learners, simulate

_MODE_ID = {"redundant": 0, "constant": 1, "state": 2}


def _gain_fn(q0, qP, q1):
    def fn(w):
        tot = w.sum()
        L0, LP, L1 = (q0 * w).sum() / tot, (qP * w).sum() / tot, (q1 * w).sum() / tot
        return {"teacher": L0 - LP, "input": LP - L1, "total": L0 - L1}
    return fn


def teacher_replicate(rep: int, mode: str, b_bar: float, cal: Calibration, cfg: dict, tsim: dict,
                      seed: int, with_placebo: bool) -> list[dict]:
    s, c = cfg["simulation"], cfg["conditional"]
    rng = np.random.default_rng([seed, rep, 4243, int(b_bar * 1000), _MODE_ID[mode]])
    args = (cal, mode, b_bar, s["state_kappa"], s["gamma_shape"], s["overlap"], s["rho_eps"])
    n_total = s["n_train"] + s["overlap"] + s["n_test"]
    path = simulate(n_total, *args, rng)
    tr, te = slice(0, s["n_train"]), slice(s["n_train"] + s["overlap"], n_total)
    train = {k: v[tr] for k, v in path.items()}
    test = {k: v[te] for k, v in path.items()}
    pop = simulate(tsim["n_pop"], *args, rng)

    floor = forecast_floor(train["Y"], cfg)
    law_a = LocationScaleLaw(c["n_draws"]).fit(train["X"], train["V"])
    laws = {"true": TrueLaw(cal, c["n_draws"]), "A_wellspec": law_a}
    pv, pw = TrueLaw(cal, tsim["pop_grid"]).draws(pop["X"])
    y, n_test = test["Y"], len(test["Y"])
    ideal_input = float(np.mean(qlike(pop["Y"], pop["m0"]) - qlike(pop["Y"], pop["m1"])))
    draws_a = law_a.draws(test["X"])
    block = s["bootstrap_block"]

    rows = []
    for name, factory in make_learners(cal, cfg, floor).items():
        m0 = factory().fit(train["X"], None, train["Y"], floor, seed=rep)
        m1 = factory().fit(train["X"], train["V"], train["Y"], floor, seed=rep)
        f0, f1 = m0.predict(test["X"]), m1.predict(test["X"], test["V"])
        q0, q1 = qlike(y, f0), qlike(y, f1)
        lf0, lf1 = m0.predict(pop["X"]), m1.predict(pop["X"], pop["V"])
        lP = project(m1, pop["X"], pv, pw)
        lq0, lqP, lq1 = qlike(pop["Y"], lf0), qlike(pop["Y"], lP), qlike(pop["Y"], lf1)
        pop_gain = {"teacher": float(np.mean(lq0 - lqP)), "input": float(np.mean(lqP - lq1)),
                    "total": float(np.mean(lq0 - lq1))}
        for law_name, law in laws.items():
            vals, wts = draws_a if law_name == "A_wellspec" else law.draws(test["X"])
            qP = qlike(y, project(m1, test["X"], vals, wts))
            fn = _gain_fn(q0, qP, q1)
            point = fn(np.ones(n_test))
            boot = block_bootstrap(fn, n_test, block, s["bootstrap_n"], rep + 31)
            row = {"rep": rep, "mode": mode, "b_bar": b_bar, "learner": name, "law": law_name,
                   "ideal_input": ideal_input}
            for key in ("teacher", "input", "total"):
                lo, hi = percentile_ci(boot[key])
                row.update({key: point[key], f"{key}_lo": lo, f"{key}_hi": hi,
                            f"{key}_p": boot_p_two_sided(boot[key]),
                            f"{key}_q95": float(np.nanquantile(boot[key], 0.95)),
                            f"pop_{key}": pop_gain[key],
                            f"{key}_covers_pop": float(lo <= pop_gain[key] <= hi)})
            row["L0"] = float(q0.mean())
            if law_name == "A_wellspec" and with_placebo and name in tsim["placebo_learners"]:
                pt, pi = [], []
                for d in range(tsim["placebo_draws"]):
                    V_tr = law_a.resample_path(train["X"], rng, cfg["placebo"]["block"])
                    V_te = law_a.resample_path(test["X"], rng, cfg["placebo"]["block"])
                    mp = factory().fit(train["X"], V_tr, train["Y"], floor, seed=rep + 1000 + d)
                    qPp = qlike(y, project(mp, test["X"], vals, wts))
                    q1p = qlike(y, mp.predict(test["X"], V_te))
                    pt.append(float(np.mean(q0 - qPp)))
                    pi.append(float(np.mean(qPp - q1p)))
                pt, pi = np.asarray(pt), np.asarray(pi)
                row["placebo_p_teacher"] = float((np.sum(pt >= point["teacher"]) + 1) / (len(pt) + 1))
                row["placebo_p_input"] = float((np.sum(pi >= point["input"]) + 1) / (len(pi) + 1))
                row["placebo_mean_teacher"], row["placebo_mean_input"] = float(pt.mean()), float(pi.mean())
            rows.append(row)
    return rows


def summarize(reps, alpha: float = 0.05, margin_rel: float = 0.05):
    """Operating characteristics per design, learner and law."""
    import pandas as pd
    keys = ["mode", "b_bar", "learner", "law"]
    r = reps.copy()
    r["detect_input_pos"] = (r["input_p"] < alpha) & (r["input"] > 0)
    r["detect_teacher_pos"] = (r["teacher_p"] < alpha) & (r["teacher"] > 0)
    r["detect_teacher_neg"] = (r["teacher_p"] < alpha) & (r["teacher"] < 0)
    r["noninferior_margin"] = r["input_q95"] < margin_rel * r["L0"]   # the C3 bound rules out a large gain
    r["bias_teacher"] = r["teacher"] - r["pop_teacher"]
    r["bias_input"] = r["input"] - r["pop_input"]
    has_pl = r["placebo_p_input"].notna() if "placebo_p_input" in r else pd.Series(False, index=r.index)
    if "placebo_p_input" in r:
        r["detect_C1"] = np.where(has_pl, r["detect_input_pos"] & (r["placebo_p_input"] <= alpha), np.nan)
        r["detect_C2"] = np.where(has_pl, r["detect_teacher_pos"] & (r["placebo_p_teacher"] <= alpha), np.nan)
    g = r.groupby(keys, sort=False)
    cols = ["teacher", "input", "total", "pop_teacher", "pop_input", "pop_total", "ideal_input",
            "bias_teacher", "bias_input", "teacher_covers_pop", "input_covers_pop", "total_covers_pop",
            "detect_input_pos", "detect_teacher_pos", "detect_teacher_neg", "noninferior_margin"]
    if "placebo_p_input" in r:
        cols += ["detect_C1", "detect_C2", "placebo_mean_teacher", "placebo_mean_input"]
    out = g[cols].mean()
    out["teacher_mcsd"], out["input_mcsd"] = g["teacher"].std(), g["input"].std()
    out["n_reps"] = g.size()
    if "placebo_p_input" in r:
        out["n_placebo_reps"] = g["placebo_p_input"].count()
    return out.reset_index()
