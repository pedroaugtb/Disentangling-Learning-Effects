"""Shared bookkeeping for the teacher / input collections (scripts 14 and 15).

Reads the tables written by script 12 and the hierarchy of script 11; no estimation here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import RESULTS_DIR

KEYS = ["gain_total", "gain_teacher", "gain_input", "rel_gain_total", "rel_gain_teacher", "rel_gain_input",
        "gain_teacher_recal", "gain_input_recal", "gain_total_recal",
        "gain_teacher_raw_on_recal", "gain_input_raw_on_recal",
        "gain_teacher_lawB", "gain_input_lawB", "gain_teacher_lawA_on_supB", "gain_input_lawA_on_supB",
        "gain_teacher_excovid", "gain_input_excovid", "gain_teacher_supx", "gain_input_supx",
        "gain_teacher_mse", "gain_input_mse", "L0", "LP", "L1"]


def _read(path: Path):
    return pd.read_csv(path) if path.exists() else None


def _row(table, statistic, **conds):
    if table is None:
        return None
    m = table["statistic"] == statistic
    for k, v in conds.items():
        m &= table[k] == v
    rows = table[m]
    return rows.iloc[0] if len(rows) else None


def p2_status(tdir: Path, family: str, alpha: float) -> dict:
    """Features claimed under the P2 rule of the hierarchy (dS interval and T placebo, Holm)."""
    h = _read(tdir / "hierarchy.csv")
    if h is None:
        return {"p2_claimed_features": np.nan}
    p2 = h[(h["group"] == "P2") & (h["family"] == family)]
    shape = h[(h["group"] == "P2_shape") & (h["family"] == family)].set_index("statistic")
    n = 0
    for _, r in p2.iterrows():
        feat = r["statistic"].split(":")[0]
        t = shape.loc[f"{feat}:T"] if f"{feat}:T" in shape.index else None
        excl = (r["ci_lo"] > 0) or (r["ci_hi"] < 0)
        ok = excl and r["boot_p_holm"] < alpha and t is not None and t["placebo_p_holm"] < alpha
        n += int(bool(ok))
    return {"p2_claimed_features": n, "p2_n_features": len(p2)}


def cell(profile: str, asset: str, spec: str, family: str, block: int) -> dict | None:
    tdir = RESULTS_DIR / profile / asset / spec / "tables"
    boot = _read(tdir / f"teacher_boot_{family}.csv")
    if boot is None:
        return None
    placebo = _read(tdir / f"teacher_placebo_{family}.csv")
    refit = _read(tdir / f"teacher_refit_{family}.csv")
    out = {"profile": profile, "asset": asset, "spec": spec, "family": family}
    for key in KEYS:
        stat = f"ti:{key}"
        b = _row(boot, stat, block=block)
        if b is None:
            continue
        out.update({key: b["estimate"], f"{key}_lo": b["ci_lo"], f"{key}_hi": b["ci_hi"],
                    f"{key}_q95": b["q95"], f"{key}_p": b["p_zero"]})
        pl = _row(placebo, stat)
        if pl is not None:
            out.update({f"{key}_placebo_mean": pl["placebo_mean"], f"{key}_placebo_q95": pl["placebo_q95"],
                        f"{key}_placebo_p_upper": pl["p_upper"], f"{key}_placebo_p_lower": pl["p_lower"]})
        rc = _row(refit, stat, source="combined")
        if rc is not None:
            out.update({f"{key}_comb_lo": rc["ci_lo"], f"{key}_comb_hi": rc["ci_hi"],
                        f"{key}_comb_q95": rc["q95"], f"{key}_comb_p": rc["p_zero"], f"{key}_comb_n": rc["n"]})
    return out


def sim_false_detection(summary: pd.DataFrame | None, proxies: list[str], rule: str) -> float:
    if summary is None or rule not in summary:
        return float("nan")
    s = summary[(summary["mode"] == "redundant") & (summary["law"] == "A_wellspec") & summary["learner"].isin(proxies)]
    return float(s[rule].max()) if len(s) else float("nan")


def _fmt(x, digits=4):
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (float, np.floating)):
        return "" if not np.isfinite(x) else f"{x:.{digits}g}"
    return str(x)


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    cols = [c for c in cols if c in df]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df[cols].iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)
