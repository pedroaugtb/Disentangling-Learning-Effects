#!/usr/bin/env python3
"""Assemble the prespecified hierarchy (docs/PREANALYSIS.md) into one table per asset.

Bookkeeping only: it reads earlier outputs, applies the Holm families fixed in advance, and
adds no new hypothesis.  Interpretation is deliberately left out of this script.
"""
import _bootstrap  # noqa: F401
import argparse

import numpy as np
import pandas as pd

from ivbehav.config import load_config, table_dir, write_csv, write_manifest
from ivbehav.inference import holm


def _get(table, **conds):
    mask = np.ones(len(table), bool)
    for key, val in conds.items():
        mask &= table[key].to_numpy() == val
    rows = table[mask]
    return rows.iloc[0] if len(rows) else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="primary.yaml")
    ap.add_argument("--asset", default="SPX")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tdir = table_dir(cfg, args.asset, "primary")
    kind = cfg["placebo"]["primary_kind"]
    block = cfg["bootstrap"]["primary_block"]
    rows = []
    for family in cfg["core_families"]:
        placebo = pd.read_csv(tdir / f"placebo_{family}.csv")
        placebo = placebo[placebo["kind"] == kind]
        response_path = tdir / f"response_boot_{family}.csv"
        response = pd.read_csv(response_path) if response_path.exists() else None   # absent for sequence models
        pairs_path = tdir / f"pairs_boot_{family}.csv"
        pairs = pd.read_csv(pairs_path) if pairs_path.exists() else pd.DataFrame(columns=["statistic", "block", "estimate", "ci_lo", "ci_hi", "p_zero"])
        decomp = pd.read_csv(tdir / f"decomp_boot_{family}.csv")
        refit_path = tdir / f"refit_boot_{family}.csv"
        refit = pd.read_csv(refit_path) if refit_path.exists() else None

        def add(group, statistic, table, p_col, tail=None):
            b = _get(table, statistic=statistic, block=block) if table is not None and "block" in table else None
            pl = _get(placebo, statistic=statistic)
            row = {"group": group, "family": family, "statistic": statistic}
            if b is not None:
                row.update(estimate=b["estimate"], ci_lo=b["ci_lo"], ci_hi=b["ci_hi"], boot_p=b["p_zero"])
            if pl is not None:
                row.update(observed=pl["observed"], placebo_mean=pl["placebo_mean"], placebo_q95=pl["placebo_q95"],
                           placebo_p=pl[f"p_{tail or 'abs'}"])
            if refit is not None:
                fit = _get(refit, source="fit_only", statistic=statistic)
                both = _get(refit, source="combined", statistic=statistic)
                if fit is not None:
                    row.update(refit_ci_lo=fit["ci_lo"], refit_ci_hi=fit["ci_hi"])
                if both is not None:
                    row.update(combined_ci_lo=both["ci_lo"], combined_ci_hi=both["ci_hi"], combined_p=both["p_zero"])
            rows.append(row)

        for stat in ("rms_D", "rms_HFS", "rms_ISD", "rms_D_rel", "rms_HFS_rel", "rms_ISD_rel"):
            add("P1_gate", stat, decomp, "p_zero", "upper")
        for feature in (cfg["response"]["primary_features"] if response is not None else []):
            add("P1_gate", f"{feature}:T", response, "p_zero", "upper")
            add("P2", f"{feature}:dS", response, "p_zero", "abs")
            add("P2_shape", f"{feature}:T", response, "p_zero", "upper")
            add("P2_secondary", f"{feature}:dSp", response, "p_zero", "abs")
            add("P2_secondary", f"{feature}:T_shape", response, "p_zero", "upper")
            add("P2_secondary", f"{feature}:S1_high_minus_low", response, "p_zero", "abs")
        add("P3", "pairs:sep_slope_rel", pairs, "p_zero", "abs")
        add("P3", "pairs:dqlike", pairs, "p_zero", "abs")
        add("P3_secondary", "pairs:conc_diff", pairs, "p_zero", "abs")
        add("P3_secondary", "pairs:mean_dlogY", pairs, "p_zero", "abs")
        add("validity", "dqlike", decomp, "p_zero", "abs")
        add("validity", "calib_slope_rel", decomp, "p_zero", "abs")

    table = pd.DataFrame(rows)
    gate_rows = []
    for family in cfg["core_families"]:
        fam = table[(table["family"] == family) & (table["group"] == "P1_gate")].set_index("statistic")
        placebo_ok = bool(fam.loc["rms_D_rel", "placebo_p"] <= 0.05) or any(
            fam.loc[f"{k}:T", "placebo_p"] <= 0.05 for k in cfg["response"]["primary_features"] if f"{k}:T" in fam.index)
        refit_path = tdir / f"refit_boot_{family}.csv"
        repeat_ok = None
        if refit_path.exists():
            refit = pd.read_csv(refit_path)
            q95 = refit[refit["source"] == "repeat"].set_index("statistic")["q95"]
            repeat_ok = bool(fam.loc["rms_D", "observed"] > max(q95["rms_self_f0"], q95["rms_self_f1"]))
        gate_rows.append({"family": family, "beats_placebo": placebo_ok, "beats_repeated_fit": repeat_ok,
                          "p1_pass": bool(placebo_ok and repeat_ok)})
    write_csv(tdir / "p1_gate.csv", pd.DataFrame(gate_rows))
    # Holm within each prespecified family, on the combined-uncertainty p-value when it exists.
    for col in ("boot_p", "combined_p", "placebo_p"):
        if col in table:
            table[f"{col}_holm"] = np.nan
    for group in ("P2", "P2_shape", "P3"):
        mask = table["group"] == group
        for col in ("boot_p", "combined_p", "placebo_p"):
            if col in table and table.loc[mask, col].notna().any():
                table.loc[mask, f"{col}_holm"] = holm(table.loc[mask, col].to_numpy())
    write_csv(tdir / "hierarchy.csv", table)
    write_manifest(cfg, tdir, "11_collect_primary")
    print(f"wrote {tdir / 'hierarchy.csv'}")


if __name__ == "__main__":
    main()
