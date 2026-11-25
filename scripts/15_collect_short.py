#!/usr/bin/env python3
"""Addendum D: post-hoc confirmation of the short-horizon input gain.

Bookkeeping only: applies the rules of docs/PREANALYSIS.md (Addendum D) and
configs/teacher_short.yaml to the outputs of scripts 12 (observed, placebo, refit) and 13.
Writes results/<profile>/teacher_short/:

  claims.csv        D0 (total), D1 (input), D2 (teacher), D3 (input exceeds teacher) for the 12
                    cells: SPX, NDX x h5td, h10td x har, enet, xgb.  Holm across the 12 cells,
                    separately per rule.  A cell without refit draws gets no claim.
  term_structure.csv  relative gains at 5, 10 and 30 trading days, with their intervals
  REPORT.md         the same, readable; no interpretation beyond the rules
"""
import _bootstrap  # noqa: F401
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ivbehav.config import CONFIG_DIR, RESULTS_DIR, load_config, write_csv, write_json
from ivbehav.inference import boot_p_two_sided, holm, percentile_ci
from ivbehav.teacher_report import _read, cell, md_table, sim_false_detection

COMPONENTS = ("total", "input", "teacher")


def combined_draws(profile: str, asset: str, spec: str, family: str) -> pd.DataFrame | None:
    """Refit + date draws of script 12 (one row per draw), with the input-minus-teacher contrast."""
    d = RESULTS_DIR / profile / asset / spec / "refit_ti" / family
    files = sorted(d.glob("b*.json"))
    if not files:
        return None
    rows = [json.load(open(p))["combined"] for p in files]
    draws = pd.DataFrame(rows)
    draws.columns = [c.removeprefix("ti:") for c in draws.columns]
    draws["input_minus_teacher"] = draws["gain_input"] - draws["gain_teacher"]
    draws["rel_input_minus_teacher"] = draws["rel_gain_input"] - draws["rel_gain_teacher"]
    return draws


def _summ(values) -> dict:
    lo, hi = percentile_ci(values)
    return {"lo": lo, "hi": hi, "p": boot_p_two_sided(values), "n": int(np.isfinite(values).sum())}


def claims_table(dcfg: dict, sim: pd.DataFrame | None) -> pd.DataFrame:
    conf = dcfg["confirmatory"]
    profile = load_config(conf["config"])["profile"]
    a, block = conf["alpha"], conf["block"]
    rows = []
    for asset in conf["assets"]:
        for spec in conf["specs"]:
            for family in conf["families"]:
                r = {"asset": asset, "spec": spec, "family": family}
                c = cell(profile, asset, spec, family, block)
                draws = combined_draws(profile, asset, spec, family)
                if c is None or draws is None:
                    r["status"] = "missing " + ("observed" if c is None else "refit draws")
                    rows.append(r)
                    continue
                r.update({"status": "ok", "n_refit_draws": len(draws), "L0": c.get("L0")})
                for comp in COMPONENTS:
                    key = f"gain_{comp}"
                    s = _summ(draws[key].to_numpy(float))
                    r.update({comp: c.get(key), f"{comp}_lo": s["lo"], f"{comp}_hi": s["hi"], f"{comp}_p": s["p"],
                              f"{comp}_rel": c.get(f"rel_{key}"),
                              f"{comp}_placebo_mean": c.get(f"{key}_placebo_mean", np.nan),
                              f"{comp}_placebo_p_upper": c.get(f"{key}_placebo_p_upper", np.nan)})
                s = _summ(draws["input_minus_teacher"].to_numpy(float))
                r.update({"input_minus_teacher": c.get("gain_input") - c.get("gain_teacher"),
                          "imt_lo": s["lo"], "imt_hi": s["hi"], "imt_p": s["p"]})
                for key in ("gain_input_lawB", "gain_input_recal", "gain_input_excovid", "gain_input_supx"):
                    r[key], r[f"{key}_p"] = c.get(key, np.nan), c.get(f"{key}_p", np.nan)
                r["sim_false_C1"] = sim_false_detection(sim, dcfg["simulation_proxy"][family], "detect_C1")
                rows.append(r)
    t = pd.DataFrame(rows)
    ok = t["status"] == "ok"
    if not ok.any():
        return t
    for comp in ("total", "input", "teacher", "imt"):
        t[f"{comp}_p_holm"] = np.nan
        t.loc[ok, f"{comp}_p_holm"] = holm(t.loc[ok, f"{comp}_p"].to_numpy(float))
    t["D0_total_gain"] = ok & (t["total"] > 0) & (t["total_p_holm"] < a)
    fires = ok & (t["input"] > 0) & (t["input_p_holm"] < a) & (t["input_placebo_p_upper"] <= a)
    downgraded = t["sim_false_C1"] > conf["max_false_detection"]
    t["D1_input_claimed"] = fires & ~downgraded
    t["D1_downgraded_by_simulation"] = fires & downgraded
    t["D2_teacher"] = np.select(
        [ok & (t["teacher"] > 0) & (t["teacher_p_holm"] < a) & (t["teacher_placebo_p_upper"] <= a),
         ok & (t["teacher"] < 0) & (t["teacher_p_holm"] < a)],
        ["teaches (beyond placebo)", "harmful reorganisation"], default="not claimed")
    t["D3_input_exceeds_teacher"] = ok & (t["input_minus_teacher"] > 0) & (t["imt_p_holm"] < a)
    return t


def headline(t: pd.DataFrame) -> pd.DataFrame:
    """Per asset and horizon: how many of the three families carry each claim (no pooling)."""
    if "D1_input_claimed" not in t:
        return pd.DataFrame()
    g = t[t["status"] == "ok"].groupby(["asset", "spec"], sort=False)
    out = g.agg(families=("family", "size"), D0_total=("D0_total_gain", "sum"), D1_input=("D1_input_claimed", "sum"),
                D3_input_exceeds_teacher=("D3_input_exceeds_teacher", "sum"),
                D2_teaches=("D2_teacher", lambda s: int((s == "teaches (beyond placebo)").sum())),
                D2_harmful=("D2_teacher", lambda s: int((s == "harmful reorganisation").sum())))
    return out.reset_index()


def term_structure(dcfg: dict) -> pd.DataFrame:
    conf = dcfg["confirmatory"]
    profile = load_config(conf["config"])["profile"]
    rows = []
    for asset in conf["assets"]:
        for spec in dcfg["term_structure_specs"]:
            for family in conf["families"]:
                c = cell(profile, asset, spec, family, conf["block"])
                if c is None:
                    continue
                draws = combined_draws(profile, asset, spec, family)
                r = {"asset": asset, "spec": spec, "family": family,
                     "interval": "refit+date" if draws is not None else "date only"}
                for comp in COMPONENTS:
                    key = f"rel_gain_{comp}"
                    r[f"{comp}_rel"] = c.get(key)
                    if draws is not None:
                        r[f"{comp}_rel_lo"], r[f"{comp}_rel_hi"] = percentile_ci(draws[key].to_numpy(float))
                    else:
                        r[f"{comp}_rel_lo"], r[f"{comp}_rel_hi"] = c.get(f"{key}_lo"), c.get(f"{key}_hi")
                rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--short-config", default="teacher_short.yaml")
    ap.add_argument("--teacher-config", default="teacher.yaml", help="for the simulation summary only")
    args = ap.parse_args()
    dpath = Path(args.short_config) if Path(args.short_config).exists() else CONFIG_DIR / args.short_config
    dtext = dpath.read_text()
    dcfg = yaml.safe_load(dtext)
    dhash = hashlib.sha256(dtext.encode()).hexdigest()[:16]
    profile = load_config(dcfg["confirmatory"]["config"])["profile"]
    outdir = RESULTS_DIR / profile / "teacher_short"
    outdir.mkdir(parents=True, exist_ok=True)

    sim = _read(RESULTS_DIR / profile / "simulation" / "summary_teacher.csv")
    claims = claims_table(dcfg, sim)
    heads = headline(claims)
    ts = term_structure(dcfg)
    write_csv(outdir / "claims.csv", claims)
    write_csv(outdir / "headline.csv", heads)
    write_csv(outdir / "term_structure.csv", ts)

    lines = ["# Addendum D results (generated; rules in docs/PREANALYSIS.md)", "",
             f"teacher_short.yaml hash `{dhash}`. Post-hoc confirmation: the cells were chosen after the "
             "Addendum C results were seen. Gains are QLIKE, positive favours the second forecast; intervals "
             "and p-values are refit+date; Holm across the 12 cells per rule.", "",
             "## Claims per asset and horizon (count of families out of 3)", "", md_table(heads, list(heads.columns)), "",
             "## Cells", "",
             md_table(claims, ["asset", "spec", "family", "status", "L0", "total", "total_p_holm", "teacher", "teacher_lo",
                               "teacher_hi", "teacher_p_holm", "teacher_placebo_p_upper", "input", "input_lo", "input_hi",
                               "input_p_holm", "input_placebo_p_upper", "input_minus_teacher", "imt_p_holm"]), "",
             md_table(claims, ["asset", "spec", "family", "D0_total_gain", "D1_input_claimed", "D1_downgraded_by_simulation",
                               "D2_teacher", "D3_input_exceeds_teacher", "sim_false_C1"]), "",
             "## Robustness of the input gain (date bootstrap, block 63)", "",
             md_table(claims, ["asset", "spec", "family", "input", "gain_input_lawB", "gain_input_lawB_p", "gain_input_recal",
                               "gain_input_recal_p", "gain_input_excovid", "gain_input_excovid_p"]), "",
             "## Term structure (relative to L(f0))", "",
             md_table(ts, ["asset", "spec", "family", "interval", "total_rel", "teacher_rel", "teacher_rel_lo", "teacher_rel_hi",
                           "input_rel", "input_rel_lo", "input_rel_hi"])]
    (outdir / "REPORT.md").write_text("\n".join(lines) + "\n")
    write_json(outdir / "manifest_15_collect_short.json", {"short_config_hash": dhash})
    print(f"wrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
