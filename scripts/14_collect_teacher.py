#!/usr/bin/env python3
"""Addendum C: apply the prespecified decision rules and assemble every teacher / input result.

Bookkeeping only: it reads the outputs of scripts 11, 12 and 13 and applies the rules written in
docs/PREANALYSIS.md (Addendum C) and configs/teacher.yaml.  Writes results/<profile>/teacher/:

  claims.csv        C1 (input), C2 (teacher), C3 (non-superiority) and the dissociation reading,
                    for the confirmatory cells (SPX and NDX x har, enet, xgb; primary spec)
  secondary.csv     every asset / spec / family / profile with a decomposition, all sensitivities
  simulation_oc.csv operating characteristics of the rules in the Monte Carlo audit
  REPORT.md         the same, readable; no interpretation beyond the rules
"""
import _bootstrap  # noqa: F401
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ivbehav.config import CONFIG_DIR, RESULTS_DIR, load_config, write_csv, write_json
from ivbehav.inference import holm
from ivbehav.teacher_report import _read, cell, md_table, p2_status, sim_false_detection

def claims_table(tcfg: dict, sim: pd.DataFrame | None) -> pd.DataFrame:
    conf = tcfg["confirmatory"]
    profile = load_config(conf["config"])["profile"]
    a, block, margin = conf["alpha"], conf["block"], conf["margin_rel"]
    rows = []
    for asset in conf["assets"]:
        for family in conf["families"]:
            c = cell(profile, asset, conf["spec"], family, block)
            if c is None:
                rows.append({"asset": asset, "family": family, "status": "missing"})
                continue
            r = {"asset": asset, "family": family, "L0": c.get("L0"), "n_refit_draws": c.get("gain_input_comb_n", np.nan)}
            for comp in ("input", "teacher", "total"):
                key = f"gain_{comp}"
                comb = f"{key}_comb_p" in c and np.isfinite(c.get(f"{key}_comb_p", np.nan))
                r.update({f"{comp}": c.get(key), f"{comp}_lo": c.get(f"{key}_comb_lo" if comb else f"{key}_lo"),
                          f"{comp}_hi": c.get(f"{key}_comb_hi" if comb else f"{key}_hi"),
                          f"{comp}_p": c.get(f"{key}_comb_p" if comb else f"{key}_p"),
                          f"{comp}_interval": "refit+date" if comb else "date only",
                          f"{comp}_placebo_p_upper": c.get(f"{key}_placebo_p_upper", np.nan),
                          f"{comp}_placebo_p_lower": c.get(f"{key}_placebo_p_lower", np.nan)})
                rel = f"rel_{key}"
                r[f"{comp}_rel"] = c.get(rel)
                r[f"{comp}_rel_q95"] = c.get(f"{rel}_comb_q95", c.get(f"{rel}_q95"))
            r.update(p2_status(RESULTS_DIR / profile / asset / conf["spec"] / "tables", family, a))
            proxies = tcfg["simulation_proxy"][family]
            r["sim_false_C1"] = sim_false_detection(sim, proxies, "detect_C1")
            r["sim_pop_teacher_redundant"] = (float(sim[(sim["mode"] == "redundant") & (sim["law"] == "A_wellspec")
                                                        & sim["learner"].isin(proxies)]["pop_teacher"].mean())
                                              if sim is not None else np.nan)
            rows.append(r)
    t = pd.DataFrame(rows)
    if "input_p" not in t:
        return t
    for comp in ("input", "teacher"):
        t[f"{comp}_p_holm"] = holm(t[f"{comp}_p"].to_numpy(float))
    t["C1_input_claimed"] = ((t["input"] > 0) & (t["input_p_holm"] < a) & (t["input_placebo_p_upper"] <= a)
                             & ~(t["sim_false_C1"] > conf["max_false_detection"]))
    t["C1_downgraded_by_simulation"] = ((t["input"] > 0) & (t["input_p_holm"] < a) & (t["input_placebo_p_upper"] <= a)
                                        & (t["sim_false_C1"] > conf["max_false_detection"]))
    t["C2_teacher"] = np.select(
        [(t["teacher"] > 0) & (t["teacher_p_holm"] < a) & (t["teacher_placebo_p_upper"] <= a),
         (t["teacher"] < 0) & (t["teacher_p_holm"] < a)],
        ["teaches (beyond placebo)", "harmful reorganisation"], default="not claimed")
    t["C3_rules_out_input_gain"] = t["input_rel_q95"] < margin
    t["C3_rules_out_total_gain"] = t["total_rel_q95"] < margin
    t["dissociation"] = (t["p2_claimed_features"] >= 1) & ~t["C1_input_claimed"] & t["C3_rules_out_input_gain"]
    return t


def secondary_table(tcfg: dict) -> pd.DataFrame:
    block = tcfg["confirmatory"]["block"]
    rows = []
    for profile_dir in sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir()):
        profile = profile_dir.name
        if profile not in (load_config(tcfg["confirmatory"]["config"])["profile"],
                           load_config(tcfg["neural"]["config"])["profile"]):
            continue
        for asset_dir in sorted(p for p in profile_dir.iterdir() if p.is_dir() and p.name.isupper()):
            for spec_dir in sorted(p for p in asset_dir.iterdir() if p.is_dir()):
                for f in sorted((spec_dir / "tables").glob("teacher_boot_*.csv")):
                    family = f.stem.replace("teacher_boot_", "")
                    c = cell(profile, asset_dir.name, spec_dir.name, family, block)
                    if c is not None:
                        rows.append(c)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher-config", default="teacher.yaml")
    args = ap.parse_args()
    tpath = Path(args.teacher_config) if Path(args.teacher_config).exists() else CONFIG_DIR / args.teacher_config
    ttext = tpath.read_text()
    tcfg = yaml.safe_load(ttext)
    profile = load_config(tcfg["confirmatory"]["config"])["profile"]
    outdir = RESULTS_DIR / profile / "teacher"
    outdir.mkdir(parents=True, exist_ok=True)

    sim = _read(RESULTS_DIR / profile / "simulation" / "summary_teacher.csv")
    claims = claims_table(tcfg, sim)
    secondary = secondary_table(tcfg)
    write_csv(outdir / "claims.csv", claims)
    write_csv(outdir / "secondary.csv", secondary)
    if sim is not None:
        write_csv(outdir / "simulation_oc.csv", sim[sim["law"] == "A_wellspec"])

    lines = [f"# Addendum C results (generated; rules in docs/PREANALYSIS.md)", "",
             f"teacher.yaml hash `{hashlib.sha256(ttext.encode()).hexdigest()[:16]}`. Gains are QLIKE, positive favours "
             "the second forecast; `rel` columns are shares of L(f0). Intervals are refit+date where available.", "",
             "## Confirmatory cells (primary spec)", ""]
    lines.append(md_table(claims, ["asset", "family", "n_refit_draws", "L0", "total", "teacher", "teacher_lo", "teacher_hi", "teacher_p_holm",
                                   "teacher_placebo_p_upper", "input", "input_lo", "input_hi", "input_p_holm",
                                   "input_placebo_p_upper", "input_rel_q95", "p2_claimed_features"]))
    lines += ["", md_table(claims, ["asset", "family", "C1_input_claimed", "C1_downgraded_by_simulation", "C2_teacher",
                                    "C3_rules_out_input_gain", "C3_rules_out_total_gain", "dissociation", "sim_false_C1",
                                    "sim_pop_teacher_redundant"])]
    if len(secondary):
        lines += ["", "## All decompositions (date bootstrap, block 63)", ""]
        sec = secondary.sort_values(["profile", "asset", "spec", "family"])
        lines.append(md_table(sec, ["profile", "asset", "spec", "family", "L0", "gain_total", "gain_teacher", "gain_teacher_p",
                                    "gain_teacher_placebo_p_upper", "gain_input", "gain_input_p", "gain_input_placebo_p_upper",
                                    "gain_teacher_recal", "gain_input_recal", "gain_teacher_lawB", "gain_input_lawB",
                                    "gain_teacher_excovid", "gain_input_excovid"]))
    if sim is not None:
        lines += ["", "## Simulation (law A)", ""]
        lines.append(md_table(sim[sim["law"] == "A_wellspec"],
                              ["mode", "b_bar", "learner", "pop_teacher", "pop_input", "bias_teacher", "bias_input",
                               "teacher_covers_pop", "input_covers_pop", "detect_input_pos", "detect_C1", "detect_C2",
                               "noninferior_margin", "n_reps"]))
    (outdir / "REPORT.md").write_text("\n".join(lines) + "\n")
    write_json(outdir / "manifest_14_collect_teacher.json", {"teacher_config_hash": hashlib.sha256(ttext.encode()).hexdigest()[:16]})
    print(f"wrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
