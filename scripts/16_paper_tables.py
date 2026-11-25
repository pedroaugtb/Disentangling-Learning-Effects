#!/usr/bin/env python3
"""Results tables for the paper (Section 5 and appendix), built from finished outputs only.

No model is fitted and no loss is computed here: every number is read from results/ (scripts
03-15 and the Addendum E rerun).  Writes paper/tables/tab_*.tex (needs booktabs, amssymb, array),
paper/tables/main_tables.tex (the three main-text tables: decisions, teacher, inference),
paper/tables/appendix_tables.tex (everything else) and paper/tables/all_tables.tex (both).

    python scripts/16_paper_tables.py

Conventions: gains are QLIKE differences, positive favours the second forecast, and are shown as
a percentage of the cell's full-sample L(f0), so subsample columns share one scale.  Holm
p-values are capped at 1.  Intervals are 95% percentile intervals.
"""
import _bootstrap  # noqa: F401
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ivbehav.inference import holm

ROOT = Path(__file__).resolve().parents[1]
PRI, NN = ROOT / "results" / "primary", ROOT / "results" / "nn2"
OUT = ROOT / "paper" / "tables"
CACHE = ROOT / "notebooks" / ".cache"

ASSETS = ["SPX", "NDX"]
ASSET_TEX = {"SPX": r"S\&P\,500", "NDX": "Nasdaq-100"}
FAMS = ["har", "enet", "xgb"]
FEATS = ["rv_w", "rv_m", "rv_neg_w"]
FEAT_TEX = {"rv_w": "Weekly RV", "rv_m": "Monthly RV", "rv_neg_w": "Weekly downside RV"}
SHORT = ["h5td", "h10td"]
SPECS = ["h5td", "h10td", "primary"]
HOR_S = {"h5td": "5d", "h10td": "10d", "primary": "30d"}
NAME = {"har": "HAR", "enet": "EN", "xgb": "BT", "har_ols": "HAR-OLS", "mlp": "MLP", "tcn": "TCN",
        "oracle": "Oracle", "enet_light": "EN (light)", "enet_heavy": "EN (heavy)"}
YES, NO = r"\checkmark", r"$\times$"


# ------------------------------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------------------------------
def _fmt(v: float, d: int) -> str:
    s = f"{v:.{d}f}"
    return s.lstrip("-") if float(s) == 0 else s


def num(x, d=1, scale=1.0, cap=None) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    v = float(x) * scale
    if cap is not None and abs(v) >= cap:
        return rf"$<\!-{cap:g}$" if v < 0 else rf"$>\!{cap:g}$"
    return f"${_fmt(v, d)}$"


def pct(x, d=1) -> str:
    return num(x, d, 100.0, cap=100)


def ci(lo, hi, d=1, scale=100.0, cap=100) -> str:
    def part(v):
        if v is None or not np.isfinite(v):
            return "?"
        v = float(v) * scale
        if cap is not None and abs(v) >= cap:
            return rf"<\!-{cap:g}" if v < 0 else rf">\!{cap:g}"
        return _fmt(v, d)
    return rf"{{\scriptsize$[{part(lo)},\,{part(hi)}]$}}"


def thou(n) -> str:
    return f"{int(n):,}".replace(",", "{,}")


def fp(p) -> str:
    if p is None or not np.isfinite(p):
        return "--"
    if p >= 0.9995:
        return "1"
    if p < 0.001:
        return r"$<$.001"
    return f"{p:.3f}"[1:]


def rate(x) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{x:.2f}"[1:] if x < 1 else "1"


def bold(cell: str) -> str:
    """Bold a table entry; math-mode numbers need \\mathbf, not \\textbf."""
    if cell.startswith("$") and cell.endswith("$"):
        return rf"$\mathbf{{{cell[1:-1]}}}$"
    return rf"\textbf{{{cell}}}"


def dec(claimed: bool, p) -> str:
    return rf"\textbf{{{fp(p)}}}\,{YES}" if claimed else fp(p)


def table(label, caption, colspec, header, rows, note=None, size=r"\footnotesize", sep="3pt") -> str:
    out = [r"\begin{table}[t]", r"\centering", size, rf"\setlength{{\tabcolsep}}{{{sep}}}",
           rf"\caption{{{caption}}}\label{{{label}}}",
           rf"\begin{{tabular*}}{{\linewidth}}{{@{{\extracolsep{{\fill}}}}{colspec}@{{}}}}",
           r"\toprule", *header, r"\midrule", *rows, r"\bottomrule", r"\end{tabular*}"]
    if note:
        out += [r"\par\smallskip", r"\begin{minipage}{\linewidth}\scriptsize\raggedright",
                note, r"\end{minipage}"]
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def line(*cells) -> str:
    return " & ".join(cells) + r" \\"


def group(text: str, ncol: int) -> str:
    return rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\emph{{{text}}}}} \\"


ASSET_HEAD = (r" & \multicolumn{3}{c}{S\&P\,500} & \multicolumn{3}{c}{Nasdaq-100} \\"
              r" \cmidrule(lr){2-4}\cmidrule(l){5-7}")


# ------------------------------------------------------------------------------------------
# Data access
# ------------------------------------------------------------------------------------------
def tdir(asset, spec="primary", profile="primary") -> Path:
    return (PRI if profile == "primary" else NN) / asset / spec / "tables"


def hierarchy(asset, profile="primary") -> pd.DataFrame:
    return pd.read_csv(tdir(asset, profile=profile) / "hierarchy.csv")


def hrow(h, group_, family, stat):
    r = h[(h.group == group_) & (h.family == family) & (h.statistic == stat)]
    return r.iloc[0] if len(r) else None


def pick(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    r = df[m]
    return r.iloc[0] if len(r) else None


SEC = pd.read_csv(PRI / "teacher" / "secondary.csv")
CC, CC_PRE = (pd.read_csv(PRI / d / "claims.csv") for d in ("teacher", "pre_addendumE/teacher"))
CD, CD_PRE = (pd.read_csv(PRI / d / "claims.csv") for d in ("teacher_short", "pre_addendumE/teacher_short"))
TERM = pd.read_csv(PRI / "teacher_short" / "term_structure.csv")
SIM_OC = pd.read_csv(PRI / "teacher" / "simulation_oc.csv")
SIM_MAIN = pd.read_csv(PRI / "simulation" / "summary_main.csv")
SIM_GATE = pd.read_csv(PRI / "simulation" / "summary_gate.csv")


def sec(asset, spec, family, profile="primary"):
    return pick(SEC, profile=profile, asset=asset, spec=spec, family=family)


def tboot(asset, spec, family, profile="primary") -> pd.Series:
    t = pd.read_csv(tdir(asset, spec, profile) / f"teacher_boot_{family}.csv")
    return t[t.block == 63].set_index("statistic")["estimate"]


def scores(asset, profile="primary") -> pd.DataFrame:
    return pd.read_csv(tdir(asset, profile=profile) / "forecast_scores.csv")


def p1_gate(asset, family, profile="primary") -> pd.Series:
    return pd.read_csv(tdir(asset, profile=profile) / "p1_gate.csv").set_index("family").loc[family]


def repeat_q95(asset, family, profile="primary") -> tuple[float, int]:
    r = pd.read_csv(tdir(asset, profile=profile) / f"refit_boot_{family}.csv")
    q = r[r.source == "repeat"].set_index("statistic")
    return max(q.loc["rms_self_f0", "q95"], q.loc["rms_self_f1", "q95"]), int(q.loc["rms_self_f0", "n"])


def p2_cells(asset, family, profile="primary"):
    """P2 rule per feature: dS interval excludes 0, dS Holm p < .05 (date), T beats placebo (Holm)."""
    h, out = hierarchy(asset, profile), {}
    for feat in FEATS:
        ds, T = hrow(h, "P2", family, f"{feat}:dS"), hrow(h, "P2_shape", family, f"{feat}:T")
        if ds is None or T is None:
            continue
        ok = bool((ds.ci_lo > 0 or ds.ci_hi < 0) and ds.boot_p_holm < 0.05 and T.placebo_p_holm < 0.05)
        out[feat] = (ds, T, ok)
    if asset == "NDX" and profile == "primary":          # Addendum B: same sign as SPX
        ref = p2_cells("SPX", family)
        out = {k: (d, t, ok and np.sign(d.estimate) == np.sign(ref[k][0].estimate)) for k, (d, t, ok) in out.items()}
    return out


def refit_draws(asset, spec, family) -> pd.DataFrame:
    cache = CACHE / f"refit_{asset}_{spec}_{family}.parquet"
    files = sorted((PRI / asset / spec / "refit_ti" / family).glob("b*.json"))
    if cache.exists():
        df = pd.read_parquet(cache)
        if len(df) == len(files):
            return df
    return pd.DataFrame([{k.removeprefix("ti:"): v for k, v in json.load(open(p))["combined"].items()}
                         for p in files])


# ------------------------------------------------------------------------------------------
# Table 1: decisions of every pre-specified rule
# ------------------------------------------------------------------------------------------
def tab_decisions() -> str:
    cells = [(a, f) for a in ASSETS for f in FAMS]
    rows = [group("Behaviour, 30 days (RQ1)", 7)]
    r = []
    for a, f in cells:
        r.append(YES if p1_gate(a, f).beats_placebo else NO)
    rows.append(line(r"P1a $D$ beyond placebo", *r))
    r = []
    for a, f in cells:
        q, _ = repeat_q95(a, f)
        rmsD = hrow(hierarchy(a), "P1_gate", f, "rms_D").estimate
        r.append((YES if rmsD > q else NO) + rf"\,{{\scriptsize {rmsD / q:.2f}}}")
    rows.append(line(r"P1b $D$ beyond refit noise", *r))
    r = []
    for a, f in cells:
        c = p2_cells(a, f)
        r.append(f"{sum(ok for *_, ok in c.values())}/{len(c)}")
    rows.append(line(r"P2 features claimed (of 3)", *r))
    rows += [r"\addlinespace", group("Accuracy and its source, 30 days (RQ2--RQ3)", 7)]
    r = []
    for a, f in cells:
        h = hierarchy(a)
        ss, dq = hrow(h, "P3", f, "pairs:sep_slope_rel"), hrow(h, "P3", f, "pairs:dqlike")
        claimed = (ss.estimate > 0 and ss.combined_p_holm < .05 and ss.estimate > ss.placebo_q95) or \
                  (dq.estimate < 0 and dq.combined_p_holm < .05)
        r.append(dec(claimed, min(ss.combined_p_holm, dq.combined_p_holm)))
    rows.append(line(r"P3 validated differentiation", *r))
    for key, label in (("C1", "input gain"), ("C2", "teacher gain")):
        r = []
        for a, f in cells:
            c = pick(CC, asset=a, family=f)
            if key == "C1":
                r.append(dec(bool(c.C1_input_claimed), c.input_p_holm))
            else:
                r.append(dec(c.C2_teacher != "not claimed", c.teacher_p_holm))
        rows.append(line(rf"{key} {label}", *r))
    for comp in ("input", "total"):
        r = []
        for a, f in cells:
            c = pick(CC, asset=a, family=f)
            q = c[f"{comp}_rel_q95"]
            r.append(rf"\textbf{{{_fmt(100 * q, 1)}}}\,{YES}" if q < 0.05 else _fmt(100 * q, 1))
        rows.append(line(rf"C3 {comp} gain $<5\%$ (q95, \%)", *r))
    rows.append(line(r"Dissociation", *[YES if pick(CC, asset=a, family=f).dissociation else NO
                                         for a, f in cells]))
    rows += [r"\addlinespace", group("Short horizons, post hoc (RQ4)", 7)]
    for key, col, flag, label in (("D0", "total_p_holm", "D0_total_gain", "total gain"),
                                  ("D1", "input_p_holm", "D1_input_claimed", "input gain"),
                                  ("D3", "imt_p_holm", "D3_input_exceeds_teacher", "input $>$ teacher")):
        for s in SHORT:
            r = []
            for a, f in cells:
                c = pick(CD, asset=a, spec=s, family=f)
                r.append(dec(bool(c[flag]), c[col]))
            rows.append(line(rf"{key} {label}, {HOR_S[s].replace('d', ' sessions')}", *r))
    header = [ASSET_HEAD, line("Rule", *[NAME[f] for f in FAMS] * 2)]
    note = (r"Entries are Holm-adjusted $p$-values; bold with \checkmark{} marks a claim. "
            r"P1b reports $\mathrm{rms}(D)$ divided by the q95 of the repeated-fit distance (pass if $>1$). "
            r"P2 counts the three history features that pass (Holm over nine cells; the Nasdaq-100 must "
            r"also match the S\&P\,500 sign). P3 shows the smaller of the two adjusted $p$-values. "
            r"P3, C and D use refit+date draws; C and D use 2{,}000 draws (Addendum~E). "
            r"C3 shows the one-sided 95\% bound of the gain as a percentage of $L(f_0)$. "
            r"D rules are Holm-adjusted over the 12 short-horizon cells; D2 (teaching) is claimed in "
            rf"none of them (smallest Holm $p$ {fp(CD.teacher_p_holm.min())}).")
    return table("tab:decisions", "Decisions of the pre-specified rules. HAR, EN and BT are the "
                 "HAR, elastic-net and boosted-tree pairs.",
                 "l" + "c" * 6, header, rows, note, sep="2.5pt")


# ------------------------------------------------------------------------------------------
# Table 2: behaviour (RQ1)
# ------------------------------------------------------------------------------------------
def tab_behaviour() -> str:
    rows = []
    for i, a in enumerate(ASSETS):
        if i:
            rows.append(r"\midrule")
        for j, (f, prof) in enumerate([(f, "primary") for f in FAMS] + [("mlp", "nn2"), ("tcn", "nn2")]):
            h = hierarchy(a, prof)
            D, Hf, I = (hrow(h, "P1_gate", f, s) for s in ("rms_D_rel", "rms_HFS_rel", "rms_ISD_rel"))
            q, _ = repeat_q95(a, f, prof)
            ratio = hrow(h, "P1_gate", f, "rms_D").estimate / q
            g = p1_gate(a, f, prof)
            c = p2_cells(a, f, prof) if f != "tcn" else {}
            p2 = f"{sum(ok for *_, ok in c.values())}/{len(c)}" if c else "n/a"
            dag = r"$^\dagger$" if f == "tcn" else ""
            if j == 3:
                rows.append(r"\addlinespace[2pt]")
            rows.append(line(ASSET_TEX[a] if j == 0 else "", NAME[f], pct(D.estimate),
                             pct(Hf.estimate) + dag, pct(I.estimate) + dag, pct(D.placebo_q95),
                             fp(D.placebo_p), f"${ratio:.2f}$",
                             (YES if g.beats_placebo else NO) + "/" + (YES if g.beats_repeated_fit else NO), p2))
            lo, hi = (D.ci_lo, D.ci_hi) if prof == "nn2" else (D.combined_ci_lo, D.combined_ci_hi)
            rows.append(line("", "", ci(lo, hi), *[""] * 7))
    header = [r" & & \multicolumn{3}{c}{rms, \% of $f_0$} & \multicolumn{2}{c}{Placebo} & & & \\",
              r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
              line("Index", "Family", "$D$", "HFS", "ISD", "q95", "$p$", r"$D/\mathrm{rf}_{95}$", "P1", "P2")]
    note = (r"$D=f_1-f_0=\mathrm{HFS}+\mathrm{ISD}$, as root mean square of each component relative to "
            r"$f_0$, with the interval of $\mathrm{rms}(D)$ below (refit+date; date only for the neural rows). Placebo: q95 and one-sided "
            r"$p$ of $\mathrm{rms}(D)$ under the persistence-matched fake signal. "
            r"$D/\mathrm{rf}_{95}$: $\mathrm{rms}(D)$ over the q95 of the distance between two refits of "
            r"the same model (larger of $f_0$, $f_1$). P1 shows the placebo and refit-noise criteria. "
            r"Refit draws: 500 (HAR, EN), 200 (BT), 100 (neural). Neural rows use the date bootstrap and "
            r"200 placebo draws. $^\dagger$TCN reads 66 raw returns, so its projection conditions on "
            r"summaries only and its split is coarser; its P2 is undefined.")
    return table("tab:behaviour", "How much adding implied volatility moves the forecast (RQ1).",
                 "llrrrrrrcc", header, rows, note)


# ------------------------------------------------------------------------------------------
# Table 3: 30-day accuracy (RQ2)
# ------------------------------------------------------------------------------------------
def tab_accuracy() -> str:
    rows = []
    for i, a in enumerate(ASSETS):
        if i:
            rows.append(r"\midrule")
        sc = scores(a)
        L = lambda f, m: pick(sc, family=f, model=m).qlike  # noqa: E731
        for j, f in enumerate(FAMS):
            c, s, g = pick(CC, asset=a, family=f), sec(a, "primary", f), sec(a, "garch", f)
            rows.append(line(ASSET_TEX[a] if j == 0 else "", NAME[f], num(L(f, "f0"), 3), num(L(f, "f1"), 3),
                             pct(c.total_rel), pct(c.total_rel_q95), pct(c.input_rel), pct(c.input_rel_q95),
                             pct(g.rel_gain_total), pct(g.rel_gain_input)))
            rows.append(line("", "", "", "", ci(s.rel_gain_total_comb_lo, s.rel_gain_total_comb_hi), "",
                             ci(s.rel_gain_input_comb_lo, s.rel_gain_input_comb_hi), "",
                             ci(g.rel_gain_total_lo, g.rel_gain_total_hi), ci(g.rel_gain_input_lo, g.rel_gain_input_hi)))
        b = {m: L(m, "bench") for m in ("bench_ivonly", "bench_rwrv", "bench_histmean")}
        rows.append(rf"\multicolumn{{10}}{{@{{}}l}}{{\scriptsize Benchmarks: $V$ only {b['bench_ivonly']:.3f}; "
                    rf"scaled monthly RV {b['bench_rwrv']:.3f}; training mean {b['bench_histmean']:.3f}}} \\")
    header = [r" & & \multicolumn{2}{c}{QLIKE} & \multicolumn{4}{c}{Gain from IV, \% of $L(f_0)$}"
              r" & \multicolumn{2}{c}{GARCH signal} \\",
              r"\cmidrule(lr){3-4}\cmidrule(lr){5-8}\cmidrule(l){9-10}",
              line("Index", "Family", "$f_0$", "$f_1$", "Total", "q95", "Input", "q95", "Total", "Input")]
    note = (r"Gains are $L(f_0)-L(f_1)$ (total) and $L(P_1)-L(f_1)$ (input); positive favours $f_1$. "
            r"Intervals and the one-sided bound q95 (rule C3, margin 5\%) use 2{,}000 refit+date draws. "
            r"GARCH: the same pair with a price-only GJR-GARCH variance forecast in place of $V$ "
            r"(date-bootstrap intervals). Benchmarks: QLIKE of a regression on $V$ alone, of the scaled "
            r"monthly realised variance and of the training mean. 2{,}241 out-of-sample origins per index.")
    return table("tab:accuracy", "Thirty-day accuracy with and without implied volatility (RQ2).",
                 "llrrrrrrrr", header, rows, note)


# ------------------------------------------------------------------------------------------
# Table 4: matched pairs (P3) and option specificity (RQ2)
# ------------------------------------------------------------------------------------------
def tab_pairs() -> str:
    rows = []
    for i, a in enumerate(ASSETS):
        if i:
            rows.append(r"\midrule")
        h = hierarchy(a)
        for j, f in enumerate(FAMS):
            ss, dq = hrow(h, "P3", f, "pairs:sep_slope_rel"), hrow(h, "P3", f, "pairs:dqlike")
            cd = hrow(h, "P3_secondary", f, "pairs:conc_diff")
            gp = pd.read_csv(tdir(a, "garch") / f"pairs_boot_{f}.csv")
            gp = gp[(gp.block == 63) & (gp.statistic == "pairs:conc_diff")].iloc[0]
            rows.append(line(ASSET_TEX[a] if j == 0 else "", NAME[f], num(ss.estimate, 2), fp(ss.combined_p_holm),
                             num(dq.estimate, 3), fp(dq.combined_p_holm), num(cd.estimate, 1, 100),
                             fp(cd.placebo_p), num(gp.estimate, 1, 100)))
            rows.append(line("", "", ci(ss.combined_ci_lo, ss.combined_ci_hi, 2, 1, None), "",
                             ci(dq.combined_ci_lo, dq.combined_ci_hi, 3, 1, None), "",
                             ci(cd.ci_lo, cd.ci_hi), "", ci(gp.ci_lo, gp.ci_hi)))
        n = pd.read_csv(tdir(a) / "pairs_design_har.csv").n.sum()
        rows.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\scriptsize {thou(n)} matched pairs}} \\")
    header = [r" & & \multicolumn{4}{c}{P3 (confirmatory)} & \multicolumn{3}{c}{Concordance gain, pp} \\",
              r"\cmidrule(lr){3-6}\cmidrule(l){7-9}",
              line("Index", "Family", "Slope", r"$p_{\mathrm{H}}$", r"$\Delta$QLIKE", r"$p_{\mathrm{H}}$",
                   "IV", r"$p_{\mathrm{pl}}$", "GARCH")]
    note = (r"Pairs of dates with matched histories and different IV surprises. Slope: regression of the "
            r"within-pair relative outcome difference on the within-pair change that $V$ induces "
            r"(1 = fully validated); $\Delta$QLIKE: mean loss change $f_1-f_0$ over pairs (negative favours "
            r"$f_1$). Both with refit+date intervals and Holm over six tests. Concordance gain: share of "
            r"pairs in which the forecast difference has the sign of the outcome difference, $f_1$ minus "
            r"$f_0$, in percentage points, with its placebo $p$; GARCH replaces $V$ by the price-only "
            r"signal (date-bootstrap intervals).")
    return table("tab:pairs", "Do the forecast differences induced by IV track outcomes? (RQ2)",
                 "llrrrrrrr", header, rows, note)


# ------------------------------------------------------------------------------------------
# Table 5: teacher or input (RQ3)
# ------------------------------------------------------------------------------------------
def tab_teacher(comp: str = "teacher") -> str:
    """Main text: the teacher gain; appendix (comp='input'): the input gain, same checks."""
    rows = []
    for a in ASSETS:
        if a == "NDX":
            rows.append(r"\midrule")
        for j, f in enumerate(["har", "har_ols", "enet", "xgb"]):
            t, s = tboot(a, "primary", f), sec(a, "primary", f)
            L0 = t["ti:L0"]
            rel = lambda k: t[f"ti:gain_{comp}{k}"] / L0  # noqa: E731
            if f == "har_ols":
                lo, hi, mark = s[f"rel_gain_{comp}_lo"], s[f"rel_gain_{comp}_hi"], r"$^\ddagger$"
            else:
                lo, hi, mark = s[f"rel_gain_{comp}_comb_lo"], s[f"rel_gain_{comp}_comb_hi"], ""
            rows.append(line(ASSET_TEX[a] if j == 0 else "", NAME[f] + mark, pct(rel("")),
                             fp(s[f"gain_{comp}_placebo_p_upper"]), pct(rel("_lawA_on_supB")),
                             pct(rel("_lawB")), pct(rel("_raw_on_recal")), pct(rel("_recal"))))
            rows.append(line("", "", ci(lo, hi), *[""] * 5))
    header = [r" & & \multicolumn{2}{c}{All origins, law A} & \multicolumn{2}{c}{Law-B support}"
              r" & \multicolumn{2}{c}{Recalibration sample} \\",
              r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
              line("Index", "Family", "Gain", r"$p_{\mathrm{pl}}$", "Law A", "Law B", "Raw", "Recal.")]
    sup = {a: tboot(a, "primary", "har")["ti:share_sup_b"] for a in ASSETS}
    rec = tboot("SPX", "primary", "har")["ti:share_recal"]
    what = (r"Teacher gain $L(f_0)-L(P_1)$" if comp == "teacher" else r"Input gain $L(P_1)-L(f_1)$")
    note = (what + r" in \% of the cell's full-sample $L(f_0)$. Law A: log location-scale law for "
            r"$V\mid x$ (refit+date interval, 2{,}000 draws); $p_{\mathrm{pl}}$: one-sided placebo $p$ "
            r"(500 draws). Law B: kernel law, on the origins where it is supported "
            rf"({100 * sup['SPX']:.0f}\% S\&P\,500, {100 * sup['NDX']:.0f}\% Nasdaq-100), next to law A on the same "
            r"origins. Recal.: every forecast scaled in real time by the mean of $Y/f$ over earlier closed "
            rf"labels ({100 * rec:.0f}\% of origins), next to the raw forecasts on the same origins. "
            r"$^\ddagger$OLS anchor: its teacher gain is zero under a linear law (Proposition in "
            r"Section~\ref{sec:method:gain}), so its value under law A measures the bias of the law; "
            r"date-bootstrap interval.")
    if comp == "teacher":
        return table("tab:teacher", "Does implied volatility teach the learner about the history? "
                     "Teacher gain at 30 days (RQ3).", "llrrrrrr", header, rows, note)
    return table("tab:input_split", "Input gain at 30 days under the checks of Table~\\ref{tab:teacher} (RQ3).",
                 "llrrrrrr", header, rows, note)


# ------------------------------------------------------------------------------------------
# Table 6: term structure (RQ4)
# ------------------------------------------------------------------------------------------
def tab_horizon() -> str:
    rows = []
    for i, a in enumerate(ASSETS):
        if i:
            rows.append(r"\midrule")
        for j, f in enumerate(FAMS):
            top, bot = [ASSET_TEX[a] if j == 0 else "", NAME[f]], ["", ""]
            for s in SPECS:
                t = pick(TERM, asset=a, spec=s, family=f)
                tot = pct(t.total_rel)
                if s in SHORT and bool(pick(CD, asset=a, spec=s, family=f).D0_total_gain):
                    tot = bold(tot) + r"$^\ast$"
                top += [tot, pct(t.teacher_rel), pct(t.input_rel)]
                bot += ["", "", ci(t.input_rel_lo, t.input_rel_hi)]
            rows += [line(*top), line(*bot)]
    header = [r" & & \multicolumn{3}{c}{5 sessions} & \multicolumn{3}{c}{10 sessions}"
              r" & \multicolumn{3}{c}{30 days} \\",
              r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-11}",
              line("Index", "Family", *["Total", "Teach.", "Input"] * 3)]
    n = {s: int(pick(scores("SPX") if s == "primary" else pd.read_csv(tdir("SPX", s) / "forecast_scores.csv"),
                     family="har", model="f0").n) for s in SPECS}
    note = (r"Gains in \% of $L(f_0)$ at each horizon, with the 2{,}000-draw refit+date interval of the "
            r"input gain below. $V$ is the 30-day implied variance at every horizon. $^\ast$Rule D0 "
            r"(total gain, Holm over 12 short-horizon cells) claimed. Horizons are compared descriptively: "
            rf"their origin sets differ ({thou(n['h5td'])}, {thou(n['h10td'])} and {thou(n['primary'])} dates), so "
            r"their draws are not paired.")
    return table("tab:horizon", "The term structure of the value of implied volatility (RQ4).",
                 "ll" + "r" * 9, header, rows, note, sep="2.5pt")


# ------------------------------------------------------------------------------------------
# Table 7: how the inference scheme changes the claims (RQ5)
# ------------------------------------------------------------------------------------------
def _cell(a, s, f) -> str:
    return f"{a} {HOR_S[s]} {NAME[f]}"


def _scheme_entry(claimed: list[tuple[str, float]], pmin: tuple[str, float] | None) -> str:
    if claimed:
        return "; ".join(rf"\textbf{{{c}}} ({fp(p)})" for c, p in claimed)
    return "none" + (f" ({fp(pmin[1])})" if pmin else "")


def tab_inference() -> str:
    cells_c = [(a, "primary", f) for a in ASSETS for f in FAMS]
    cells_d = [(a, s, f) for a in ASSETS for s in SHORT for f in FAMS]

    def date_rule(cells, comp, positive=True, placebo=True):
        r = [sec(a, s, f) for a, s, f in cells]
        ph = holm([x[f"gain_{comp}_p"] for x in r])
        sim = {f: pick(CC, asset="SPX", family=f).sim_false_C1 for f in FAMS}
        claimed, best = [], None
        for (a, s, f), x, p in zip(cells, r, ph):
            sign_ok = x[f"gain_{comp}"] > 0 if positive else x[f"gain_{comp}"] < 0
            if not sign_ok:
                continue
            if best is None or p < best[1]:
                best = (_cell(a, s, f), p)
            ok = p < 0.05 and (not placebo or x[f"gain_{comp}_placebo_p_upper"] <= 0.05)
            if comp == "input":
                ok = ok and sim[f] <= 0.10
            if ok:
                claimed.append((_cell(a, s, f), p))
        return claimed, best

    def refit_rule(df, cells, pcol, flag, comp, positive=True, value=None):
        claimed, best = [], None
        for a, s, f in cells:
            c = pick(df, asset=a, family=f) if s == "primary" else pick(df, asset=a, spec=s, family=f)
            g = c[comp]
            if (g > 0) != positive:
                continue
            if best is None or c[pcol] < best[1]:
                best = (_cell(a, s, f), c[pcol])
            ok = (c[flag] == value) if value else bool(c[flag])
            if ok:
                claimed.append((_cell(a, s, f), c[pcol]))
        return claimed, best

    rows = []
    specs = [("C1 input gain, 30d", cells_c, "input", True, True, CC_PRE, CC, "input_p_holm", "C1_input_claimed", None),
             ("C2 teaches, 30d", cells_c, "teacher", True, True, CC_PRE, CC, "teacher_p_holm", "C2_teacher",
              "teaches (beyond placebo)"),
             ("C2 harmful, 30d", cells_c, "teacher", False, False, CC_PRE, CC, "teacher_p_holm", "C2_teacher",
              "harmful reorganisation"),
             ("D0 total gain, 5--10d", cells_d, "total", True, False, CD_PRE, CD, "total_p_holm", "D0_total_gain", None),
             ("D1 input gain, 5--10d", cells_d, "input", True, True, CD_PRE, CD, "input_p_holm", "D1_input_claimed", None),
             ("D2 teaches, 5--10d", cells_d, "teacher", True, True, CD_PRE, CD, "teacher_p_holm", "D2_teacher",
              "teaches (beyond placebo)")]
    for label, cells, comp, positive, placebo, pre, post, pcol, flag, value in specs:
        entries = [_scheme_entry(*date_rule(cells, comp, positive, placebo))]
        for df in (pre, post):
            entries.append(_scheme_entry(*refit_rule(df, cells, pcol, flag, comp, positive, value)))
        rows.append(line(label, *entries))
    entries = ["n/c"]
    for df in (CD_PRE, CD):
        df = df.assign(imt=df.input - df.teacher)
        entries.append(_scheme_entry(*refit_rule(df, cells_d, "imt_p_holm", "D3_input_exceeds_teacher", "imt")))
    rows.append(line("D3 input $>$ teacher, 5--10d", *entries))
    # C3: count of cells whose one-sided bound rules out a 5% gain
    for comp in ("input", "total"):
        e = []
        e.append(", ".join(_cell(a, s, f) for a, s, f in cells_c if sec(a, s, f)[f"rel_gain_{comp}_q95"] < 0.05) or "none")
        for df in (CC_PRE, CC):
            e.append(", ".join(_cell(a, "primary", f) for a in ASSETS for f in FAMS
                               if pick(df, asset=a, family=f)[f"{comp}_rel_q95"] < 0.05) or "none")
        rows.append(line(f"C3 {comp} gain $<5\\%$, 30d", *e))

    head_a = [line("Rule", "Date only", r"Refit+date, $B=200/500$", r"Refit+date, $B=2{,}000$")]
    out = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
           r"\caption{Cells claimed under three ways of measuring uncertainty (RQ5).}\label{tab:inference}",
           r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}>{\raggedright\arraybackslash}p{0.27\linewidth}"
           r">{\raggedright\arraybackslash}p{0.23\linewidth}>{\raggedright\arraybackslash}p{0.23\linewidth}"
           r">{\raggedright\arraybackslash}p{0.23\linewidth}@{}}",
           r"\toprule", *head_a, r"\midrule", *rows, r"\bottomrule", r"\end{tabular*}",
           r"\par\smallskip", r"\begin{minipage}{\linewidth}\scriptsize\raggedright",
           r"The same rules and Holm families under three uncertainty schemes. A claimed cell is "
           r"bold with its Holm $p$; otherwise the smallest Holm $p$ among cells with the right sign is in "
           r"parentheses. Date only: block bootstrap of origins (2{,}000 draws), with the placebo and "
           r"simulation conditions of each rule. The original refit budgets were 500 draws (HAR, EN) and "
           r"200 (BT); Addendum~E raised them to 2{,}000, with the original draws as a subset. "
           r"n/c: not computed (D3 needs paired draws).",
           r"\end{minipage}", r"\end{table}"]
    return "\n".join(out) + "\n"


def tab_inflation() -> str:
    rows_b = []
    for f in FAMS:
        r, cat = [NAME[f]], []
        for s in SPECS:
            for a in ASSETS:
                x = sec(a, s, f)
                r.append(num((x.rel_gain_input_comb_hi - x.rel_gain_input_comb_lo) /
                             (x.rel_gain_input_hi - x.rel_gain_input_lo), 1))
                d = refit_draws(a, s, f)["rel_gain_input"].to_numpy(float)
                cat.append(np.mean(d < -0.5))
        lo_, hi_ = 100 * min(cat), 100 * max(cat)
        r.append(f"${_fmt(lo_, 1)}$" if round(lo_, 1) == round(hi_, 1) else f"${_fmt(lo_, 1)}$--${_fmt(hi_, 1)}$")
        rows_b.append(line(*r))

    header = [r" & \multicolumn{2}{c}{5 sessions} & \multicolumn{2}{c}{10 sessions} & \multicolumn{2}{c}{30 days}"
              r" & Draws $<-50\%$ \\",
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
              line("Family", *["SPX", "NDX"] * 3, r"range, \%")]
    note = (r"Width of the 95\% refit+date interval of the input gain (2{,}000 draws) over the width of the "
            r"date-only interval, and the range over the six cells of each family of the share of refit+date "
            r"draws in which the input gain is below $-50\%$ of $L(f_0)$ (catastrophic refits).")
    return table("tab:inflation", "How much refit uncertainty widens the intervals of the input gain (RQ5).",
                 "lrrrrrrr", header, rows_b, note)


# ------------------------------------------------------------------------------------------
# Table 8: Monte Carlo audit (RQ5)
# ------------------------------------------------------------------------------------------
def sim_reps() -> pd.DataFrame:
    """Per-replicate teacher/input audit (law A), for the naive tests and gains in % of L(f0)."""
    d = pd.concat(pd.read_parquet(f) for f in sorted((PRI / "simulation" / "teacher").glob("reps_*.parquet")))
    return d[d.law == "A_wellspec"]


def sim_ideal_pct(reps: pd.DataFrame) -> dict:
    """Ideal input gain (population gain of observing V) in % of the oracle's loss, per design."""
    o = reps[reps.learner == "oracle"]
    return {(m, b): 100 * (x.ideal_input / x.L0).mean() for (m, b), x in o.groupby(["mode", "b_bar"])}


def tab_simulation() -> str:
    reps, oc = sim_reps(), SIM_OC[SIM_OC.law == "A_wellspec"]
    mn = SIM_MAIN[SIM_MAIN.law == "A_wellspec"]
    tru = pd.read_csv(PRI / "simulation" / "summary_teacher.csv")
    tru = tru[tru.law == "true"]
    learners = ["oracle", "har", "enet_light", "enet_heavy", "xgb"]
    designs = [("constant", 0.10), ("constant", 0.25), ("state", 0.10), ("state", 0.25)]
    na = "--"

    def gate(mode, b, lr):
        g = pick(SIM_GATE, mode=mode, b_bar=b, learner=lr, statistic="rms_D")
        return rate(g.detect) if g is not None else na

    def plac(x, col):  # rules that need the placebo are not run for the oracle
        return na if not np.isfinite(x[col]) else rate(x[col])

    def rng(vals):
        lo, hi = min(vals), max(vals)
        return rate(lo) if rate(lo) == rate(hi) else f"{rate(lo)}--{rate(hi)}"

    rows_a, rows_b, rows_c = [], [], []
    for lr in learners:
        r0 = reps[(reps["mode"] == "redundant") & (reps.learner == lr)]
        red, red_m = pick(oc, mode="redundant", learner=lr), pick(mn, mode="redundant", learner=lr)
        rows_a.append(line(NAME[lr], num(100 * (r0.pop_teacher / r0.L0).mean(), 1),
                           rate((r0.total_p < 0.05).mean()), rate((r0.teacher_p < 0.05).mean()),
                           plac(red, "detect_C1"), plac(red, "detect_C2"), gate("redundant", 0.0, lr),
                           rate(red_m["detect_pairs_slope_pos"]), rate(red_m["detect_pairs_dqlike_neg"])))
        x, t = oc[oc.learner == lr], tru[tru.learner == lr]
        rows_b.append(line(NAME[lr], rng(x.teacher_covers_pop), rng(t.teacher_covers_pop),
                           rng(x.input_covers_pop), rng(x.total_covers_pop)))
        st, st_m = pick(oc, mode="state", b_bar=0.25, learner=lr), pick(mn, mode="state", b_bar=0.25, learner=lr)
        rows_c.append(line(NAME[lr], *[plac(pick(oc, mode=m, b_bar=b, learner=lr), "detect_C1") for m, b in designs],
                           plac(st, "detect_C2"), gate("state", 0.25, lr),
                           rate(st_m["detect_pairs_slope_pos"]), rate(st_m["detect_pairs_dqlike_neg"]),
                           rate(red.noninferior_margin)))
    ideal = sim_ideal_pct(reps)
    out = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
           r"\caption{Operating characteristics of the rules in the Monte Carlo audit (RQ5).}\label{tab:simulation}",
           r"\textit{Panel A: false detections when $V$ is redundant ($b=0$)}\par\smallskip",
           r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrrr@{}}", r"\toprule",
           r" & & \multicolumn{2}{c}{Naive tests} & \multicolumn{2}{c}{Gain rules} & "
           r"\multicolumn{3}{c}{Behaviour rules} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-9}",
           line("Learner", r"$G^{\mathrm{pop}}_{\mathrm{T}}$", "Total", "Teacher", "C1", "C2", "P1a", "P3 slope",
                r"P3 $\Delta$Q"),
           r"\midrule", *rows_a, r"\bottomrule", r"\end{tabular*}",
           r"\par\medskip\textit{Panel B: coverage of nominal 95\% date-bootstrap intervals, range over the "
           r"five designs}\par\smallskip",
           r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}", r"\toprule",
           line("Learner", "Teacher (law A)", "Teacher (true law)", "Input (law A)", "Total"),
           r"\midrule", *rows_b, r"\bottomrule", r"\end{tabular*}",
           r"\par\medskip\textit{Panel C: power}\par\smallskip",
           r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrrrr@{}}", r"\toprule",
           r" & \multicolumn{4}{c}{C1} & \multicolumn{4}{c}{State, $b=0.25$} & C3 \\",
           r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(l){10-10}",
           line("", "c.10", "c.25", "s.10", "s.25", r"C2$^\dagger$", "P1a", "P3 slope", r"P3 $\Delta$Q", "$b=0$"),
           line(r"\emph{Ideal gain, \%}", *[rf"\emph{{{ideal[d]:.0f}}}" for d in designs], "", "", "", "", ""),
           r"\midrule", *rows_c, r"\bottomrule", r"\end{tabular*}",
           r"\par\smallskip", r"\begin{minipage}{\linewidth}\scriptsize\raggedright",
           r"500 replicates per design of a process calibrated on pre-2017 S\&P\,500 data (1{,}760 training and "
           r"2{,}250 test sessions, 30-day overlapping target). Each learner is fitted once per replicate, so "
           r"intervals are date-bootstrap only and target the population value of the fitted pair; the refit "
           r"bootstrap is not simulated. Rates are shares of replicates in which a test fires at the 5\% level, "
           r"unadjusted. $G^{\mathrm{pop}}_{\mathrm{T}}$: population teacher gain of the fitted pair, \% of "
           r"$L(f_0)$; the ideal teacher gain is zero, so this is a functional-form effect. Naive tests: two-sided "
           r"interval of the total or teacher gain excludes zero, with no placebo. C1, C2 (``teaches''), P1a: "
           r"law A with the placebo, 100 replicates with 50 placebo draws; not run for the oracle. "
           r"Power designs: constant (c) or state-dependent (s) information of strength $b$; ideal gain: "
           r"population gain of observing $V$, \% of the oracle's loss. $^\dagger$C2 fires on the fitted pair's "
           r"teacher gain; the ideal teacher gain is still zero. C3: share of redundant replicates in which the "
           r"bound rules out a 5\% input gain. EN (heavy) is not run through the P1a gate.",
           r"\end{minipage}", r"\end{table}"]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------
# Appendix A1: P2 cells
# ------------------------------------------------------------------------------------------
def tab_p2() -> str:
    rows = []
    blocks = [(f, "primary") for f in FAMS] + [("mlp", "nn2")]
    for feat in FEATS:
        for j, (f, prof) in enumerate(blocks):
            top, bot = [FEAT_TEX[feat] if j == 0 else "", NAME[f]], ["", ""]
            for a in ASSETS:
                ds, T, ok = p2_cells(a, f, prof)[feat]
                top += [num(ds.estimate, 2, 1e4) + (rf"\,{YES}" if ok else ""), f"${T.estimate / T.placebo_q95:.1f}$",
                        fp(ds.boot_p_holm), fp(T.placebo_p_holm)]
                bot += [ci(ds.ci_lo, ds.ci_hi, 2, 1e4, None), "", "", ""]
            rows += [line(*top), line(*bot)]
        if feat != FEATS[-1]:
            rows.append(r"\addlinespace[2pt]")
    header = [r" & & \multicolumn{4}{c}{S\&P\,500} & \multicolumn{4}{c}{Nasdaq-100} \\",
              r"\cmidrule(lr){3-6}\cmidrule(l){7-10}",
              line("Feature", "Family", *[r"$\Delta S$", r"$T/q_{95}$", r"$p_{\Delta S}$", r"$p_{T}$"] * 2)]
    note = (r"$\Delta S$: change in the average slope of the response curve when $V$ is added "
            r"($\times10^{-4}$; negative means $f_1$ is flatter), with its date-bootstrap interval. "
            r"$T/q_{95}$: shape change over the q95 of the placebo. $p_{\Delta S}$ (date) and $p_T$ (placebo) "
            r"are Holm-adjusted over the nine classical cells, or over the three MLP cells. \checkmark: P2 claimed "
            r"(and, on the Nasdaq-100, the S\&P\,500 sign replicated).")
    return table("tab:p2", "Response changes of the P2 cells (RQ1).",
                 "ll" + "r" * 8, header, rows, note, sep="2.5pt")


# ------------------------------------------------------------------------------------------
# Appendix A2: alternative specifications and neural learners (date bootstrap)
# ------------------------------------------------------------------------------------------
def tab_specs() -> str:
    specs = [("primary", "Primary"), ("vlag1", "Signal lagged"), ("vol_level", "IV, not variance"),
             ("weekday_cal", "Weekday-scaled"), ("rolling5y", "Rolling 5 years"), ("garch", "GARCH signal")]
    rows = []

    def cellv(x, comp):
        v = pct(x[f"rel_gain_{comp}"])
        sig = x[f"rel_gain_{comp}_lo"] > 0 or x[f"rel_gain_{comp}_hi"] < 0
        return bold(v) if sig else v

    for i, (s, label) in enumerate(specs):
        if i:
            rows.append(r"\addlinespace[2pt]")
        for j, f in enumerate(FAMS):
            r = [label if j == 0 else "", NAME[f]]
            for a in ASSETS:
                x = sec(a, s, f)
                r += [cellv(x, c) for c in ("total", "teacher", "input")]
            rows.append(line(*r))
    header = [r" & & \multicolumn{3}{c}{S\&P\,500} & \multicolumn{3}{c}{Nasdaq-100} \\"
              r" \cmidrule(lr){3-5}\cmidrule(l){6-8}",
              line("Specification", "Family", *["Total", "Teach.", "Input"] * 2)]
    note = (r"30-day gains in \% of $L(f_0)$ with the date bootstrap only; bold: the 95\% interval excludes "
            r"zero. Each specification changes one element of the primary design "
            r"(Section~\ref{sec:protocol:specs}). EN and BT are invariant to the IV-level change "
            r"(a monotone transform of $V$), so their rows repeat the primary. Neural learners: Table~\ref{tab:neural}.")
    return table("tab:specs", "Gains under alternative specifications.",
                 "ll" + "r" * 6, header, rows, note)


# ------------------------------------------------------------------------------------------
# Appendix A2b: neural learners (date bootstrap)
# ------------------------------------------------------------------------------------------
def tab_neural() -> str:
    rows = []
    for i, a in enumerate(ASSETS):
        if i:
            rows.append(r"\midrule")
        for j, f in enumerate(["mlp", "tcn"]):
            x = sec(a, "primary", f, "nn2")
            top, bot = [ASSET_TEX[a] if j == 0 else "", NAME[f]], ["", ""]
            for comp in ("total", "teacher", "input"):
                top.append(pct(x[f"rel_gain_{comp}"]))
                bot.append(ci(x[f"rel_gain_{comp}_lo"], x[f"rel_gain_{comp}_hi"]))
                if comp != "total":
                    top.append(fp(x[f"rel_gain_{comp}_placebo_p_upper"]))
                    bot.append("")
            for comp in ("teacher", "input"):
                k = f"gain_{comp}_excovid"
                top.append(pct(x[k] / x.L0))
                bot.append(ci(x[f"{k}_lo"] / x.L0, x[f"{k}_hi"] / x.L0))
            rows += [line(*top), line(*bot)]
    header = [r" & & & \multicolumn{2}{c}{Teacher} & \multicolumn{2}{c}{Input} & "
              r"\multicolumn{2}{c}{Ex-COVID} \\",
              r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
              line("Index", "Learner", "Total", "Gain", r"$p_{\mathrm{pl}}$", "Gain", r"$p_{\mathrm{pl}}$",
                   "Teacher", "Input")]
    note = (r"30-day gains in \% of the full-sample $L(f_0)$ under law A, with 95\% date-bootstrap intervals "
            r"(the refit bootstrap is not run for the neural learners, and the Monte Carlo audit does not "
            r"cover them). $p_{\mathrm{pl}}$: one-sided placebo $p$ (200 draws). Ex-COVID: origins from 15 "
            r"February to 30 April 2020 removed (52 of 2{,}241). MLP: 8-member ensemble on the seven "
            r"summaries; TCN: 8-member ensemble on the last 66 daily returns. Their behaviour is in "
            r"Table~\ref{tab:behaviour}.")
    return table("tab:neural", "Neural learners: 30-day gains from adding implied volatility.",
                 "ll" + "r" * 7, header, rows, note)


# ------------------------------------------------------------------------------------------
# Appendix A3: robustness of the short-horizon input gain
# ------------------------------------------------------------------------------------------
def tab_short_robust() -> str:
    rows = []
    for i, a in enumerate(ASSETS):
        if i:
            rows.append(r"\midrule")
        first = True
        for s in SHORT:
            for f in FAMS:
                c = pick(CD, asset=a, spec=s, family=f)
                cell = lambda k: f"{pct(c[k] / c.L0)} {{\\scriptsize({fp(c[k + '_p'])})}}"  # noqa: E731
                rows.append(line(ASSET_TEX[a] if first else "", HOR_S[s] if f == "har" else "", NAME[f],
                                 pct(c.input_rel), cell("gain_input_lawB"), cell("gain_input_recal"),
                                 cell("gain_input_excovid"), cell("gain_input_supx")))
                first = False
    header = [line("Index", "Hor.", "Family", "Law A", "Law B", "Recal.", "Ex-COVID", "Supported $x$")]
    note = (r"Input gain in \% of $L(f_0)$ with date-bootstrap two-sided $p$ in parentheses. Law B: on "
            r"its supported origins. Recal.: real-time level recalibration. Ex-COVID: excluding 15 February "
            r"to 30 April 2020. Supported $x$: origins whose history lies inside the training range.")
    return table("tab:short_robust", "Robustness of the short-horizon input gain (RQ4).",
                 "lllrrrrr", header, rows, note)


# ------------------------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_text = {"tab_decisions": tab_decisions, "tab_teacher": tab_teacher, "tab_inference": tab_inference}
    appendix = {"tab_behaviour": tab_behaviour, "tab_p2": tab_p2, "tab_accuracy": tab_accuracy,
                "tab_pairs": tab_pairs, "tab_input_split": lambda: tab_teacher("input"),
                "tab_horizon": tab_horizon, "tab_short_robust": tab_short_robust, "tab_specs": tab_specs,
                "tab_neural": tab_neural,
                "tab_inflation": tab_inflation, "tab_simulation": tab_simulation}
    for name, fn in {**main_text, **appendix}.items():
        (OUT / f"{name}.tex").write_text(f"% generated by scripts/16_paper_tables.py; do not edit\n{fn()}")
        print("wrote", OUT / f"{name}.tex")
    for name, group_ in (("main_tables", main_text), ("appendix_tables", appendix)):
        (OUT / f"{name}.tex").write_text("".join(rf"\input{{tables/{n}}}" + "\n" for n in group_))
    (OUT / "all_tables.tex").write_text(r"\input{tables/main_tables}" + "\n" + r"\clearpage" + "\n"
                                        + r"\input{tables/appendix_tables}" + "\n")


if __name__ == "__main__":
    main()
