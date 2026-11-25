"""Style, data access and drawing helpers for the paper figures (paper_figures.ipynb).

Reads only finished outputs under results/ (no estimation).  Draw-level files (refit and placebo
JSON) are gathered once into notebooks/.cache/*.parquet; a cache is rebuilt when the number of
draw files changes.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import PathPatch
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
PRI = RES / "primary"
FIG = HERE / "figures"
CACHE = HERE / ".cache"

# ------------------------------------------------------------------------------------------
# Vocabulary and palette (one meaning per colour across all figures)
# ------------------------------------------------------------------------------------------
ASSETS = ["SPX", "NDX"]
FAMILIES = ["har", "enet", "xgb"]
SPECS = ["h5td", "h10td", "primary"]                     # short to long horizon
HORIZON = {"h5td": "5 sessions", "h10td": "10 sessions", "primary": "30 days"}
HORIZON_SHORT = {"h5td": "5d", "h10td": "10d", "primary": "30d"}
HORIZON_SESSIONS = {"h5td": 5, "h10td": 10, "primary": 21}
FEATS = ["rv_w", "rv_m", "rv_neg_w"]
FEAT_LABEL = {"rv_w": "Weekly RV", "rv_m": "Monthly RV", "rv_neg_w": "Weekly downside RV"}

FAM_LABEL = {"har": "HAR", "enet": "Elastic net", "xgb": "Boosted trees", "har_ols": "HAR (OLS)",
             "mlp": "MLP", "tcn": "TCN"}
FAM_COLOR = {"har": "#2E5EAA", "enet": "#E07A1F", "xgb": "#1B9E77", "har_ols": "#8FB3E3",
             "mlp": "#7B5EA7", "tcn": "#B0577A"}
COMP_COLOR = {"teacher": "#7B4FA0", "input": "#D1495B", "total": "#2B2B2B"}
PLACEBO = "#9AA0A6"
INK = "#222222"
GRID = "#E6E6E6"
HORIZON_MARKER = {"h5td": "^", "h10td": "s", "primary": "o"}
ASSET_FILL = {"SPX": True, "NDX": False}

# Diverging map for gains: losses brown, gains teal (colour-blind safe, prints in grey).
GAIN_CMAP = LinearSegmentedColormap.from_list(
    "gain", ["#7F3B08", "#C9772B", "#F3D9B1", "#F7F7F7", "#B8E2DC", "#3F9C8F", "#01434A"])
SURPRISE_CMAP = LinearSegmentedColormap.from_list(
    "surprise", ["#2C5F8A", "#8DB7D6", "#F7F7F7", "#E8A07E", "#A63A1F"])

EPISODES = [("2018-02-01", "2018-04-15", "Feb-2018\nvol shock"),
            ("2020-02-20", "2020-05-15", "COVID\ncrash"),
            ("2022-01-03", "2022-10-31", "2022\nbear market"),
            ("2025-04-01", "2025-05-15", "Apr-2025\ntariffs")]


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "axes.edgecolor": "#3A3A3A", "axes.labelcolor": INK,
        "xtick.color": "#3A3A3A", "ytick.color": "#3A3A3A",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlepad": 4,
        "legend.frameon": False, "legend.handlelength": 1.6,
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "lines.solid_capstyle": "round", "lines.dash_capstyle": "round",
        "hatch.linewidth": 0.5, "image.interpolation": "bicubic",
    })


def save(fig, name: str) -> None:
    """Vector PDF (fonts embedded) plus a 600-dpi PNG."""
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=600)


# ------------------------------------------------------------------------------------------
# Tables
# ------------------------------------------------------------------------------------------
def table(asset: str, spec: str, name: str, profile: str = "primary") -> pd.DataFrame:
    return pd.read_csv(RES / profile / asset / spec / "tables" / name)


@lru_cache(None)
def origin(asset: str, spec: str, family: str, profile: str = "primary") -> pd.DataFrame:
    f = pd.read_parquet(RES / profile / asset / spec / "behavior" / family / "origin.parquet")
    f["date"] = pd.to_datetime(f["date"])
    return f


def claims_C(pre: bool = False) -> pd.DataFrame:
    return pd.read_csv(PRI / ("pre_addendumE/teacher" if pre else "teacher") / "claims.csv")


def claims_D(pre: bool = False) -> pd.DataFrame:
    return pd.read_csv(PRI / ("pre_addendumE/teacher_short" if pre else "teacher_short") / "claims.csv")


@lru_cache(None)
def secondary() -> pd.DataFrame:
    return pd.read_csv(PRI / "teacher" / "secondary.csv")


def sec_row(asset: str, spec: str, family: str, profile: str = "primary") -> pd.Series:
    s = secondary()
    r = s[(s.profile == profile) & (s.asset == asset) & (s.spec == spec) & (s.family == family)]
    return r.iloc[0]


def boot_stat(asset, spec, family, statistic, block=63, profile="primary") -> pd.Series:
    t = table(asset, spec, f"teacher_boot_{family}.csv", profile)
    return t[(t.statistic == f"ti:{statistic}") & (t.block == block)].iloc[0]


def refit_summary(asset, spec, family, statistic, pre=False) -> pd.Series:
    if pre:
        t = pd.read_csv(PRI / "pre_addendumE" / "tables" / asset / spec / f"teacher_refit_{family}.csv")
    else:
        t = table(asset, spec, f"teacher_refit_{family}.csv")
    return t[(t.source == "combined") & (t.statistic == f"ti:{statistic}")].iloc[0]


def placebo_summary(asset, spec, family, statistic) -> pd.Series:
    t = table(asset, spec, f"teacher_placebo_{family}.csv")
    return t[t.statistic == f"ti:{statistic}"].iloc[0]


# ------------------------------------------------------------------------------------------
# Draw-level data (cached)
# ------------------------------------------------------------------------------------------
def _gather(files: list[Path], pick, cache: Path) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        df = pd.read_parquet(cache)
        if len(df) == len(files):
            return df
    df = pd.DataFrame([pick(json.load(open(p))) for p in files])
    df.to_parquet(cache)
    return df


@lru_cache(None)
def refit_draws(asset: str, spec: str, family: str) -> pd.DataFrame:
    """Refit+date draws of the teacher/input decomposition (Addenda C, D, E)."""
    files = sorted((PRI / asset / spec / "refit_ti" / family).glob("b*.json"))

    def pick(rec):
        return {k.removeprefix("ti:"): v for k, v in rec["combined"].items()}
    return _gather(files, pick, CACHE / f"refit_{asset}_{spec}_{family}.parquet")


@lru_cache(None)
def placebo_draws(asset: str, spec: str, family: str, kind: str = "cond_block") -> pd.DataFrame:
    """Persistence-matched placebo draws of the behavioural statistics (script 06)."""
    files = sorted((PRI / asset / spec / "placebo" / family / kind).glob("r*.json"))
    return _gather(files, lambda rec: rec, CACHE / f"placebo_{asset}_{spec}_{family}_{kind}.parquet")


# ------------------------------------------------------------------------------------------
# Small numerical helpers
# ------------------------------------------------------------------------------------------
def qlike(y, f):
    r = np.asarray(y) / np.asarray(f)
    return r - np.log(r) - 1.0


def boot_p(values, null=0.0) -> float:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    lo = (np.sum(v <= null) + 1) / (v.size + 1)
    hi = (np.sum(v >= null) + 1) / (v.size + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def kde_1d(values, grid, bw=None):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return gaussian_kde(v, bw_method=bw)(grid)


def hdr_level(kde, pts, mass):
    """Density threshold enclosing `mass` of the sample (highest-density region)."""
    d = kde(pts)
    return float(np.quantile(d, 1 - mass))


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "–"
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}".lstrip("0") if p < 1 else "1"


# ------------------------------------------------------------------------------------------
# Drawing helpers
# ------------------------------------------------------------------------------------------
def gradient_between(ax, x, y1, y2, color, weight=None, alpha=(0.06, 0.6), zorder=1, n=400):
    """Fill between y1 and y2 with an alpha ramp along x.

    The alpha at each x follows `weight` (default |y2 - y1|), so the fill is strongest where the
    two curves diverge most.  Implemented as an RGBA image clipped to the fill polygon.
    """
    x = np.asarray(x, float)
    poly = ax.fill_between(x, y1, y2, facecolor="none", edgecolor="none", zorder=zorder)
    patch = PathPatch(poly.get_paths()[0], facecolor="none", edgecolor="none", transform=ax.transData)
    ax.add_patch(patch)
    xs = np.linspace(x.min(), x.max(), n)
    w = np.abs(np.asarray(y2) - np.asarray(y1)) if weight is None else np.asarray(weight, float)
    w = np.interp(xs, x, w)
    w = (w - w.min()) / (np.ptp(w) + 1e-300)
    img = np.zeros((2, n, 4))
    img[..., :3] = to_rgb(color)
    img[..., 3] = alpha[0] + (alpha[1] - alpha[0]) * w
    lo, hi = min(np.min(y1), np.min(y2)), max(np.max(y1), np.max(y2))
    im = ax.imshow(img, extent=[x.min(), x.max(), lo, hi], aspect="auto", origin="lower",
                   zorder=zorder, interpolation="bilinear")
    im.set_clip_path(patch)
    return im


def cmap_fill(ax, x, ylow, yhigh, cmap, vmin, vmax, alpha=1.0, zorder=1, n=600):
    """Fill between ylow and yhigh, coloured by x through `cmap` (clipped image)."""
    x = np.asarray(x, float)
    poly = ax.fill_between(x, ylow, yhigh, facecolor="none", edgecolor="none", zorder=zorder)
    patch = PathPatch(poly.get_paths()[0], facecolor="none", edgecolor="none", transform=ax.transData)
    ax.add_patch(patch)
    xs = np.linspace(x.min(), x.max(), n)
    rgba = cmap(np.clip((xs - vmin) / (vmax - vmin), 0, 1))
    rgba[:, 3] = alpha
    lo, hi = min(np.min(ylow), np.min(yhigh)), max(np.max(ylow), np.max(yhigh))
    im = ax.imshow(rgba[None, :, :].repeat(2, 0), extent=[x.min(), x.max(), lo, hi], aspect="auto",
                   origin="lower", zorder=zorder, interpolation="bilinear")
    im.set_clip_path(patch)
    return im


def horizon(ax, x, y, band, n_bands=3, pos=("#FDD9B5", "#F4A261", "#C8553D"),
            neg=("#CFE3F2", "#7FB3D8", "#2C6E9E")):
    """Horizon chart: |y| folded into `n_bands` layers of height `band`; positives warm, negatives cool."""
    y = np.asarray(y, float)
    for k in range(n_bands):
        up = np.clip(y - k * band, 0, band)
        dn = np.clip(-y - k * band, 0, band)
        ax.fill_between(x, 0, up, color=pos[k], lw=0, step="mid", zorder=2 + k)
        ax.fill_between(x, 0, dn, color=neg[k], lw=0, step="mid", zorder=2 + k)
    ax.set_ylim(0, band)


def density_strip(ax, y, values, color, xlim, height=0.72, n=500, gamma=0.75, bw=None):
    """A horizontal strip whose opacity follows the KDE of `values` (a 'gradient interval')."""
    xs = np.linspace(*xlim, n)
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    d = kde_1d(v[(v > xlim[0] - 1) & (v < xlim[1] + 1)], xs, bw)
    d = (d / d.max()) ** gamma
    img = np.zeros((1, n, 4))
    img[..., :3] = to_rgb(color)
    img[..., 3] = 0.92 * d
    ax.imshow(img, extent=[xlim[0], xlim[1], y - height / 2, y + height / 2], aspect="auto",
              origin="lower", zorder=2, interpolation="bilinear")


def panel_label(ax, letter: str, x=-0.02, y=1.02) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


def soft_grid(ax, axis="y") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def cell_label(asset: str, family: str) -> str:
    return f"{asset} · {FAM_LABEL[family]}"


def shade_episodes(ax, label: bool = False, y=1.0, color="#EDEDED") -> None:
    for a, b, name in EPISODES:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=color, lw=0, zorder=0)
        if label:
            ax.text(pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2, y, name,
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6,
                    color="#555555", linespacing=0.95)


def fmt_pct(v: float) -> str:
    """Signed whole percent with a true minus sign; tiny values print as 0."""
    x = round(100 * v)
    return "0" if x == 0 else f"{x:+d}".replace("-", "−")


# ------------------------------------------------------------------------------------------
# Wide (single-column) layouts for RQ2–RQ5
# ------------------------------------------------------------------------------------------
W = 7.0                                       # text width of a single-column template, inches
FAM_SHORT = {"har": "HAR", "enet": "EN", "xgb": "BT"}


def cell_grid(specs=SPECS, gap_asset=0.35, gap_spec=0.9):
    """x positions for horizon × asset × family cells: [(x, spec, asset, family), ...]."""
    out, x = [], 0.0
    for s in specs:
        for a in ASSETS:
            for f in FAMILIES:
                out.append((x, s, a, f))
                x += 1
            x += gap_asset
        x += gap_spec - gap_asset
    return out


def multilevel_xlabels(ax, items, horizon=True, y_asset=-0.13, y_hor=-0.24, fs=6.2):
    """Family abbreviations as ticks, then asset and (optionally) horizon labels underneath."""
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.set_xticks([i[0] for i in items], [FAM_SHORT[i[3]] for i in items], fontsize=fs)
    for lab, i in zip(ax.get_xticklabels(), items):
        lab.set_color(FAM_COLOR[i[3]])
    ax.tick_params(axis="x", length=0, pad=2)
    groups = {}
    for x, s, a, f in items:
        groups.setdefault((s, a), []).append(x)
    for (s, a), xs in groups.items():
        ax.text(np.mean(xs), y_asset, a, transform=tr, ha="center", va="top", fontsize=fs, color="#444444")
    if horizon:
        hs = {}
        for x, s, a, f in items:
            hs.setdefault(s, []).append(x)
        for s, xs in hs.items():
            ax.text(np.mean(xs), y_hor, HORIZON[s], transform=tr, ha="center", va="top", fontsize=fs + 0.8,
                    fontweight="bold")
            ax.plot([min(xs) - 0.3, max(xs) + 0.3], [y_hor + 0.035] * 2, transform=tr, color="#9E9E9E", lw=0.6,
                    clip_on=False)


def horizon_separators(ax, items, color="#D5D5D5"):
    xs = {}
    for x, s, a, f in items:
        xs.setdefault(s, []).append(x)
    bounds = [(min(v), max(v)) for v in xs.values()]
    for (lo0, hi0), (lo1, hi1) in zip(bounds[:-1], bounds[1:]):
        ax.axvline((hi0 + lo1) / 2, color=color, lw=0.6, zorder=0)


def density_strip_v(ax, x, values, color, ylim, width=0.62, n=500, gamma=0.75, bw=None):
    """Vertical 'gradient interval': a column whose opacity follows the KDE of `values`."""
    ys = np.linspace(*ylim, n)
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    d = kde_1d(v[(v > ylim[0] - 1) & (v < ylim[1] + 1)], ys, bw)
    d = (d / d.max()) ** gamma
    img = np.zeros((n, 1, 4))
    img[..., :3] = to_rgb(color)
    img[..., 0, 3] = 0.92 * d
    ax.imshow(img, extent=[x - width / 2, x + width / 2, ylim[0], ylim[1]], aspect="auto", origin="lower",
              zorder=2, interpolation="bilinear")


def minus(s: str) -> str:
    return s.replace("-", "−")
