"""Daily data: raw snapshots, cross-source QC, and the leak-free per-asset panel.

Timing convention (proposal, Sec. 5.2): the forecast origin t sits after both the index
close and the IV close of date t are observable, so r_t and I_t are predetermined at t and
the target starts with the return of the next session.  Nothing dated after
``sample.end`` enters a label or a feature.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize

from .config import PROCESSED_DIR, RAW_DIR, write_json, write_parquet

CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

HIST_COLS = ["rv_d", "rv_w", "rv_m", "rv_neg_w", "ret_1", "ret_5", "dd_22"]
VARIANCE_COLS = ["rv_d", "rv_w", "rv_m", "rv_neg_w"]
V_COLS = ["V", "V_lag1", "V_vol", "V_wkcal", "V_garch"]
TARGETS = {"Y30": ("calendar", 30), "Y5td": ("trading", 5), "Y10td": ("trading", 10)}

# ETFs enter the QC only (return-gap screen for index ticks); never a predictor or a target.
QC_ETFS = {"SPX": "SPY", "NDX": "QQQ", "RUT": "IWM"}

# Pre-existing project snapshots, used only as extra cross-checks in the QC report.
LEGACY_FILES = {
    "VIX": (RAW_DIR.parents[3] / "cripto/dvol/data/base_data/spy_vix_daily.csv", "Timestamp", "IV"),
    "VXN": (RAW_DIR.parents[3] / "cripto/dvol/data/base_data/qqq_vxn_daily.csv", "Timestamp", "IV"),
    "RVX": (RAW_DIR.parents[3] / "cripto/dvol/data/base_data/iwm_rvx_daily.csv", "Timestamp", "IV"),
    "^GSPC": (RAW_DIR.parents[3] / "FRL'26/data/raw/spx_ohlc.csv", "Date", "Close"),
}


# ------------------------------------------------------------------------------------
# Raw snapshots
# ------------------------------------------------------------------------------------
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _yahoo_path(ticker: str) -> Path:
    return RAW_DIR / f"yahoo_{ticker.replace('^', '')}.csv"


def _fetch(url: str, path: Path) -> None:
    resp = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    path.write_bytes(resp.content)


def download_raw(cfg: dict, force: bool = False) -> dict:
    """Snapshot every source once.  Existing files are never overwritten without ``force``."""
    import yfinance as yf

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = {}

    def record(path: Path, source: str) -> None:
        manifest[path.name] = {"source": source, "sha256": _sha(path), "retrieved_utc": now}

    def snapshot_yahoo(ticker: str) -> None:
        path = _yahoo_path(ticker)
        if force or not path.exists():
            frame = yf.download(ticker, start=cfg["sample"]["download_start"], end="2026-02-01",
                                progress=False, auto_adjust=False)
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            frame.to_csv(path)
        record(path, f"yahoo:{ticker}")

    for asset, spec in cfg["assets"].items():
        snapshot_yahoo(spec["yahoo"])
        snapshot_yahoo(QC_ETFS[asset])

        path = RAW_DIR / f"cboe_{spec['iv']}.csv"
        if force or not path.exists():
            _fetch(CBOE_URL.format(spec["iv"]), path)
        record(path, CBOE_URL.format(spec["iv"]))

        for series in (spec["fred_iv"], spec["fred_index"]):
            if series:
                path = RAW_DIR / f"fred_{series}.csv"
                if force or not path.exists():
                    _fetch(FRED_URL.format(series), path)
                record(path, FRED_URL.format(series))

    write_json(RAW_DIR / "manifest.json", manifest)
    return manifest


def _read_yahoo(ticker: str, field: str = "Close") -> pd.Series:
    frame = pd.read_csv(_yahoo_path(ticker), index_col=0, parse_dates=True)
    return frame[field].astype(float).dropna().rename("close")


def _read_cboe(name: str) -> pd.Series:
    frame = pd.read_csv(RAW_DIR / f"cboe_{name}.csv")
    frame["DATE"] = pd.to_datetime(frame["DATE"], format="%m/%d/%Y")
    return frame.set_index("DATE")["CLOSE"].astype(float).where(lambda s: s > 0).dropna().sort_index()


def _read_fred(series: str) -> pd.Series:
    frame = pd.read_csv(RAW_DIR / f"fred_{series}.csv", parse_dates=["observation_date"])
    return pd.to_numeric(frame.set_index("observation_date")[series], errors="coerce").dropna().sort_index()


def _read_legacy(key: str) -> pd.Series | None:
    entry = LEGACY_FILES.get(key)
    if entry is None or not entry[0].exists():
        return None
    frame = pd.read_csv(entry[0])
    idx = pd.to_datetime(frame[entry[1]], utc=True).dt.tz_localize(None).dt.normalize()
    return pd.Series(pd.to_numeric(frame[entry[2]], errors="coerce").values, index=idx).dropna().sort_index()


def _compare(a: pd.Series, b: pd.Series, tol: float, start, end) -> dict:
    a, b = a.loc[start:end], b.loc[start:end]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return {"n_common": 0}
    rel = (a.loc[common] / b.loc[common] - 1).abs()
    return {
        "n_common": int(len(common)),
        "only_in_primary": [str(d.date()) for d in a.index.difference(b.index)][:40],
        "only_in_other": [str(d.date()) for d in b.index.difference(a.index)][:40],
        "max_abs_rel_diff": float(rel.max()),
        "n_above_tol": int((rel > tol).sum()),
        "dates_above_tol": [str(d.date()) for d in rel[rel > tol].index][:40],
    }


# ------------------------------------------------------------------------------------
# Targets and predictors
# ------------------------------------------------------------------------------------
def calendar_target(dates: pd.DatetimeIndex, ret: np.ndarray, days: int):
    """Y_{t,days} = sum of r_s^2 over t < s <= t + `days` calendar days (Eq. 12).

    A label is complete only if t + days does not pass the last available date; the count
    of summands varies with weekends and holidays and is returned, never normalised away.
    """
    r2 = np.nan_to_num(ret) ** 2
    cum = np.concatenate([[0.0], np.cumsum(r2)])
    pos = np.arange(len(dates))
    ends = dates + pd.Timedelta(days=days)
    last = np.searchsorted(dates.values, ends.values, side="right") - 1
    y = cum[last + 1] - cum[pos + 1]
    n_obs = last - pos
    complete = np.asarray(ends <= dates[-1])
    y = np.where(complete & (n_obs > 0), y, np.nan)
    return y, n_obs, pd.DatetimeIndex(ends)


def trading_target(dates: pd.DatetimeIndex, ret: np.ndarray, h: int):
    r2 = np.nan_to_num(ret) ** 2
    cum = np.concatenate([[0.0], np.cumsum(r2)])
    n = len(dates)
    y = np.full(n, np.nan)
    ends = np.full(n, np.datetime64("NaT", "ns"))
    if n > h:
        y[: n - h] = cum[h + 1 :] - cum[1 : n - h + 1]
        ends[: n - h] = dates.values[h:]
    return y, np.full(n, h), pd.DatetimeIndex(ends)


def history_features(close: pd.Series) -> pd.DataFrame:
    """Price-history predictors; each value at t uses closes up to and including t only."""
    logc = np.log(close)
    ret = logc.diff()
    r2 = ret**2
    return pd.DataFrame({
        "ret": ret,
        "rv_d": r2,
        "rv_w": r2.rolling(5).mean(),
        "rv_m": r2.rolling(22).mean(),
        "rv_neg_w": r2.where(ret < 0, 0.0).where(ret.notna()).rolling(5).mean(),
        "ret_1": ret,
        "ret_5": ret.rolling(5).sum(),
        "dd_22": logc - logc.rolling(22).max(),
    })


def _gjr_filter(params, ret):
    omega, alpha, gamma, beta = params
    n = len(ret)
    h = np.empty(n + 1)
    h[0] = np.var(ret)
    for t in range(n):
        h[t + 1] = omega + (alpha + gamma * (ret[t] < 0)) * ret[t] ** 2 + beta * h[t]
    return h


def _gjr_fit(ret: np.ndarray) -> np.ndarray:
    def nll(theta):
        if theta[0] <= 0 or min(theta[1:]) < 0 or theta[1] + theta[2] / 2 + theta[3] >= 0.9995:
            return 1e12
        h = _gjr_filter(theta, ret)[:-1]
        if np.any(h <= 0):
            return 1e12
        return 0.5 * np.sum(np.log(h) + ret**2 / h)

    var = np.var(ret)
    x0 = np.array([var * 0.02, 0.02, 0.12, 0.90])
    res = minimize(nll, x0, method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-4})
    return res.x


def garch_forecast(dates: pd.DatetimeIndex, ret: np.ndarray, n_ahead: np.ndarray, start_year: int) -> np.ndarray:
    """Price-only GJR-GARCH forecast of the variance over the next n_ahead sessions.

    Parameters for each calendar year are estimated on returns strictly before that year,
    so every value is ex ante.  Used as a non-option control signal, spec ``garch``.
    """
    out = np.full(len(dates), np.nan)
    good = np.isfinite(ret)
    years = dates.year.values
    for year in range(start_year, int(years.max()) + 1):
        mask = years == year
        if not mask.any():
            continue
        hist = ret[good & (years < year)]
        if len(hist) < 500:
            continue
        theta = _gjr_fit(hist)
        omega, alpha, gamma, beta = theta
        persistence = alpha + gamma / 2 + beta
        uncond = omega / max(1 - persistence, 1e-6)
        first_valid = np.argmax(good)
        h = _gjr_filter(theta, ret[first_valid:])
        for i in np.flatnonzero(mask):
            pos = i - first_valid
            if pos < 0 or n_ahead[i] < 1:
                continue
            k = np.arange(int(n_ahead[i]))
            out[i] = np.sum(uncond + persistence**k * (h[pos + 1] - uncond))
    return out


# ------------------------------------------------------------------------------------
# Panel
# ------------------------------------------------------------------------------------
def build_panel(cfg: dict, asset: str) -> tuple[pd.DataFrame, dict]:
    spec = cfg["assets"][asset]
    start, end = pd.Timestamp(cfg["sample"]["start"]), pd.Timestamp(cfg["sample"]["end"])
    dl_start = pd.Timestamp(cfg["sample"]["download_start"])

    yahoo = _read_yahoo(spec["yahoo"]).loc[dl_start:end]
    yahoo = yahoo[~yahoo.index.duplicated(keep="first")].sort_index()
    # The session calendar comes from the market-data file (FRED carries non-sessions such as
    # 2019-04-19).  The value is the issuer's close through FRED wherever it exists: Yahoo has
    # at least one wrong SPX print in the sample (2021-08-11: 4442.41 vs the official 4447.70).
    official = _read_fred(spec["fred_index"]).reindex(yahoo.index) if spec["fred_index"] else None
    close = official.combine_first(yahoo).rename("close") if official is not None else yahoo
    cboe, fred_iv = _read_cboe(spec["iv"]), _read_fred(spec["fred_iv"])
    iv = cboe.combine_first(fred_iv).rename("iv_close")   # issuer file first; FRED only pre-dates it

    frame = history_features(close)
    frame.insert(0, "close", close)
    frame["close_source"] = "yahoo" if official is None else np.where(official.notna(), "fred", "yahoo")
    frame["iv_close"] = iv.reindex(frame.index)
    frame["iv_source"] = np.where(frame.index.isin(cboe.index), "cboe",
                                  np.where(frame.index.isin(fred_iv.index), "fred", "missing"))
    dates = pd.DatetimeIndex(frame.index)
    ret = frame["ret"].to_numpy()

    for name, (kind, h) in TARGETS.items():
        y, n_obs, ends = (calendar_target if kind == "calendar" else trading_target)(dates, ret, h)
        frame[name], frame[f"n_{name}"], frame[f"end_{name}"] = y, n_obs, ends

    # Eq. (13) and its non-affine alternatives.  Constant rescalings such as annualised
    # variance or 21/252 give identical forecasts in all families (asserted in the tests).
    frame["V"] = (frame["iv_close"] / 100.0) ** 2 * 30.0 / 365.0
    frame["V_lag1"] = frame["V"].shift(1)
    frame["V_vol"] = frame["iv_close"] / 100.0
    weekdays = np.array([np.busday_count((d + pd.Timedelta(days=1)).date(),
                                         (d + pd.Timedelta(days=31)).date()) for d in dates])
    frame["V_wkcal"] = (frame["iv_close"] / 100.0) ** 2 * weekdays / 261.0
    frame["V_garch"] = garch_forecast(dates, ret, weekdays, start_year=dl_start.year + 2)

    frame["iv_lag_ok"] = frame["iv_close"].shift(1).notna()
    panel = frame.loc[frame.index >= dl_start].reset_index(names="date")
    panel["in_sample"] = (panel["date"] >= start) & (panel["date"] <= end)
    return panel, _panel_qc(cfg, asset, panel, yahoo, cboe, fred_iv)


def _etf_gap_screen(panel: pd.DataFrame, etf: pd.Series, threshold: float = 0.005) -> dict:
    """Sessions where the index return leaves the ETF total-return move by more than `threshold`."""
    frame = panel.set_index("date")
    gap = (frame["ret"] - np.log(etf).diff().reindex(frame.index)).dropna()
    flagged = gap[gap.abs() > threshold]
    return {"etf_sessions": int(len(gap)), "threshold": threshold,
            "n_flagged": int(len(flagged)),
            "flagged": {str(d.date()): round(float(v), 5) for d, v in flagged.items()}}


def _panel_qc(cfg, asset, panel, yahoo, cboe, fred_iv) -> dict:
    spec = cfg["assets"][asset]
    start, end = cfg["sample"]["start"], cfg["sample"]["end"]
    insample = panel[panel["in_sample"]]
    labeled = insample[insample["Y30"].notna()]
    gaps = insample["date"].diff().dt.days
    qc = {
        "asset": asset,
        "n_sessions_in_sample": int(len(insample)),
        "first_date": str(insample["date"].iloc[0].date()),
        "last_date": str(insample["date"].iloc[-1].date()),
        "last_labeled_origin_Y30": str(labeled["date"].iloc[-1].date()),
        "missing_iv_on_index_sessions": [str(d.date()) for d in insample.loc[insample["iv_close"].isna(), "date"]],
        "iv_source_counts": insample["iv_source"].value_counts().to_dict(),
        "duplicate_dates": int(panel["date"].duplicated().sum()),
        "zero_returns": [str(d.date()) for d in insample.loc[insample["ret"] == 0, "date"]],
        "abs_return_above_8pct": [str(d.date()) for d in insample.loc[insample["ret"].abs() > 0.08, "date"]],
        "calendar_gaps_above_4_days": [str(d.date()) for d in insample.loc[gaps > 4, "date"]],
        "n_summands_Y30_complete_labels": labeled["n_Y30"].value_counts().sort_index().to_dict(),
        "close_source_counts": insample["close_source"].value_counts().to_dict(),
        "iv_range": [float(insample["iv_close"].min()), float(insample["iv_close"].max())],
        "cross_checks": {},
    }
    checks = qc["cross_checks"]
    checks["iv_cboe_vs_fred"] = _compare(cboe, fred_iv, 0.005, start, end)
    if spec["fred_index"]:
        checks["index_yahoo_vs_fred"] = _compare(yahoo, _read_fred(spec["fred_index"]), 0.0001, start, end)
    checks["index_vs_etf_return_gap"] = _etf_gap_screen(insample, _read_yahoo(QC_ETFS[asset], "Adj Close"))
    legacy_iv = _read_legacy(spec["iv"])
    if legacy_iv is not None:
        checks["iv_cboe_vs_legacy"] = _compare(cboe, legacy_iv, 0.005, start, end)
    legacy_idx = _read_legacy(spec["yahoo"])
    if legacy_idx is not None:
        checks["index_yahoo_vs_legacy"] = _compare(yahoo, legacy_idx, 0.0001, start, end)
    return qc


def panel_path(cfg: dict, asset: str) -> Path:
    profile = cfg.get("panel_tag", cfg["profile"])
    tag = "" if profile == "primary" else f"_{profile}"
    return PROCESSED_DIR / f"panel_{asset}{tag}.parquet"


def save_panel(cfg: dict, asset: str) -> dict:
    panel, qc = build_panel(cfg, asset)
    write_parquet(panel_path(cfg, asset), panel)
    write_json(panel_path(cfg, asset).with_suffix(".qc.json"), qc)
    return qc


# ------------------------------------------------------------------------------------
# Arrays for one (asset, spec)
# ------------------------------------------------------------------------------------
@dataclass
class SpecData:
    """Aligned arrays for one asset and spec.  Row i is forecast origin dates[i]."""
    asset: str
    spec: str
    dates: pd.DatetimeIndex
    X: np.ndarray
    V: np.ndarray
    y: np.ndarray
    label_end: pd.DatetimeIndex
    n_obs: np.ndarray
    in_sample: np.ndarray
    ret: np.ndarray
    log_iv_vol: np.ndarray

    @property
    def usable(self) -> np.ndarray:
        """Origins with complete predictors in both information sets (common origin set)."""
        return self.in_sample & np.isfinite(self.X).all(axis=1) & np.isfinite(self.V) & (self.V > 0)


def return_lags(ret: pd.Series, seq_len: int) -> np.ndarray:
    """Columns r_{t-seq_len+1}, ..., r_t (oldest first); all known at the close of t."""
    return np.column_stack([ret.shift(k).to_numpy(float) for k in range(seq_len - 1, -1, -1)])


def load_spec_data(cfg: dict, asset: str, spec: str = "primary") -> SpecData:
    from .config import resolve_spec

    rs = resolve_spec(cfg, spec)
    panel = pd.read_parquet(panel_path(cfg, asset))
    target = rs["target"]
    X = panel[HIST_COLS].to_numpy(float)
    if rs.get("seq_len"):
        # Sequence models read the raw return path; everything else keeps using X[:, :7].
        X = np.column_stack([X, return_lags(panel["ret"], int(rs["seq_len"]))])
    return SpecData(
        asset=asset, spec=spec,
        dates=pd.DatetimeIndex(panel["date"]),
        X=X,
        V=panel[rs["v_col"]].to_numpy(float),
        y=panel[target].to_numpy(float),
        label_end=pd.DatetimeIndex(panel[f"end_{target}"]),
        n_obs=panel[f"n_{target}"].to_numpy(float),
        in_sample=panel["in_sample"].to_numpy(bool),
        ret=panel["ret"].to_numpy(float),
        log_iv_vol=np.log(panel["iv_close"].to_numpy(float) / 100.0),
    )
