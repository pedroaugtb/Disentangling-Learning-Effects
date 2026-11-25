# Disentangling Learning Effects and Predictive Information in Volatility Forecasting

Code for the paper *Disentangling Learning Effects and Predictive Information in Volatility
Forecasting* (Chagas, Buzelin, Aquino, Bento, Meira Jr., Arbex Valle — UFMG).

Does option-implied volatility (IV) make variance forecasts more accurate, or does it only change
how a model learns from price history? For each learner we fit a pair of forecasters on identical
rows, seeds and candidates — a base `f0(X)` on price history only and an augmented `f1(X, V)` that
also sees the implied variance — and split their difference exactly:

```
f1(X,V) - f0(X)  =  [P1(X) - f0(X)]  +  [f1(X,V) - P1(X)]        P1(x) = E[f1(x,V) | X=x]
       D                  HFS                    ISD
```

- **HFS (history-function shift):** how training with IV changed the response to history.
- **ISD (information-state differentiation):** the effect of observing today's IV surprise.

The loss gain splits the same way into a **teacher gain** `L(f0) − L(P1)` and an **input gain**
`L(P1) − L(f1)`. Claims are tested against a persistence-matched placebo signal, a refit
bootstrap (model-estimation uncertainty) and a Monte Carlo audit, under pre-specified decision
rules (`docs/PREANALYSIS.md`).

**Setting.** S&P 500 / VIX (primary) and Nasdaq-100 / VXN (pre-specified replication), daily
2010–2025, 30-day realised variance target (plus 5 and 10 sessions), walk-forward out-of-sample
from 2017 with monthly refits. Families: HAR, elastic net, gradient-boosted trees (XGBoost), plus
an MLP and a temporal CNN as a neural extension.

**Main finding.** IV consistently attenuates how all families respond to historical volatility, in
both markets and beyond the placebo, yet this does not translate into statistically detectable
30-day accuracy gains once model-estimation uncertainty is accounted for.

## Layout

| Path | Contents |
|---|---|
| `src/ivbehav/` | library: data, walk-forward folds, models, conditional laws, decomposition, statistics, inference, simulation, teacher/input split |
| `scripts/` | numbered experiments (`00`–`16`) and the `run_*.sh` drivers |
| `configs/` | `primary.yaml` (design and budgets), `teacher*.yaml` (addenda C–E), `nn*.yaml` (neural), `dev.yaml` (pre-2017 debug run) |
| `docs/` | `PREANALYSIS.md` (frozen hypotheses and claim rules), `DESIGN_NOTES.md` (decisions and data findings) |
| `tests/` | theory, leakage and estimator tests on synthetic data |
| `notebooks/` | `paper_figures.ipynb` and `ivviz.py` for the paper's figures |
| `data/raw/manifest.json` | sources and sha256 of the data snapshots used in the paper |

| Script | Purpose |
|---|---|
| `00_build_data.py` | download raw series (Cboe, FRED, Yahoo) and build the per-asset panels |
| `design_check.py` | outcome-blind design check |
| `01_forecasts.py` | walk-forward fits of `f0` / `f1` for every family |
| `02_simulation.py`, `13_teacher_simulation.py` | Monte Carlo audit |
| `03_forecast_eval.py` | accuracy scores, DM / GW tests |
| `04_decomposition.py` | HFS / ISD decomposition under laws A and B |
| `05_response.py` | ALE response curves and contrasts |
| `06_placebo.py` | persistence-matched placebo |
| `07_matching.py` | matched-pair validation of ISD |
| `08_preonset.py`, `09_multiplicity.py` | exploratory paths, fitting-variability specs |
| `10_refit_bootstrap.py` | refit bootstrap |
| `11_collect_primary.py` | behavioural hierarchy (P1–P3) |
| `12_teacher_input.py`, `14_collect_teacher.py`, `15_collect_short.py` | teacher / input gains and claims (C1–C3, D0–D3) |
| `16_paper_tables.py` | LaTeX tables of the paper |

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.lock.txt
.venv/bin/python scripts/00_build_data.py          # downloads data/raw, builds data/processed
.venv/bin/python -m pytest tests -q

scripts/run_all.sh all SPX 24                      # stage, asset, parallel jobs
scripts/run_all.sh core NDX 24                     # replication
scripts/run_final.sh 40 3                          # neural extension (GPU) + teacher/input split
scripts/run_short.sh 40                            # short-horizon confirmation
scripts/run_amend.sh 56                            # refit budget raised to 2000 draws
.venv/bin/python scripts/16_paper_tables.py
```

Single stages: `check`, `simulation`, `forecasts`, `behavior`, `placebo`, `refit`,
`multiplicity`, `preonset`, `collect`, `sensitivity`. Runs are resumable; outputs go to
`results/<profile>/` and every output stores the hash of its config. For a fast debug pass on
pre-2017 data only: `scripts/run_all.sh core SPX 12 dev.yaml`.

The raw price data are not redistributed. `00_build_data.py` downloads them again, and
`data/raw/manifest.json` records the sha256 of the snapshots used in the paper (retrieved
2026-11-03). FRED serves only the last ten years of the S&P 500 close, so a later download can
differ slightly from the paper's panel; see `docs/DESIGN_NOTES.md` for how sources are combined.
