"""Retail sales forecasting pipeline (steps are run one by one by Airflow, or all at once by hand).

    python scripts/retail_forecast.py --step all --as-of 2026-10-01

Steps: ingest -> validate -> evaluate -> forecast -> report
Every step only uses data up to --as-of, so runs are reproducible and each daily run can
score the forecast made by an earlier run.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no screen on an Airflow worker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
STORES = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata"]
CATEGORIES = ["Grocery", "Electronics", "Apparel", "Home"]
N_SERIES = len(STORES) * len(CATEGORIES)
DATA_START = pd.Timestamp("2024-01-01")
LAGS = (1, 7, 14, 28)
HOLIDAYS = {"01-01", "01-26", "08-15", "10-02", "12-25",            # fixed every year
            "2024-11-01", "2025-10-20", "2026-11-08", "2027-10-29"}  # Diwali (varies)


def root() -> Path:
    r = Path(os.environ.get("RETAIL_HOME", BASE_DIR))
    return r


def path(*parts) -> Path:
    p = root().joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _d(x) -> pd.Timestamp:
    return pd.Timestamp(x).normalize()


# =============================================================== DATA GENERATION
def _params() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "base": rng.uniform(20_000, 150_000, N_SERIES),
        "growth": rng.uniform(0.05, 0.15, N_SERIES),
        "weekend": rng.uniform(1.0, 1.5, N_SERIES),
        "amp": rng.uniform(0.08, 0.25, N_SERIES),
        "phase": rng.uniform(0, 2 * np.pi, N_SERIES),
        "promo_up": rng.uniform(1.2, 1.5, N_SERIES),
        "price": rng.uniform(150, 1500, N_SERIES),
    })


def is_holiday(d) -> int:
    d = _d(d)
    return int(d.strftime("%m-%d") in HOLIDAYS or d.strftime("%Y-%m-%d") in HOLIDAYS)


@lru_cache(maxsize=None)
def _promo_array(day_key: str) -> tuple:
    rng = np.random.default_rng(int(day_key.replace("-", "")) + 1)
    return tuple((rng.random(N_SERIES) < 0.12).astype(int))


def planned_promo(d) -> np.ndarray:
    """Promotion plan (known in advance, like a marketing calendar). 1 = on promo."""
    return np.array(_promo_array(_d(d).strftime("%Y-%m-%d")))


def promo_share(d, series: str) -> float:
    grid = planned_promo(d).reshape(len(STORES), len(CATEGORIES))
    return float(grid.mean() if series == "TOTAL" else grid[STORES.index(series)].mean())


def generate_day(d) -> pd.DataFrame:
    d = _d(d)
    p = _params()
    t = (d - DATA_START).days
    weekly = np.where(d.dayofweek >= 5, p["weekend"], 1.0)
    yearly = 1 + p["amp"] * np.sin(2 * np.pi * d.dayofyear / 365.25 + p["phase"])
    promo = planned_promo(d)
    hol = 1.3 if is_holiday(d) else 1.0
    noise = np.random.default_rng(int(d.strftime("%Y%m%d")) + 2).lognormal(0, 0.06, N_SERIES)
    sales = p["base"] * (1 + p["growth"] * t / 365) * weekly * yearly * np.where(promo == 1, p["promo_up"], 1.0) * hol * noise
    return pd.DataFrame({
        "date": d,
        "store": np.repeat(STORES, len(CATEGORIES)),
        "category": CATEGORIES * len(STORES),
        "units_sold": np.round(sales / p["price"]).astype(int),
        "sales": np.round(sales, 2),
        "promo": promo,
    })


# ====================================================================== LOADING
def raw_file() -> Path:
    return path("data", "raw", "retail_sales.csv")


def load_raw(as_of=None) -> pd.DataFrame:
    if not raw_file().exists():
        raise FileNotFoundError("data/raw/retail_sales.csv missing - run the ingest step first")
    df = pd.read_csv(raw_file(), parse_dates=["date"])
    if as_of is not None:
        df = df[df["date"] <= _d(as_of)]
    return df


def resolve_as_of(as_of) -> pd.Timestamp:
    return _d(as_of) if as_of else load_raw()["date"].max()


def get_series(as_of) -> dict[str, pd.Series]:
    df = load_raw(as_of)
    out = {"TOTAL": df.groupby("date")["sales"].sum().asfreq("D").interpolate()}
    for s in STORES:
        out[s] = df[df["store"] == s].groupby("date")["sales"].sum().asfreq("D").interpolate()
    return out


# ================================================================== FEATURES/MODELS
CAL_FEATS = ["dow", "month", "doy_sin", "doy_cos", "holiday", "promo"]
FEATS = [f"lag_{k}" for k in LAGS] + ["roll7", "roll28"] + CAL_FEATS


def calendar_frame(idx: pd.DatetimeIndex, series: str) -> pd.DataFrame:
    f = pd.DataFrame(index=idx)
    f["dow"] = idx.dayofweek
    f["month"] = idx.month
    f["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    f["holiday"] = [is_holiday(d) for d in idx]
    f["promo"] = [promo_share(d, series) for d in idx]
    return f


def make_features(s: pd.Series, series: str = "TOTAL") -> pd.DataFrame:
    """Features for day t use only days before t (shifted lags/rolling means) + the known calendar."""
    ls = np.log1p(s)
    f = pd.DataFrame(index=ls.index)
    for k in LAGS:
        f[f"lag_{k}"] = ls.shift(k)
    f["roll7"] = ls.shift(1).rolling(7).mean()
    f["roll28"] = ls.shift(1).rolling(28).mean()
    f = f.join(calendar_frame(ls.index, series))
    f["y"] = ls
    return f


class SeasonalNaive:
    """Baseline: the same weekday last week."""
    def fit(self, X, y): return self
    def predict(self, X): return X["lag_7"].to_numpy()


MODELS = {
    "seasonal_naive": lambda: SeasonalNaive(),
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "random_forest": lambda: RandomForestRegressor(n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=-1),
}
BASELINE = "seasonal_naive"


def fit_model(model_key: str, s: pd.Series, series: str = "TOTAL"):
    f = make_features(s, series).dropna()
    return MODELS[model_key]().fit(f[FEATS], f["y"])


def recursive_forecast(model, s: pd.Series, series: str, horizon: int) -> pd.DataFrame:
    """Predict day by day; each prediction becomes a lag for the next day."""
    h = list(np.log1p(s.to_numpy(dtype=float)))
    fdates = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=horizon)
    cal = calendar_frame(fdates, series)
    rows = []
    for t in fdates:
        x = {f"lag_{k}": h[-k] for k in LAGS}
        x["roll7"], x["roll28"] = float(np.mean(h[-7:])), float(np.mean(h[-28:]))
        for c in CAL_FEATS:
            x[c] = cal.loc[t, c]
        yhat = float(model.predict(pd.DataFrame([x])[FEATS])[0])
        h.append(yhat)
        rows.append((t, float(np.expm1(yhat))))
    return pd.DataFrame(rows, columns=["date", "forecast"])


def error_metrics(actual, fc) -> dict:
    actual, fc = np.asarray(actual, float), np.asarray(fc, float)
    return {"wape": float(np.abs(fc - actual).sum() / np.abs(actual).sum()),
            "mae": float(np.abs(fc - actual).mean()), "bias_pct": float((fc - actual).sum() / actual.sum())}


def backtest(model_key: str, s: pd.Series, series="TOTAL", horizon=14, folds=3) -> dict:
    """Rolling-origin backtest: train on the past, forecast `horizon` days, score, repeat."""
    scores = []
    for i in range(folds, 0, -1):
        cut = len(s) - horizon * i
        train, test = s.iloc[:cut], s.iloc[cut:cut + horizon]
        fc = recursive_forecast(fit_model(model_key, train, series), train, series, horizon)
        scores.append(error_metrics(test.to_numpy(), fc["forecast"].to_numpy()))
    return {"model": model_key, **{k: round(float(np.mean([x[k] for x in scores])), 4) for k in scores[0]}}


# ========================================================================== STEPS
def step_ingest(as_of, **_):
    """Make sure data exists up to `as_of` (simulates the new day's sales arriving)."""
    as_of = _d(as_of)
    f = raw_file()
    if f.exists():
        old = pd.read_csv(f, parse_dates=["date"])
        start = old["date"].max() + pd.Timedelta(days=1)
    else:
        old, start = None, DATA_START
    if start <= as_of:
        new = pd.concat([generate_day(d) for d in pd.date_range(start, as_of)], ignore_index=True)
        pd.concat([old, new], ignore_index=True).to_csv(f, index=False) if old is not None else new.to_csv(f, index=False)
        print(f"Ingested {new['date'].nunique()} new day(s): {start.date()} -> {as_of.date()}")
    else:
        print(f"Data already available up to {load_raw()['date'].max().date()}; nothing to ingest")


def step_validate(as_of, **_):
    as_of = resolve_as_of(as_of)
    df = load_raw(as_of)
    problems = []
    need = {"date", "store", "category", "units_sold", "sales", "promo"}
    if not need <= set(df.columns): problems.append(f"missing columns {need - set(df.columns)}")
    if df.duplicated(["date", "store", "category"]).any(): problems.append("duplicate date/store/category rows")
    if df[["sales", "units_sold"]].isna().any().any(): problems.append("missing sales/units values")
    if (df["sales"] < 0).any(): problems.append("negative sales")
    if df["date"].max() != as_of: problems.append(f"no data for {as_of.date()}")
    recent = df[df["date"] > as_of - pd.Timedelta(days=60)]
    if recent["date"].nunique() < 60: problems.append("gaps in the last 60 days")
    if recent.groupby("date").size().ne(N_SERIES).any(): problems.append(f"some days do not have {N_SERIES} store/category rows")
    if len(df["date"].unique()) < 90: problems.append("fewer than 90 days of history")
    if problems:
        raise ValueError("Validation failed: " + "; ".join(problems))
    med = df.groupby(["store", "category"])["sales"].transform("median")
    info = {"as_of": str(as_of.date()), "rows": int(len(df)), "days": int(df["date"].nunique()),
            "first_date": str(df["date"].min().date()), "outlier_rows": int((df["sales"] > 5 * med).sum())}
    path("data", "metrics", "validation.json").write_text(json.dumps(info, indent=2))
    print("Validation passed:", info)


def step_evaluate(as_of, horizon=14, min_improvement=0.05, **_):
    as_of = resolve_as_of(as_of)
    total = get_series(as_of)["TOTAL"]
    results = [backtest(m, total, "TOTAL", horizon) for m in MODELS]
    base = next(r for r in results if r["model"] == BASELINE)
    best = min((r for r in results if r["model"] != BASELINE), key=lambda r: r["wape"])
    gain = (base["wape"] - best["wape"]) / base["wape"]
    chosen = best["model"] if gain >= min_improvement else BASELINE
    sel = {"as_of": str(as_of.date()), "horizon": horizon, "results": results, "best_ml": best["model"],
           "improvement_vs_baseline": round(gain, 4), "min_improvement": min_improvement, "chosen": chosen}
    path("data", "metrics", "selection.json").write_text(json.dumps(sel, indent=2))
    for r in results:
        print(f"  {r['model']:<15} WAPE {r['wape']:.2%}  MAE {r['mae']:,.0f}  bias {r['bias_pct']:+.2%}")
    print(f"Chosen model: {chosen} (best ML improves WAPE by {gain:.1%} vs baseline)")


def step_forecast(as_of, horizon=14, **_):
    as_of = resolve_as_of(as_of)
    sel = json.loads(path("data", "metrics", "selection.json").read_text())
    if sel["as_of"] != str(as_of.date()):
        raise RuntimeError(f"selection.json is for {sel['as_of']}, not {as_of.date()} - run the evaluate step first")
    chosen = sel["chosen"]
    parts = []
    for name, s in get_series(as_of).items():
        model = fit_model(chosen, s, name)
        fc = recursive_forecast(model, s, name, horizon)
        fc.insert(1, "series", name)
        parts.append(fc)
        if name == "TOTAL":
            joblib.dump({"model": model, "model_name": chosen, "trained_through": str(as_of.date())},
                        path("models", "retail_sales_model.joblib"))
    out = pd.concat(parts, ignore_index=True)
    out["forecast"] = out["forecast"].round(2)
    out["model"], out["as_of"] = chosen, as_of.date()
    f = path("data", "forecasts", f"forecast_{as_of:%Y-%m-%d}.csv")
    out.to_csv(f, index=False)
    print(f"Saved {f.name}: {len(out)} rows, model={chosen}")
    print(out[out["series"] == "TOTAL"][["date", "forecast"]].to_string(index=False))


def score_previous_forecast(as_of):
    prev = sorted(p for p in path("data", "forecasts", "x").parent.glob("forecast_*.csv")
                  if pd.Timestamp(p.stem.split("_")[1]) < as_of)
    if not prev:
        return None
    fc = pd.read_csv(prev[-1], parse_dates=["date"])
    fc = fc[fc["series"] == "TOTAL"]
    act = get_series(as_of)["TOTAL"].rename("actual").reset_index().rename(columns={"index": "date"})
    act.columns = ["date", "actual"]
    m = fc.merge(act, on="date")
    if m.empty:
        return None
    res = {"forecast_file": prev[-1].name, "model": str(fc["model"].iloc[0]), "days_scored": int(len(m)),
           **{k: round(v, 4) for k, v in error_metrics(m["actual"], m["forecast"]).items()}}
    log = path("data", "metrics", "forecast_accuracy.csv")
    row = pd.DataFrame([{"scored_on": as_of.date(), **res}])
    if log.exists():
        old = pd.read_csv(log)
        row = pd.concat([old[old["scored_on"] != str(as_of.date())], row], ignore_index=True)
    row.to_csv(log, index=False)
    return res


def step_report(as_of, wape_alert=0.20, **_):
    as_of = resolve_as_of(as_of)
    sel = json.loads(path("data", "metrics", "selection.json").read_text())
    val = json.loads(path("data", "metrics", "validation.json").read_text())
    fc = pd.read_csv(path("data", "forecasts", f"forecast_{as_of:%Y-%m-%d}.csv"), parse_dates=["date"])
    prev = score_previous_forecast(as_of)
    total_fc = fc[fc["series"] == "TOTAL"].set_index("date")["forecast"]
    hist = get_series(as_of)["TOTAL"].tail(56)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(hist.index, hist / 1e6, label="Actual (last 8 weeks)", color="#1D6A72")
    ax.plot(total_fc.index, total_fc / 1e6, label=f"Forecast ({sel['chosen']})", color="#E8A33D", marker="o", ms=3)
    ax.set_ylabel("Sales (million)"); ax.set_title(f"Total daily sales - forecast issued {as_of:%d %b %Y}")
    ax.legend(); ax.grid(alpha=0.25); fig.autofmt_xdate(); fig.tight_layout()
    png = path("data", "reports", f"forecast_{as_of:%Y-%m-%d}.png"); fig.savefig(png, dpi=130); plt.close(fig)

    L = [f"# Retail sales forecast report - {as_of:%d %b %Y}", "",
         f"Data: {val['rows']:,} rows, {val['days']} days ({val['first_date']} to {val['as_of']}), outlier rows flagged: {val['outlier_rows']}.", "",
         f"## Model comparison (rolling backtest, 3 folds x {sel['horizon']} days; lower WAPE is better)", "",
         "| Model | WAPE | MAE | Bias |", "|---|---|---|---|"]
    L += [f"| {r['model']} | {r['wape']:.2%} | {r['mae']:,.0f} | {r['bias_pct']:+.2%} |" for r in sorted(sel["results"], key=lambda r: r["wape"])]
    why = "ML model beat the baseline" if sel["chosen"] != BASELINE else "ML did not beat the baseline by the required margin"
    L += ["", f"**Chosen model: `{sel['chosen']}`** - {why} (improvement {sel['improvement_vs_baseline']:.1%}, required {sel['min_improvement']:.0%}).", "",
          f"## {sel['horizon']}-day forecast", "", f"Total forecast sales: **{total_fc.sum():,.0f}**", "",
          "| Store | Forecast total |", "|---|---|"]
    L += [f"| {n} | {v:,.0f} |" for n, v in fc[fc["series"] != "TOTAL"].groupby("series")["forecast"].sum().sort_values(ascending=False).items()]
    L += ["", "## Accuracy of the previous forecast", ""]
    if prev:
        flag = "  **ALERT: error above threshold**" if prev["wape"] > wape_alert else ""
        L.append(f"`{prev['forecast_file']}` ({prev['model']}), {prev['days_scored']} day(s) scored: WAPE {prev['wape']:.2%}, bias {prev['bias_pct']:+.2%}.{flag}")
    else:
        L.append("No earlier forecast to score yet.")
    L += ["", f"![forecast]({png.name})"]
    out = path("data", "reports", f"report_{as_of:%Y-%m-%d}.md"); out.write_text("\n".join(L))
    print(f"Report written: {out}")
    print("\n" + out.read_text())  # shows the full report in the Airflow task log


STEPS = {"ingest": step_ingest, "validate": step_validate, "evaluate": step_evaluate,
         "forecast": step_forecast, "report": step_report}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=[*STEPS, "all"])
    ap.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: today for ingest, latest data otherwise)")
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--min-improvement", type=float, default=0.05)
    ap.add_argument("--wape-alert", type=float, default=0.20)
    a = ap.parse_args()
    kw = dict(horizon=a.horizon, min_improvement=a.min_improvement, wape_alert=a.wape_alert)
    todo = list(STEPS) if a.step == "all" else [a.step]
    as_of = a.as_of or (str(date.today()) if todo[0] == "ingest" else None)
    for st in todo:
        print(f"\n=== {st} ===")
        STEPS[st](as_of, **kw)
