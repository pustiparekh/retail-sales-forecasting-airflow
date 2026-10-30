import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
AS_OF = "2026-06-30"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RETAIL_HOME", str(tmp_path))


@pytest.fixture
def rf():
    import retail_forecast
    return retail_forecast


def test_generator_deterministic_and_complete(rf):
    a, b = rf.generate_day("2026-03-05"), rf.generate_day("2026-03-05")
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 20 and (a["sales"] > 0).all()


def test_ingest_idempotent_and_appends(rf):
    rf.step_ingest("2026-05-01")
    n1 = len(rf.load_raw())
    rf.step_ingest("2026-05-01")
    assert len(rf.load_raw()) == n1
    rf.step_ingest("2026-05-03")
    assert len(rf.load_raw()) == n1 + 40


def test_validation_rejects_bad_data(rf):
    rf.step_ingest(AS_OF)
    rf.step_validate(AS_OF)  # clean data passes
    df = rf.load_raw()
    df.loc[0, "sales"] = -5
    df.to_csv(rf.raw_file(), index=False)
    with pytest.raises(ValueError, match="negative"):
        rf.step_validate(AS_OF)


def test_no_leakage(rf):
    rf.step_ingest(AS_OF)
    s = rf.get_series(AS_OF)["TOTAL"]
    s2 = s.copy(); s2.iloc[-1] *= 100
    a = rf.make_features(s)[rf.FEATS].iloc[-1].astype(float)
    b = rf.make_features(s2)[rf.FEATS].iloc[-1].astype(float)
    pd.testing.assert_series_equal(a, b)


def test_baseline_is_last_week(rf):
    rf.step_ingest(AS_OF)
    f = rf.make_features(rf.get_series(AS_OF)["TOTAL"]).dropna()
    assert np.allclose(rf.SeasonalNaive().predict(f), f["lag_7"])


def test_models_forecast_and_beat_naive(rf):
    rf.step_ingest(AS_OF)
    s = rf.get_series(AS_OF)["TOTAL"]
    res = {m: rf.backtest(m, s, "TOTAL", 14, 2)["wape"] for m in rf.MODELS}
    assert min(res["ridge"], res["random_forest"]) < res["seasonal_naive"]
    fc = rf.recursive_forecast(rf.fit_model("ridge", s), s, "TOTAL", 14)
    assert len(fc) == 14 and (fc["forecast"] > 0).all()


def test_full_pipeline_and_scoring(rf):
    for d in ("2026-06-30", "2026-07-01"):
        for step in ("ingest", "validate", "evaluate", "forecast", "report"):
            rf.STEPS[step](d, horizon=7, min_improvement=0.05, wape_alert=0.2)
    assert (rf.path("data", "forecasts", "forecast_2026-07-01.csv")).exists()
    assert (rf.path("data", "reports", "report_2026-07-01.md")).exists()
    acc = pd.read_csv(rf.path("data", "metrics", "forecast_accuracy.csv"))
    assert acc["days_scored"].iloc[-1] == 1


def test_dag_imports():
    pytest.importorskip("airflow")
    from airflow.models import DagBag
    bag = DagBag(dag_folder=str(ROOT / "dags"), include_examples=False)
    assert not bag.import_errors, bag.import_errors
    assert {"ingest", "validate", "evaluate", "forecast", "report"} <= set(bag.get_dag("retail_sales_forecast").task_ids)
