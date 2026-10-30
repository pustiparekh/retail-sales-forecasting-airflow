from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import subprocess
import sys

from airflow.sdk import Param, dag, get_current_context, task

PROJECT_DIR = Path(__file__).resolve().parents[1]
FORECAST_SCRIPT = PROJECT_DIR / "scripts" / "retail_forecast.py"


@dag(
    dag_id="retail_sales_forecast",
    description="Automated retail sales ingestion, validation, model selection and forecasting",
    schedule="0 8 * * *",
    start_date=datetime(2026, 1, 1, tzinfo=ZoneInfo("Asia/Kolkata")),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "student", "retries": 1, "retry_delay": timedelta(minutes=2)},
    params={
        "horizon": Param(14, type="integer", description="Days to forecast"),
        "min_improvement": Param(0.05, type="number", description="Relative WAPE gain over the baseline needed to use an ML model"),
        "wape_alert": Param(0.20, type="number", description="Alert if the previous forecast's WAPE is above this"),
    },
    tags=["retail", "forecasting", "machine-learning"],
)
def retail_sales_forecast():

    @task
    def run_step(step: str):
        if not FORECAST_SCRIPT.is_file():
            raise FileNotFoundError(f"Script not found: {FORECAST_SCRIPT}")
        ctx = get_current_context()
        logical = ctx.get("logical_date") or ctx["dag_run"].run_after
        p = ctx["params"]
        cmd = [sys.executable, str(FORECAST_SCRIPT), "--step", step,
               "--as-of", logical.strftime("%Y-%m-%d"),
               "--horizon", str(p["horizon"]),
               "--min-improvement", str(p["min_improvement"]),
               "--wape-alert", str(p["wape_alert"])]
        res = subprocess.run(cmd, cwd=str(PROJECT_DIR), capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr)
            raise RuntimeError(f"Step '{step}' failed (exit code {res.returncode}):\n{res.stderr[-2000:]}")
        return res.stdout[-3000:]  # visible in the Airflow UI under the task's XCom tab

    ingest = run_step.override(task_id="ingest")("ingest")
    validate = run_step.override(task_id="validate")("validate")
    evaluate = run_step.override(task_id="evaluate")("evaluate")
    forecast = run_step.override(task_id="forecast")("forecast")
    report = run_step.override(task_id="report")("report")

    ingest >> validate >> evaluate >> forecast >> report


retail_sales_forecast()
