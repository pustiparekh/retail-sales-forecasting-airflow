"""Create the dataset: 5 stores x 4 categories, daily sales from 2024-01-01 to the given date.

    python scripts/make_dataset.py 2026-09-30        # writes data/raw/retail_sales.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retail_forecast as rf

if __name__ == "__main__":
    end = sys.argv[1] if len(sys.argv) > 1 else "2026-09-30"
    rf.raw_file().unlink(missing_ok=True)
    rf.step_ingest(end)
    df = rf.load_raw()
    print(df.shape, df["date"].min().date(), "->", df["date"].max().date())
