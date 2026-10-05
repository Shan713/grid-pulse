"""Fit the common model suite for every region and save reproducible results."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.grid_pulse_analysis import classify_model, load_region, run_forecasts  # noqa: E402


def main() -> None:
    result_dir = ROOT / "results"
    result_dir.mkdir(exist_ok=True)
    metrics, diagnostics, forecasts = [], [], []

    for region in ["IN-NO", "IN-WE", "IN-SO"]:
        print(f"Running {region}...")
        bundle = run_forecasts(region, load_region(region, ROOT))

        region_metrics = bundle.metrics.copy()
        region_metrics.insert(0, "Region", region)
        region_metrics.insert(3, "Model family", region_metrics["Model"].map(classify_model))
        metrics.append(region_metrics)

        region_diagnostics = bundle.diagnostics.copy()
        region_diagnostics.insert(0, "Region", region)
        diagnostics.append(region_diagnostics)

        region_forecasts = bundle.forecasts.reset_index().rename(columns={"datetime": "Datetime"})
        region_forecasts.insert(0, "Region", region)
        forecasts.append(region_forecasts)
        print(
            f"{region}: winner={bundle.metrics.iloc[0]['Model']}; "
            f"RMSE={bundle.metrics.iloc[0]['RMSE']:.3f}"
        )

    pd.concat(metrics, ignore_index=True).to_csv(result_dir / "forecast_metrics.csv", index=False)
    pd.concat(diagnostics, ignore_index=True).to_csv(result_dir / "residual_diagnostics.csv", index=False)
    pd.concat(forecasts, ignore_index=True).to_csv(result_dir / "test_forecasts.csv", index=False)
    print("Saved model metrics, residual diagnostics, and test forecasts in results/")


if __name__ == "__main__":
    main()
