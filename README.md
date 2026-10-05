# Grid Pulse

Grid Pulse is a reproducible hourly carbon-intensity forecasting study for three Electricity Maps regions in India:

- `IN-NO`: Northern India
- `IN-WE`: Western India
- `IN-SO`: Southern India

Every region uses the same complete timeline, from **2017-01-01 00:00 UTC** through **2026-09-02 15:00 UTC**. Each carbon-intensity file and renewable-percentage file contains **84,760 hourly rows**, with no missing hours or duplicate timestamps.

## Contributors

| Region | Contributor |
|---|---|
| IN-NO | Tejeshwar C D R |
| IN-SO | Shantharam P |
| IN-WE | Harish Krishna B |

## Main outputs

| File | Purpose |
|---|---|
| `IN_NO/main.ipynb` | Complete Northern India analysis and forecast comparison |
| `IN_WE/main.ipynb` | Complete Western India analysis and forecast comparison |
| `IN_SO/main.ipynb` | Complete Southern India analysis and forecast comparison |
| `cross_region_comparison.ipynb` | Regional comparison and carbon-aware routing analysis |
| `results/forecast_metrics.csv` | Accuracy metrics for every model and region |
| `results/residual_diagnostics.csv` | Bias, residual spread, normality, and Ljung-Box checks |
| `results/test_forecasts.csv` | Actual values and all forecasts for the common test week |

The notebooks contain responsive Plotly charts, clear written discussion, and the full marked sections: research question, dataset understanding, visualization, trend/seasonality, stationarity, ACF/PACF, hypothesis, ARIMA/SARIMA/SARIMAX, Holt-Winters, ML/DL, baselines, accuracy, comparison, residuals, conclusion, and individual contribution. Chart sizing, margins, labels, and controls are standardized for Jupyter, VS Code, and presentation mode.

## Required model combinations

The shared pipeline implements these four combinations in every region:

1. ARIMA on stationary lag-24-differenced data
2. ARIMA on raw/non-stationary data with internal differencing
3. SARIMA on stationary lag-24-differenced data
4. SARIMA on raw/non-stationary data with internal seasonal differencing

It also evaluates leakage-aware SARIMAX, two Holt-Winters specifications, Random Forest, an MLP neural network, and two simple baselines.

## Repository structure

```text
grid-pulse/
├── IN_NO/
│   ├── IN-NO_carbon_intensity.csv
│   ├── IN-NO_renewable_percentage.csv
│   └── main.ipynb
├── IN_WE/
│   ├── IN-WE_carbon_intensity.csv
│   ├── IN-WE_renewable_percentage.csv
│   └── main.ipynb
├── IN_SO/
│   ├── IN-SO_carbon_intensity.csv
│   ├── IN-SO_renewable_percentage.csv
│   └── main.ipynb
├── results/
├── scripts/
│   ├── build_notebooks.py
│   ├── fetch_data.py
│   └── run_models.py
├── src/
│   └── grid_pulse_analysis.py
├── cross_region_comparison.ipynb
└── requirements.txt
```

## Reproduce the analysis

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Recalculate the common model results:

```bash
python scripts/run_models.py
```

Open and run the four notebooks from the repository root. Their saved outputs are already included for review.

To rebuild the notebook source consistently after editing the generator:

```bash
python scripts/build_notebooks.py
```

This command creates clean, unexecuted notebooks. Run them afterward to refresh their saved output.

## Refresh the Electricity Maps data

Data refresh requires an Electricity Maps API key. Keep the key outside the repository:

```bash
export ELECTRICITYMAPS_API_KEY="your-key"
python scripts/fetch_data.py --zones IN-NO IN-WE IN-SO --overwrite
```

`scripts/fetch_data.py` requests short hourly windows, retries temporary failures, removes duplicate timestamps, sorts the result, and refuses to save an incomplete common timeline.

## Evaluation design

- 180 days for training
- 7 days for validation
- 7 untouched days for final testing
- 168-hour multi-step forecast for every model
- MAE, RMSE, MAPE, sMAPE, and R² on the same timestamps
- residual bias, standard deviation, Jarque-Bera normality, and lag-24 Ljung-Box checks

SARIMAX does not receive the true future renewable percentage. It receives a seasonal-naïve renewable forecast available at the prediction cutoff. This prevents future-data leakage.

## Data source and limitations

The source is the [Electricity Maps API](https://portal.electricitymaps.com/docs/getting-started). Historical values may include provider estimates. Results use one final seven-day holdout period; rolling-origin validation is recommended before production deployment. Carbon-aware regional routing also needs real operational constraints such as capacity, cost, latency, reliability, and data residency.
