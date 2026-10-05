"""Reusable, leakage-aware time-series utilities for the Grid Pulse project."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


START = pd.Timestamp("2017-01-01 00:00:00", tz="UTC")
END = pd.Timestamp("2026-09-02 15:00:00", tz="UTC")
EXPECTED_INDEX = pd.date_range(START, END, freq="h")
SEASONAL_PERIOD = 24
TRAIN_DAYS = 180
VALID_HOURS = 7 * 24
TEST_HOURS = 7 * 24
RANDOM_STATE = 42

REGION_NAMES = {
    "IN-NO": "Northern India",
    "IN-WE": "Western India",
    "IN-SO": "Southern India",
}

FEATURES = [
    "lag_1", "lag_2", "lag_24", "lag_48", "lag_168",
    "roll_mean_24", "roll_std_24", "roll_mean_168",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


@dataclass
class ForecastBundle:
    region: str
    frame: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    train_full: pd.DataFrame
    test: pd.DataFrame
    forecasts: pd.DataFrame
    metrics: pd.DataFrame
    diagnostics: pd.DataFrame
    fit_times: dict[str, float]


def region_folder(region: str) -> Path:
    return Path(region.replace("-", "_"))


def load_region(region: str, root: str | Path = ".") -> pd.DataFrame:
    """Load and align carbon intensity and renewable percentage for one region."""
    root = Path(root)
    folder = root / region_folder(region)
    carbon = pd.read_csv(folder / f"{region}_carbon_intensity.csv")
    renewable = pd.read_csv(folder / f"{region}_renewable_percentage.csv")
    carbon["datetime"] = pd.to_datetime(carbon["datetime"], utc=True)
    renewable["datetime"] = pd.to_datetime(renewable["datetime"], utc=True)
    carbon = carbon[["datetime", "carbonIntensity"]].rename(columns={"carbonIntensity": "carbon"})
    renewable_value = "value" if "value" in renewable.columns else "renewablePercentage"
    renewable = renewable[["datetime", renewable_value]].rename(columns={renewable_value: "renewable_pct"})
    frame = carbon.merge(renewable, on="datetime", how="inner", validate="one_to_one")
    frame = frame.sort_values("datetime").set_index("datetime").astype(float).asfreq("h")
    return frame


def validate_region(frame: pd.DataFrame) -> dict[str, object]:
    """Return explicit quality checks and fail if the common timeline is not exact."""
    checks = {
        "Rows": len(frame),
        "Start": frame.index.min(),
        "End": frame.index.max(),
        "Duplicate timestamps": int(frame.index.duplicated().sum()),
        "Missing target values": int(frame[["carbon", "renewable_pct"]].isna().sum().sum()),
        "Missing hours": int(len(EXPECTED_INDEX.difference(frame.index))),
        "Extra hours": int(len(frame.index.difference(EXPECTED_INDEX))),
    }
    if (
        len(frame) != len(EXPECTED_INDEX)
        or checks["Duplicate timestamps"]
        or checks["Missing target values"]
        or checks["Missing hours"]
        or checks["Extra hours"]
    ):
        raise ValueError(f"Dataset does not match the common project timeline: {checks}")
    return checks


def split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = TRAIN_DAYS * 24 + VALID_HOURS + TEST_HOURS
    model_frame = frame.iloc[-required:].copy()
    train = model_frame.iloc[:-(VALID_HOURS + TEST_HOURS)]
    validation = model_frame.iloc[-(VALID_HOURS + TEST_HOURS):-TEST_HOURS]
    test = model_frame.iloc[-TEST_HOURS:]
    train_full = pd.concat([train, validation])
    return model_frame, train, validation, test, train_full


def accuracy_row(actual: pd.Series, predicted: pd.Series | np.ndarray, model: str) -> dict[str, float | str]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    error = actual_values - predicted_values
    safe_actual = np.where(np.abs(actual_values) < 1e-8, np.nan, actual_values)
    denominator = np.abs(actual_values) + np.abs(predicted_values)
    return {
        "Model": model,
        "MAE": mean_absolute_error(actual_values, predicted_values),
        "RMSE": np.sqrt(mean_squared_error(actual_values, predicted_values)),
        "MAPE (%)": np.nanmean(np.abs(error / safe_actual)) * 100,
        "sMAPE (%)": np.mean(2 * np.abs(error) / (denominator + 1e-8)) * 100,
        "R²": r2_score(actual_values, predicted_values),
    }


def invert_seasonal_difference(
    difference_forecast: pd.Series | np.ndarray,
    level_history: pd.Series | np.ndarray,
    period: int = SEASONAL_PERIOD,
) -> np.ndarray:
    reconstructed = list(np.asarray(level_history, dtype=float))
    output = []
    for difference in np.asarray(difference_forecast, dtype=float):
        level = difference + reconstructed[-period]
        reconstructed.append(level)
        output.append(level)
    return np.asarray(output)


def make_supervised(series: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"target": series})
    for lag in [1, 2, 24, 48, 168]:
        frame[f"lag_{lag}"] = series.shift(lag)
    shifted = series.shift(1)
    frame["roll_mean_24"] = shifted.rolling(24).mean()
    frame["roll_std_24"] = shifted.rolling(24).std()
    frame["roll_mean_168"] = shifted.rolling(168).mean()
    frame["hour_sin"] = np.sin(2 * np.pi * frame.index.hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame.index.hour / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * frame.index.dayofweek / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * frame.index.dayofweek / 7)
    return frame.dropna()


def _feature_vector(history: list[float], timestamp: pd.Timestamp) -> pd.DataFrame:
    values = np.asarray(history, dtype=float)
    values_out = [[
        values[-1], values[-2], values[-24], values[-48], values[-168],
        values[-24:].mean(), values[-24:].std(ddof=1), values[-168:].mean(),
        np.sin(2 * np.pi * timestamp.hour / 24),
        np.cos(2 * np.pi * timestamp.hour / 24),
        np.sin(2 * np.pi * timestamp.dayofweek / 7),
        np.cos(2 * np.pi * timestamp.dayofweek / 7),
    ]]
    return pd.DataFrame(values_out, columns=FEATURES)


def recursive_forecast(model, history_series: pd.Series, future_index: pd.DatetimeIndex) -> np.ndarray:
    history = list(np.asarray(history_series, dtype=float))
    predictions = []
    for timestamp in future_index:
        value = float(model.predict(_feature_vector(history, timestamp))[0])
        history.append(value)
        predictions.append(value)
    return np.asarray(predictions)


def run_forecasts(region: str, frame: pd.DataFrame) -> ForecastBundle:
    """Fit every required model and score one common seven-day test period."""
    validate_region(frame)
    model_frame, train, validation, test, train_full = split_frame(frame)
    y_train = train_full["carbon"]
    y_test = test["carbon"]
    stationary_train = y_train.diff(SEASONAL_PERIOD).dropna()

    predictions: OrderedDict[str, pd.Series] = OrderedDict()
    fit_times: dict[str, float] = {}

    def register(name: str, values, started: float | None = None) -> None:
        predictions[name] = pd.Series(np.asarray(values, dtype=float), index=test.index)
        fit_times[name] = 0.0 if started is None else perf_counter() - started

    register("Baseline: naive last value", np.repeat(y_train.iloc[-1], TEST_HOURS))
    register("Baseline: seasonal naive (24h)", np.resize(y_train.iloc[-24:].to_numpy(), TEST_HOURS))

    started = perf_counter()
    model = ARIMA(stationary_train, order=(2, 0, 1), trend="c").fit()
    prediction = invert_seasonal_difference(model.forecast(TEST_HOURS), y_train.iloc[-24:])
    register("ARIMA stationary: (2,0,1) on Δ24", prediction, started)

    started = perf_counter()
    model = ARIMA(y_train, order=(2, 1, 1), trend="t").fit()
    register("ARIMA non-stationary: (2,1,1)", model.forecast(TEST_HOURS), started)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        started = perf_counter()
        model = SARIMAX(
            stationary_train,
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 24),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=75)
        prediction = invert_seasonal_difference(model.forecast(TEST_HOURS), y_train.iloc[-24:])
        register("SARIMA stationary: (1,0,1)(1,0,1,24) on Δ24", prediction, started)

        started = perf_counter()
        model = SARIMAX(
            y_train,
            order=(1, 0, 1),
            seasonal_order=(1, 1, 1, 24),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=75)
        register("SARIMA non-stationary: (1,0,1)(1,1,1,24)", model.forecast(TEST_HOURS), started)

        # The future renewable share is not revealed to the model.  It is forecast
        # with the same-hour-yesterday rule, which is available at the cutoff.
        renewable_future = np.resize(train_full["renewable_pct"].iloc[-24:].to_numpy(), TEST_HOURS)
        started = perf_counter()
        model = SARIMAX(
            y_train,
            exog=train_full[["renewable_pct"]],
            order=(1, 0, 0),
            seasonal_order=(1, 1, 0, 24),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=75)
        prediction = model.forecast(TEST_HOURS, exog=pd.DataFrame(
            {"renewable_pct": renewable_future}, index=test.index
        ))
        register("SARIMAX + forecast renewable %", prediction, started)

    started = perf_counter()
    model = ExponentialSmoothing(
        y_train,
        trend=None,
        seasonal="add",
        seasonal_periods=24,
        initialization_method="estimated",
    ).fit(optimized=True, use_brute=False)
    register("Holt-Winters additive seasonal", model.forecast(TEST_HOURS), started)

    started = perf_counter()
    model = ExponentialSmoothing(
        y_train,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=24,
        initialization_method="estimated",
    ).fit(optimized=True, use_brute=False)
    register("Holt-Winters damped trend", model.forecast(TEST_HOURS), started)

    supervised = make_supervised(y_train)
    x_train, y_ml = supervised[FEATURES], supervised["target"]

    started = perf_counter()
    forest = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    forest.fit(x_train, y_ml)
    register("ML: Random Forest", recursive_forecast(forest, y_train, test.index), started)

    started = perf_counter()
    network = Pipeline([
        ("scale", StandardScaler()),
        ("network", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        )),
    ])
    network.fit(x_train, y_ml)
    register("DL: MLP neural network", recursive_forecast(network, y_train, test.index), started)

    forecast_frame = pd.DataFrame(predictions)
    forecast_frame.insert(0, "Actual", y_test)
    metrics = pd.DataFrame([
        accuracy_row(y_test, forecast_frame[name], name) for name in predictions
    ])
    metrics["Fit time (s)"] = metrics["Model"].map(fit_times)
    metrics = metrics.sort_values("RMSE").reset_index(drop=True)
    metrics.insert(0, "Rank", np.arange(1, len(metrics) + 1))
    best_baseline = metrics.loc[metrics["Model"].str.startswith("Baseline"), "RMSE"].min()
    metrics["RMSE improvement vs best baseline (%)"] = (
        (best_baseline - metrics["RMSE"]) / best_baseline * 100
    )

    diagnostic_rows = []
    for name in predictions:
        residual = y_test - forecast_frame[name]
        ljung_p = acorr_ljungbox(residual, lags=[24], return_df=True).loc[24, "lb_pvalue"]
        diagnostic_rows.append({
            "Model": name,
            "Mean residual (bias)": residual.mean(),
            "Residual SD": residual.std(),
            "Ljung-Box p (lag 24)": ljung_p,
            "White-noise at 5%?": "Yes" if ljung_p >= 0.05 else "No",
            "Jarque-Bera p": stats.jarque_bera(residual).pvalue,
        })
    diagnostics = pd.DataFrame(diagnostic_rows).merge(metrics[["Model", "Rank"]], on="Model")
    diagnostics = diagnostics.sort_values("Rank")

    return ForecastBundle(
        region=region,
        frame=model_frame,
        train=train,
        validation=validation,
        train_full=train_full,
        test=test,
        forecasts=forecast_frame,
        metrics=metrics,
        diagnostics=diagnostics,
        fit_times=fit_times,
    )


def classify_model(model: str) -> str:
    if model.startswith("Baseline"):
        return "Baseline"
    if model.startswith("SARIMAX"):
        return "SARIMAX"
    if model.startswith("ARIMA") or model.startswith("SARIMA"):
        return "ARIMA/SARIMA"
    if model.startswith("Holt-Winters"):
        return "Holt-Winters"
    if model.startswith("ML:"):
        return "Machine learning"
    return "Deep learning"
