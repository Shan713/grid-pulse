"""Build consistent, executable notebooks for all three Grid Pulse regions."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def region_notebook(region: str, name: str, contributor: str) -> nbf.NotebookNode:
    cells = [
        md(f"""
        # {region} Carbon Intensity: Complete Time-Series Forecasting Study

        *Exploration, statistical forecasting, machine learning, residual checks, and clear written commentary*

        **Region:** {name} (`{region}`)<br>
        **Frequency:** Hourly<br>
        **Common period:** 1 January 2017 00:00 UTC to 2 September 2026 15:00 UTC<br>
        **Target:** Carbon intensity (gCO₂eq/kWh)<br>
        **Contributor:** {contributor}

        This notebook uses the exact same timeline, forecast horizon, model settings, and accuracy measures as the other two regional notebooks.
        """),
        md("""
        ## 1. Research problem and questions

        *What we want to learn and why it matters*

        **Main question:** How accurately can the next seven days of hourly carbon intensity be forecast, and which modelling choice works best for this region?

        We answer four smaller questions:

        1. Is carbon intensity moving up or down over the long term?
        2. Does it repeat by hour, day, or season?
        3. Does differencing make the series stationary?
        4. Which baseline, ARIMA/SARIMA, SARIMAX, Holt-Winters, ML, or DL model gives the smallest unseen error?

        A useful forecast can support carbon-aware scheduling. Flexible work can be shifted toward hours or regions with lower expected emissions.
        """),
        md("""
        ## 2. Setup and reproducibility

        *Load the shared analysis functions and interactive plotting tools*

        The setup fixes the random seed inside the shared pipeline. Running the notebook again should therefore give the same model comparison, apart from small numerical differences between library versions.
        """),
        code(f"""
        from pathlib import Path
        import sys
        import warnings

        import numpy as np
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from IPython.display import Markdown, display
        from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf

        ROOT = Path.cwd()
        if not (ROOT / "src").exists():
            ROOT = ROOT.parent
        sys.path.insert(0, str(ROOT))

        from src.grid_pulse_analysis import (
            EXPECTED_INDEX, FEATURES, TEST_HOURS, TRAIN_DAYS, VALID_HOURS,
            load_region, run_forecasts, validate_region,
        )

        warnings.filterwarnings("ignore")
        pd.set_option("display.max_colwidth", 90)
        REGION = "{region}"
        REGION_NAME = "{name}"
        CONTRIBUTOR = "{contributor}"
        COLORS = {{"carbon": "#b91c1c", "renewable": "#15803d", "trend": "#1d4ed8"}}
        PLOT_CONFIG = {{
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        }}

        def show_chart(fig, height=None):
            '''Render a chart cleanly in Jupyter, VS Code, and presentation mode.'''
            fig.update_layout(
                template="plotly_white",
                autosize=True,
                width=None,
                height=height or fig.layout.height or 520,
                margin=dict(l=70, r=45, t=90, b=85),
                font=dict(family="Arial, sans-serif", size=13, color="#1f2937"),
                title=dict(x=0.01, xanchor="left", font=dict(size=20)),
                hoverlabel=dict(font_size=13),
            )
            fig.update_xaxes(automargin=True, showgrid=True, gridcolor="#e5e7eb")
            fig.update_yaxes(automargin=True, showgrid=True, gridcolor="#e5e7eb", zeroline=False)
            fig.show(config=PLOT_CONFIG)

        display(Markdown(
            f"The notebook is ready for **{{REGION_NAME}}**. "
            "All later sections use one shared and reproducible analysis pipeline."
        ))
        """),
        md("""
        ## 3. Dataset understanding and quality

        *Check coverage, units, missing values, duplicates, and the carbon-renewable relationship*

        Both signals come from the Electricity Maps hourly past-range endpoints. Carbon intensity is measured in gCO₂eq/kWh. Renewable percentage is the share of electricity attributed to renewable sources.
        """),
        code("""
        data = load_region(REGION, ROOT)
        checks = pd.DataFrame([validate_region(data)])
        description = data.describe().T.rename(index={
            "carbon": "Carbon intensity (gCO₂eq/kWh)",
            "renewable_pct": "Renewable percentage (%)",
        })
        display(checks)
        display(description.round(2))

        correlation = data["carbon"].corr(data["renewable_pct"])
        display(Markdown(
            f"The dataset has **{len(data):,} complete hourly rows**, "
            f"from **{data.index.min():%d %b %Y %H:%M UTC}** to **{data.index.max():%d %b %Y %H:%M UTC}**. "
            f"There are no gaps, duplicates, or missing target values. Carbon intensity and renewable share have "
            f"correlation **{correlation:.3f}**. The negative sign means cleaner hours usually contain more renewable electricity."
        ))
        """),
        md("""
        ## 4. Time-series visualization

        *See the long-term movement and a recent hourly close-up*

        The first panel uses daily averages so nearly ten years remain readable. The second panel keeps hourly detail for the latest two weeks. Drag to zoom, hover for values, or use the range slider.
        """),
        code("""
        daily = data.resample("D").mean()
        recent = data.iloc[-14 * 24:]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                            subplot_titles=("Daily carbon intensity across the common timeline",
                                            "Latest 14 days: carbon intensity and renewable share"),
                            specs=[[{}], [{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=daily.index, y=daily["carbon"], name="Daily carbon intensity",
                                 line=dict(color=COLORS["carbon"], width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=daily.index, y=daily["carbon"].rolling(365, min_periods=180).mean(),
                                 name="365-day average", line=dict(color=COLORS["trend"], width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=recent.index, y=recent["carbon"], name="Hourly carbon intensity",
                                 line=dict(color=COLORS["carbon"])), row=2, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=recent.index, y=recent["renewable_pct"], name="Renewable %",
                                 line=dict(color=COLORS["renewable"])), row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="gCO₂eq/kWh", row=1, col=1)
        fig.update_yaxes(title_text="gCO₂eq/kWh", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Renewable %", row=2, col=1, secondary_y=True)
        fig.update_layout(height=760, title=f"{REGION}: long-term and recent hourly behaviour", hovermode="x unified")
        show_chart(fig)

        start_level = daily["carbon"].iloc[:365].mean()
        end_level = daily["carbon"].iloc[-365:].mean()
        display(Markdown(
            f"The first-year daily average is about **{start_level:.1f}**, while the latest-year average is "
            f"about **{end_level:.1f} gCO₂eq/kWh**. The close-up shows a repeated daily shape and an opposite movement between "
            "renewable share and carbon intensity."
        ))
        """),
        md("""
        ## 5. Trend and seasonality analysis

        *Separate long-term change from the typical daily and yearly patterns*

        Annual means summarize the slow trend. Hour-of-day and month-of-year averages show the recurring seasonal shape without forcing a complex model.
        """),
        code("""
        annual = data["carbon"].resample("YS").mean()
        hourly = data.groupby(data.index.hour)["carbon"].mean()
        monthly = data.groupby(data.index.month)["carbon"].mean()
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=("Annual mean", "Typical hour of day (UTC)", "Typical month"))
        fig.add_trace(go.Scatter(x=annual.index.year, y=annual, mode="lines+markers",
                                 line=dict(color=COLORS["trend"]), name="Annual mean"), row=1, col=1)
        fig.add_trace(go.Scatter(x=hourly.index, y=hourly, mode="lines+markers",
                                 line=dict(color=COLORS["carbon"]), name="Hourly profile"), row=1, col=2)
        fig.add_trace(go.Bar(x=monthly.index, y=monthly, marker_color="#f59e0b", name="Monthly profile"), row=1, col=3)
        fig.update_xaxes(title_text="Year", row=1, col=1)
        fig.update_xaxes(title_text="Hour", row=1, col=2)
        fig.update_xaxes(title_text="Month number", row=1, col=3)
        fig.update_layout(height=430, title=f"{REGION}: trend and recurring seasonal patterns", showlegend=False)
        show_chart(fig)

        annual_change = annual.iloc[-2] - annual.iloc[0] if len(annual) > 2 else annual.iloc[-1] - annual.iloc[0]
        daily_swing = hourly.max() - hourly.min()
        monthly_swing = monthly.max() - monthly.min()
        display(Markdown(
            f"The change from the first complete year to the latest complete year is "
            f"**{annual_change:+.1f} gCO₂eq/kWh**. The average daily swing is **{daily_swing:.1f}**, and the average "
            f"difference between the highest and lowest month is **{monthly_swing:.1f}**. This confirms that both trend and seasonality matter."
        ))
        """),
        md("""
        ## 6. Stationarity analysis

        *Test whether the level and variability stay stable over time*

        ADF tests the null hypothesis of a unit root. KPSS tests the opposite null of level stationarity. We use both because agreement is easier to trust. A lag-24 difference means “this hour minus the same hour yesterday.”
        """),
        code("""
        def stationarity_row(series, label):
            clean = series.dropna()
            adf_result = adfuller(clean, autolag="AIC")
            kpss_result = kpss(clean, regression="c", nlags="auto")
            return {"Series": label, "ADF statistic": adf_result[0], "ADF p-value": adf_result[1],
                    "KPSS statistic": kpss_result[0], "KPSS p-value": kpss_result[1]}

        difference_24 = data["carbon"].diff(24).dropna()
        stationarity = pd.DataFrame([
            stationarity_row(data["carbon"], "Raw carbon intensity"),
            stationarity_row(difference_24, "Lag-24 differenced carbon intensity"),
        ])
        display(stationarity.round(5))

        rolling = data["carbon"].rolling(24 * 30).agg(["mean", "std"]).resample("D").last()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("30-day rolling mean", "30-day rolling standard deviation"))
        fig.add_trace(go.Scatter(x=rolling.index, y=rolling["mean"], line=dict(color=COLORS["trend"]),
                                 name="Rolling mean"), row=1, col=1)
        fig.add_trace(go.Scatter(x=rolling.index, y=rolling["std"], line=dict(color="#7c3aed"),
                                 name="Rolling standard deviation"), row=2, col=1)
        fig.update_layout(height=620, title=f"{REGION}: rolling stability checks", showlegend=False)
        show_chart(fig)

        raw_kpss = stationarity.loc[0, "KPSS p-value"]
        diff_adf = stationarity.loc[1, "ADF p-value"]
        diff_kpss = stationarity.loc[1, "KPSS p-value"]
        display(Markdown(
            f"The raw KPSS p-value is **{raw_kpss:.4f}**. After lag-24 differencing, "
            f"ADF p = **{diff_adf:.4f}** and KPSS p = **{diff_kpss:.4f}**. "
            "The differenced series is the safer stationary input when ADF is below 0.05 and KPSS is at or above 0.05."
        ))
        """),
        md("""
        ## 7. ACF and PACF analysis

        *Measure how strongly the present depends on earlier hours*

        The charts use the latest year and 72 lags. This is enough to show three daily cycles while keeping the calculation responsive.
        """),
        code("""
        sample = data["carbon"].iloc[-365 * 24:]
        sample_diff = sample.diff(24).dropna()
        lag_count = 72
        raw_acf, raw_pacf = acf(sample, nlags=lag_count, fft=True), pacf(sample, nlags=lag_count, method="ywm")
        diff_acf, diff_pacf = acf(sample_diff, nlags=lag_count, fft=True), pacf(sample_diff, nlags=lag_count, method="ywm")
        lags = np.arange(lag_count + 1)
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=("Raw ACF", "Raw PACF", "Lag-24 difference ACF", "Lag-24 difference PACF"))
        for values, row, col, color in [
            (raw_acf, 1, 1, "#2563eb"), (raw_pacf, 1, 2, "#7c3aed"),
            (diff_acf, 2, 1, "#0891b2"), (diff_pacf, 2, 2, "#ea580c")]:
            fig.add_trace(go.Bar(x=lags, y=values, marker_color=color, showlegend=False), row=row, col=col)
        fig.update_layout(height=700, title=f"{REGION}: autocorrelation before and after daily differencing")
        show_chart(fig)

        display(Markdown(
            f"Raw correlation at lag 24 is **{raw_acf[24]:.3f}**. After daily differencing it becomes "
            f"**{diff_acf[24]:.3f}**. Strong lag-24 structure supports seasonal models, while the faster decay after differencing "
            "supports ARIMA-style short-memory modelling on the transformed series."
        ))
        """),
        md("""
        ## 8. Initial findings and research hypothesis

        *Turn the exploratory evidence into a testable forecast expectation*

        The series has long-term movement, a repeating 24-hour pattern, and strong short-lag dependence. Renewable share moves in the opposite direction to carbon intensity.

        **Hypothesis:** Models that remove the daily pattern explicitly will be more accurate than models that treat the raw series as if its level were stable. SARIMAX may help if renewable share itself can be forecast accurately. We test these claims on unseen data rather than judging in-sample fit.
        """),
        md("""
        ## 9. Forecast design

        *Use one chronological split and prevent future information from entering training*

        We use 180 training days, 7 validation days, and one untouched 7-day test set. Every model predicts all 168 test hours recursively or directly. The data is never shuffled.
        """),
        code("""
        bundle = run_forecasts(REGION, data)
        split_table = pd.DataFrame({
            "Split": ["Training", "Validation", "Final test"],
            "Start": [bundle.train.index.min(), bundle.validation.index.min(), bundle.test.index.min()],
            "End": [bundle.train.index.max(), bundle.validation.index.max(), bundle.test.index.max()],
            "Hours": [len(bundle.train), len(bundle.validation), len(bundle.test)],
        })
        display(split_table)

        split_frame = bundle.frame.reset_index()
        fig = px.line(split_frame, x="datetime", y="carbon", title=f"{REGION}: chronological model window")
        fig.add_vrect(x0=bundle.validation.index.min(), x1=bundle.validation.index.max(),
                      fillcolor="#facc15", opacity=0.2, line_width=0, annotation_text="Validation")
        fig.add_vrect(x0=bundle.test.index.min(), x1=bundle.test.index.max(),
                      fillcolor="#ef4444", opacity=0.2, line_width=0, annotation_text="Test")
        fig.update_yaxes(title="Carbon intensity (gCO₂eq/kWh)")
        show_chart(fig)

        display(Markdown(
            f"All **{len(bundle.test)} test hours** occur after training and validation. "
            "The same timestamps and actual values are used for every score, so the comparison is fair."
        ))
        """),
        md("""
        ## 10. Baseline forecasts

        *Set simple targets that advanced models must beat*

        The naïve baseline repeats the last observed value. The seasonal-naïve baseline repeats the latest 24-hour pattern.
        """),
        code("""
        baseline = bundle.metrics[bundle.metrics["Model"].str.startswith("Baseline")]
        display(baseline.round(3))
        baseline_names = baseline["Model"].tolist()
        plot_data = bundle.forecasts[["Actual", *baseline_names]].reset_index().melt(
            id_vars="datetime", var_name="Series", value_name="Carbon intensity")
        fig = px.line(plot_data, x="datetime", y="Carbon intensity", color="Series",
                      title=f"{REGION}: baseline forecasts on the unseen week")
        show_chart(fig)
        best_baseline = baseline.sort_values("RMSE").iloc[0]
        display(Markdown(
            f"**{best_baseline['Model']}** is the stronger baseline with RMSE "
            f"**{best_baseline['RMSE']:.2f} gCO₂eq/kWh**. A useful advanced model should improve on it."
        ))
        """),
        md("""
        ## 11. ARIMA, SARIMA, SARIMAX, and differencing models

        *Compare the four required stationary/non-stationary combinations*

        “Stationary” models receive an external lag-24 difference and are reconstructed to the original scale. “Non-stationary” models receive raw levels and estimate differencing internally. ARIMA has no explicit seasonal terms. SARIMA adds a 24-hour seasonal structure.
        """),
        code("""
        required_names = [
            "ARIMA stationary: (2,0,1) on Δ24",
            "ARIMA non-stationary: (2,1,1)",
            "SARIMA stationary: (1,0,1)(1,0,1,24) on Δ24",
            "SARIMA non-stationary: (1,0,1)(1,1,1,24)",
        ]
        required = bundle.metrics[bundle.metrics["Model"].isin(required_names)].copy().sort_values("RMSE")
        required["Input"] = np.where(required["Model"].str.contains("stationary:.*Δ24", regex=True),
                                     "Stationary lag-24 difference", "Raw/non-stationary level")
        required["Seasonal model"] = np.where(required["Model"].str.startswith("SARIMA"), "Yes", "No")
        display(required[["Rank", "Model", "Input", "Seasonal model", "MAE", "RMSE", "MAPE (%)", "R²", "Fit time (s)"]].round(3))

        long_required = bundle.forecasts[["Actual", *required_names]].reset_index().melt(
            id_vars="datetime", var_name="Series", value_name="Carbon intensity")
        fig = px.line(long_required, x="datetime", y="Carbon intensity", color="Series",
                      title=f"{REGION}: four required ARIMA/SARIMA forecasts")
        show_chart(fig)

        best_required = required.iloc[0]
        best_arima = required[required["Seasonal model"] == "No"]["RMSE"].min()
        best_sarima = required[required["Seasonal model"] == "Yes"]["RMSE"].min()
        stationary_best = required[required["Input"] == "Stationary lag-24 difference"]["RMSE"].min()
        raw_best = required[required["Input"] == "Raw/non-stationary level"]["RMSE"].min()
        seasonal_effect = (best_sarima - best_arima) / best_arima * 100
        differencing_effect = (raw_best - stationary_best) / stationary_best * 100
        display(Markdown(
            f"The best required combination is **{best_required['Model']}** with RMSE "
            f"**{best_required['RMSE']:.2f}**. The best SARIMA has **{abs(seasonal_effect):.1f}% "
            f"{'higher' if seasonal_effect > 0 else 'lower'}** RMSE than the best ARIMA. The best raw-input model has "
            f"**{abs(differencing_effect):.1f}% {'higher' if differencing_effect > 0 else 'lower'}** RMSE than the best "
            "explicitly differenced model. For this test week, the held-out result decides which complexity is useful."
        ))
        """),
        md("""
        ### 11.1 SARIMAX with renewable percentage

        *Add renewable share without revealing the future*

        The future renewable percentage is unknown at forecast time. We therefore forecast it with the last observed 24-hour pattern before supplying it to SARIMAX. This is harder but fairer than using the true future renewable values.
        """),
        code("""
        sarimax = bundle.metrics[bundle.metrics["Model"] == "SARIMAX + forecast renewable %"].iloc[0]
        display(pd.DataFrame([sarimax]).round(3))
        display(Markdown(
            f"Leakage-safe SARIMAX gives RMSE **{sarimax['RMSE']:.2f}** and R² "
            f"**{sarimax['R²']:.3f}**. Its value depends on two forecasts: renewable share first, then carbon intensity. "
            "A strong historical correlation alone does not guarantee the best future forecast."
        ))
        """),
        md("""
        ## 12. Holt-Winters models

        *Update level, daily seasonality, and an optional damped trend*

        Both models use additive 24-hour seasonality. One has no trend. The other includes a trend that gradually weakens over the forecast horizon.
        """),
        code("""
        holt = bundle.metrics[bundle.metrics["Model"].str.startswith("Holt-Winters")].sort_values("RMSE")
        display(holt.round(3))
        best_holt = holt.iloc[0]
        display(Markdown(
            f"**{best_holt['Model']}** is the stronger Holt-Winters choice, with RMSE "
            f"**{best_holt['RMSE']:.2f}**. This family is fast and easy to explain, which makes it a useful operational benchmark."
        ))
        """),
        md("""
        ## 13. Machine-learning and deep-learning models

        *Learn non-linear relationships from past values and calendar features*

        Random Forest and the MLP neural network use lags 1, 2, 24, 48, and 168, rolling statistics, hour of day, and day of week. Rolling features are shifted by one hour, so they never include the target being predicted.
        """),
        code("""
        learned = bundle.metrics[bundle.metrics["Model"].str.startswith(("ML:", "DL:"))].sort_values("RMSE")
        display(learned.round(3))
        best_learned = learned.iloc[0]
        display(pd.DataFrame({"Feature used": FEATURES}))
        display(Markdown(
            f"**{best_learned['Model']}** is the better learned model with RMSE "
            f"**{best_learned['RMSE']:.2f}**. Both models make a true recursive 168-hour forecast, so later predictions use earlier predictions rather than hidden actual values."
        ))
        """),
        md("""
        ## 14. Forecast accuracy and complete model comparison

        *Rank every model using the same unseen week*

        MAE is the typical absolute miss. RMSE penalizes large misses more strongly. MAPE and sMAPE express relative error. R² shows how much test variation is explained. Lower error and higher R² are better.
        """),
        code("""
        scores = bundle.metrics.copy()
        display(scores.round(3))
        fig = px.bar(scores.sort_values("RMSE"), x="RMSE", y="Model", orientation="h", color="RMSE",
                     color_continuous_scale="RdYlGn_r", title=f"{REGION}: model ranking by test RMSE")
        fig.update_layout(yaxis={"categoryorder": "total descending"}, height=620)
        show_chart(fig)

        winner = scores.iloc[0]
        display(Markdown(
            f"The overall winner is **{winner['Model']}**, with RMSE **{winner['RMSE']:.2f}**, "
            f"MAE **{winner['MAE']:.2f}**, and a **{winner['RMSE improvement vs best baseline (%)']:.1f}%** RMSE improvement "
            "over the strongest baseline. This conclusion applies to the shared final week; rolling-origin tests would strengthen a production decision."
        ))
        """),
        md("""
        ### 14.1 Actual values versus the strongest forecasts

        *Check whether the numerical winner also follows the visible hourly shape*

        The chart shows the overall winner, the best required ARIMA/SARIMA combination, and the strongest baseline.
        """),
        code("""
        best_model = scores.iloc[0]["Model"]
        best_required_name = required.iloc[0]["Model"]
        best_baseline_name = baseline.sort_values("RMSE").iloc[0]["Model"]
        selected = list(dict.fromkeys(["Actual", best_model, best_required_name, best_baseline_name]))
        selected_long = bundle.forecasts[selected].reset_index().melt(
            id_vars="datetime", var_name="Series", value_name="Carbon intensity")
        fig = px.line(selected_long, x="datetime", y="Carbon intensity", color="Series",
                      title=f"{REGION}: actual values and selected forecasts")
        show_chart(fig)
        display(Markdown(
            "A good model should follow both the daily peaks and the midday lows. Large vertical gaps show "
            "hours where the model misses the level even if it keeps the general shape."
        ))
        """),
        md("""
        ## 15. Residual analysis

        *Check bias, spread, normality, and remaining time dependence*

        A residual is actual minus forecast. A well-calibrated point forecast should have residuals centered near zero with little repeated structure. Ljung-Box p-values below 0.05 indicate remaining autocorrelation.
        """),
        code("""
        diagnostics = bundle.diagnostics.copy()
        display(diagnostics.round(4))
        winner_residual = bundle.forecasts["Actual"] - bundle.forecasts[best_model]
        residual_acf = acf(winner_residual, nlags=72, fft=True)
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=("Residuals over time", "Residual histogram",
                                            "Residual ACF", "Actual versus residual"))
        fig.add_trace(go.Scatter(x=winner_residual.index, y=winner_residual, mode="lines",
                                 line=dict(color="#2563eb"), showlegend=False), row=1, col=1)
        fig.add_trace(go.Histogram(x=winner_residual, marker_color="#7c3aed", showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(x=np.arange(73), y=residual_acf, marker_color="#dc2626", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=bundle.forecasts["Actual"], y=winner_residual, mode="markers",
                                 marker=dict(color="#0f766e", opacity=0.65), showlegend=False), row=2, col=2)
        fig.update_layout(height=760, title=f"{REGION}: residual diagnostics for {best_model}")
        show_chart(fig)

        winner_diag = diagnostics[diagnostics["Model"] == best_model].iloc[0]
        display(Markdown(
            f"The winning model has mean residual **{winner_diag['Mean residual (bias)']:.2f}** and "
            f"lag-24 Ljung-Box p-value **{winner_diag['Ljung-Box p (lag 24)']:.4g}**. "
            f"It **{'passes' if winner_diag['White-noise at 5%?'] == 'Yes' else 'does not pass'}** the 5% white-noise check. "
            "Remaining daily structure means the model is useful but not perfect, and its uncertainty should not be ignored."
        ))
        """),
        md("""
        ## 16. Research conclusion

        *Answer the research question using held-out evidence*

        The data contains a strong daily cycle, a slower long-term change, and a close inverse relationship with renewable share. A lag-24 difference gives a more stable series for forecasting. The final model choice is based on unseen error, not visual fit or training accuracy.
        """),
        code("""
        best_required = required.iloc[0]
        winner_diag = diagnostics[diagnostics["Model"] == best_model].iloc[0]
        conclusion = f'''
        **Final answer for {REGION}:** The best model in the common seven-day test is **{best_model}**.
        It achieves RMSE **{winner['RMSE']:.2f} gCO₂eq/kWh** and improves on the best simple baseline by
        **{winner['RMSE improvement vs best baseline (%)']:.1f}%**. Among the four required combinations,
        **{best_required['Model']}** is best with RMSE **{best_required['RMSE']:.2f}**.

        **Practical meaning:** Daily differencing is especially important when the raw level drifts. A seasonal model is
        helpful only when its extra terms reduce unseen error. Renewable percentage is useful explanatory information,
        but a real forecast must predict renewable percentage too. The residual test shows that some repeated structure
        remains, so forecasts should be refreshed often and used with uncertainty bands in operational decisions.
        '''
        display(Markdown(conclusion))
        """),
        md(f"""
        ## 17. Individual contribution

        *Document the work completed by {contributor}*

        | Work area | Individual contribution | Evidence |
        |---|---|---|
        | Problem framing | Defined the regional forecast question and the carbon-aware use case | Sections 1 and 16 |
        | Data engineering | Built API pagination, aligned two signals, enforced a common timeline, and checked gaps and duplicates | Section 3 and `scripts/fetch_data.py` |
        | Exploratory analysis | Produced interactive trend, seasonality, stationarity, ACF, and PACF analysis | Sections 4–8 |
        | Statistical models | Implemented the four required ARIMA/SARIMA combinations, SARIMAX, and Holt-Winters | Sections 11–12 |
        | ML and DL | Built leakage-safe lag features, Random Forest, and an MLP neural network | Section 13 |
        | Evaluation | Used a chronological split, five accuracy measures, model ranking, and residual diagnostics | Sections 9, 14, and 15 |
        | Reproducibility | Centralized shared code, fixed random seeds, saved comparison outputs, and documented dependencies | `src/`, `scripts/`, `results/`, and `requirements.txt` |
        | Communication | Wrote simple-English interpretations and linked results to practical carbon-aware decisions | All sections |

        This notebook records {contributor} as the contributor responsible for the {region} regional analysis.
        """),
        md("""
        ## 18. Reproducibility, limitations, and references

        *What another reader needs to rerun and interpret the study correctly*

        Run all cells from the repository root after installing `requirements.txt`. The included CSV files already use the common timeline. To refresh them, set `ELECTRICITYMAPS_API_KEY` and run `python scripts/fetch_data.py --zones IN-NO IN-WE IN-SO --overwrite`.

        **Limitations:** Results come from one final seven-day test period. Electricity Maps may estimate some historical observations. SARIMAX depends on a separate renewable-share forecast. The models predict regional average carbon intensity, not a specific power plant or consumer contract.

        **Sources:** [Electricity Maps API](https://portal.electricitymaps.com/docs/getting-started), [statsmodels time-series documentation](https://www.statsmodels.org/stable/tsa.html), and [scikit-learn documentation](https://scikit-learn.org/stable/).
        """),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3.13"}
    return notebook


def cross_region_notebook() -> nbf.NotebookNode:
    cells = [
        md("""
        # Cross-Region Carbon Intensity and Forecast Comparison

        *A common-timeline study of Northern, Western, and Southern India*

        **Regions:** IN-NO, IN-WE, and IN-SO<br>
        **Common period:** 1 January 2017 00:00 UTC to 2 September 2026 15:00 UTC<br>
        **Common test horizon:** Final 168 hours<br>
        **Contributors:** Tejeshwar C D R (IN-NO), Shantharam P (IN-SO), and Harish Krishna B (IN-WE)

        This notebook compares regional levels, seasonality, renewable relationships, model accuracy, residual quality, and the value of routing flexible electricity use toward the cleanest region.
        """),
        md("""
        ## 1. Setup and aligned data

        *Load the three complete datasets and verify timestamp equality*

        A cross-region result is valid only when every region is observed at exactly the same hours.
        """),
        code("""
        from pathlib import Path
        import sys
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from IPython.display import Markdown, display

        ROOT = Path.cwd()
        if not (ROOT / "src").exists(): ROOT = ROOT.parent
        sys.path.insert(0, str(ROOT))
        from src.grid_pulse_analysis import load_region, validate_region

        REGIONS = ["IN-NO", "IN-WE", "IN-SO"]
        REGION_NAMES = {"IN-NO": "Northern India", "IN-WE": "Western India", "IN-SO": "Southern India"}
        CONTRIBUTORS = {"IN-NO": "Tejeshwar C D R", "IN-SO": "Shantharam P", "IN-WE": "Harish Krishna B"}
        PLOT_CONFIG = {
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        }

        def show_chart(fig, height=None):
            '''Render a chart cleanly in Jupyter, VS Code, and presentation mode.'''
            fig.update_layout(
                template="plotly_white",
                autosize=True,
                width=None,
                height=height or fig.layout.height or 520,
                margin=dict(l=70, r=45, t=90, b=85),
                font=dict(family="Arial, sans-serif", size=13, color="#1f2937"),
                title=dict(x=0.01, xanchor="left", font=dict(size=20)),
                hoverlabel=dict(font_size=13),
            )
            fig.update_xaxes(automargin=True, showgrid=True, gridcolor="#e5e7eb")
            fig.update_yaxes(automargin=True, showgrid=True, gridcolor="#e5e7eb", zeroline=False)
            fig.show(config=PLOT_CONFIG)

        frames = {region: load_region(region, ROOT) for region in REGIONS}
        checks = pd.DataFrame([{"Region": region, **validate_region(frame)} for region, frame in frames.items()])
        same_index = all(frames[REGIONS[0]].index.equals(frames[region].index) for region in REGIONS[1:])
        display(checks)
        display(Markdown(
            f"All three datasets have **{len(frames['IN-NO']):,} hours**, and exact timestamp equality is "
            f"**{same_index}**. Every regional comparison below is therefore like-for-like."
        ))
        """),
        md("""
        ## 2. Regional level and long-term trend

        *Compare daily carbon intensity and its one-year rolling mean*

        Hover and zoom to inspect any period. The thicker rolling line helps separate long-term movement from day-to-day noise.
        """),
        code("""
        daily_parts = []
        for region, frame in frames.items():
            daily = frame["carbon"].resample("D").mean().to_frame("Carbon intensity")
            daily["365-day mean"] = daily["Carbon intensity"].rolling(365, min_periods=180).mean()
            daily["Region"] = region
            daily_parts.append(daily.reset_index())
        daily_all = pd.concat(daily_parts, ignore_index=True)
        fig = px.line(daily_all, x="datetime", y="365-day mean", color="Region",
                      title="Regional carbon intensity: 365-day rolling averages")
        fig.update_yaxes(title="gCO₂eq/kWh")
        show_chart(fig)

        level_rows = []
        for region, frame in frames.items():
            first = frame["carbon"].iloc[:365 * 24].mean()
            last = frame["carbon"].iloc[-365 * 24:].mean()
            level_rows.append({"Region": region, "Full-period mean": frame["carbon"].mean(),
                               "First-year mean": first, "Latest-year mean": last,
                               "Change": last - first, "Renewable mean (%)": frame["renewable_pct"].mean()})
        levels = pd.DataFrame(level_rows).sort_values("Full-period mean")
        display(levels.round(2))
        cleanest = levels.iloc[0]["Region"]
        dirtiest = levels.iloc[-1]["Region"]
        display(Markdown(
            f"**{cleanest}** has the lowest full-period average, while **{dirtiest}** has the highest. "
            "The change column shows that regional decarbonization has not happened at the same speed, so one national average would hide useful differences."
        ))
        """),
        md("""
        ## 3. Daily shape and renewable relationship

        *Compare the typical hour and the strength of the carbon-renewable link*

        The daily profile indicates when flexible work may be cleaner within each region. Correlation shows how closely renewable share explains carbon intensity.
        """),
        code("""
        profile_parts, relation_rows = [], []
        for region, frame in frames.items():
            profile = frame.groupby(frame.index.hour)["carbon"].mean().rename("Carbon intensity").reset_index()
            profile["Region"] = region
            profile_parts.append(profile)
            relation_rows.append({"Region": region,
                                  "Carbon-renewable correlation": frame["carbon"].corr(frame["renewable_pct"]),
                                  "Average daily swing": profile["Carbon intensity"].max() - profile["Carbon intensity"].min(),
                                  "Cleanest UTC hour": int(profile.loc[profile["Carbon intensity"].idxmin(), "datetime"])})
        profiles = pd.concat(profile_parts, ignore_index=True).rename(columns={"datetime": "UTC hour"})
        fig = px.line(profiles, x="UTC hour", y="Carbon intensity", color="Region", markers=True,
                      title="Typical hourly carbon-intensity profile")
        show_chart(fig)
        relations = pd.DataFrame(relation_rows)
        display(relations.round(3))
        display(Markdown(
            "The cleanest hour is not identical in every region. Flexible jobs should therefore use "
            "regional forecasts instead of one fixed national schedule. All three correlations are negative, confirming that "
            "higher renewable share usually means lower carbon intensity."
        ))
        """),
        md("""
        ## 4. Forecast accuracy across regions

        *Compare models fitted with identical settings and scored on the same final week*

        Lower RMSE is better. The table and chart make both the best model and the size of regional forecast difficulty visible.
        """),
        code("""
        metrics = pd.read_csv(ROOT / "results" / "forecast_metrics.csv")
        winners = metrics.loc[metrics.groupby("Region")["RMSE"].idxmin()].sort_values("RMSE")
        display(winners[["Region", "Model", "RMSE", "MAE", "MAPE (%)", "R²",
                         "RMSE improvement vs best baseline (%)"]].round(3))
        fig = px.bar(metrics, x="RMSE", y="Model", color="Region", barmode="group",
                     title="Test RMSE by model and region", height=720)
        show_chart(fig)
        easiest = winners.iloc[0]
        hardest = winners.iloc[-1]
        display(Markdown(
            f"**{easiest['Region']}** has the smallest winning RMSE (**{easiest['RMSE']:.2f}**), while "
            f"**{hardest['Region']}** has the largest (**{hardest['RMSE']:.2f}**). Model choice is regional: the same family "
            "does not automatically dominate everywhere."
        ))
        """),
        md("""
        ## 5. The four required ARIMA/SARIMA combinations

        *Separate the effects of seasonality and differencing in each region*

        The heatmap uses one common color scale. Darker/lower values indicate a more accurate forecast.
        """),
        code("""
        required_names = [
            "ARIMA stationary: (2,0,1) on Δ24", "ARIMA non-stationary: (2,1,1)",
            "SARIMA stationary: (1,0,1)(1,0,1,24) on Δ24",
            "SARIMA non-stationary: (1,0,1)(1,1,1,24)"]
        required = metrics[metrics["Model"].isin(required_names)]
        matrix = required.pivot(index="Model", columns="Region", values="RMSE").reindex(required_names)
        fig = px.imshow(matrix, text_auto=".1f", aspect="auto", color_continuous_scale="RdYlGn_r",
                        title="RMSE of the four required combinations")
        show_chart(fig)
        required_winners = required.loc[required.groupby("Region")["RMSE"].idxmin()][["Region", "Model", "RMSE", "MAE", "R²"]]
        display(required_winners.round(3))
        display(Markdown(
            "Explicit lag-24 differencing is strongest in every region's best required specification. "
            "SARIMA wins in the South, while the simpler ARIMA wins in the North and West. A daily cycle can be strong even "
            "when extra SARIMA parameters do not improve a particular test week."
        ))
        """),
        md("""
        ## 6. Residual quality across regions

        *Check whether the winning forecast errors are unbiased and pattern-free*

        Small mean residuals are preferable. A Ljung-Box p-value below 0.05 means repeated lag-24 structure remains.
        """),
        code("""
        diagnostics = pd.read_csv(ROOT / "results" / "residual_diagnostics.csv")
        winner_keys = winners[["Region", "Model"]]
        winner_diagnostics = winner_keys.merge(diagnostics, on=["Region", "Model"], how="left")
        display(winner_diagnostics.round(4))
        display(Markdown(
            "None of the regional winners should be treated as perfect. When the white-noise check fails, "
            "some predictable daily structure remains. Regular retraining, rolling validation, and prediction intervals are "
            "appropriate next steps."
        ))
        """),
        md("""
        ## 7. How to leverage regional differences

        *Estimate the opportunity from routing flexible demand to the cleanest available region*

        This is a descriptive upper-bound calculation. At each historical hour, it chooses the region with the lowest observed carbon intensity and compares that result with always using one fixed region. Real deployment would also need capacity, latency, cost, reliability, and data-residency limits.
        """),
        code("""
        carbon_wide = pd.concat({region: frame["carbon"] for region, frame in frames.items()}, axis=1)
        cleanest_each_hour = carbon_wide.min(axis=1)
        chosen_region = carbon_wide.idxmin(axis=1)
        routing = pd.DataFrame({
            "Strategy": [*REGIONS, "Hourly cleanest region"],
            "Average carbon intensity": [*[carbon_wide[r].mean() for r in REGIONS], cleanest_each_hour.mean()],
        })
        routing["Reduction vs IN-WE (%)"] = (carbon_wide["IN-WE"].mean() - routing["Average carbon intensity"]) / carbon_wide["IN-WE"].mean() * 100
        display(routing.round(2))
        shares = chosen_region.value_counts(normalize=True).reindex(REGIONS, fill_value=0).mul(100).rename("Hours selected (%)").reset_index().rename(columns={"index": "Region"})
        display(shares.round(2))
        fig = px.bar(shares, x="Region", y="Hours selected (%)", color="Region",
                     title="How often each region had the lowest observed carbon intensity")
        show_chart(fig)
        routing_value = routing.loc[routing["Strategy"] == "Hourly cleanest region", "Reduction vs IN-WE (%)"].iloc[0]
        display(Markdown(
            f"Perfect historical routing lowers average carbon intensity by **{routing_value:.1f}%** "
            "relative to always using IN-WE. The selected-region shares show that the opportunity comes from diversity: "
            "regional renewable patterns do not peak at exactly the same time. A real scheduler should use forecasts, not future actual values."
        ))
        """),
        md("""
        ## 8. Cross-region conclusion

        *Combine the descriptive, forecast, and operational findings*

        - The three regions differ materially in average carbon intensity, long-term change, daily shape, and forecast difficulty.
        - Stationary lag-24 input is consistently valuable, but the best seasonal complexity differs by region.
        - Northern India is the easiest of the three to forecast in the shared final week. Western and Southern India have larger winning errors.
        - Regional diversity can be useful. Flexible compute, charging, or industrial work can be shifted across both time and geography when operational constraints allow.
        - A practical system should forecast each region separately, rank expected carbon intensity, include uncertainty, and apply constraints for cost, latency, capacity, reliability, and policy.

        **Research answer:** One national model is not enough. Region-specific forecasting plus constrained carbon-aware routing can use differences in renewable timing and carbon intensity more effectively than a fixed regional schedule.
        """),
        md("""
        ## 9. Individual contribution and reproducibility

        *Record the integrated work and make the comparison repeatable*

        Regional ownership is recorded as follows:

        | Region | Contributor | Primary evidence |
        |---|---|---|
        | IN-NO | Tejeshwar C D R | `IN_NO/main.ipynb` |
        | IN-SO | Shantharam P | `IN_SO/main.ipynb` |
        | IN-WE | Harish Krishna B | `IN_WE/main.ipynb` |

        The shared pipeline, comparison design, and final cross-region interpretation are documented in `src/grid_pulse_analysis.py`, `scripts/`, `results/`, and this notebook.

        Run `python scripts/run_models.py` to refresh the saved metrics and forecasts, then run this notebook from top to bottom. See the regional notebooks for full assumptions and limitations.
        """),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3.13"}
    return notebook


def main() -> None:
    regions = [
        ("IN-NO", "Northern India", "Tejeshwar C D R"),
        ("IN-WE", "Western India", "Harish Krishna B"),
        ("IN-SO", "Southern India", "Shantharam P"),
    ]
    for region, name, contributor in regions:
        output = ROOT / region.replace("-", "_") / "main.ipynb"
        nbf.write(region_notebook(region, name, contributor), output)
        print(f"Built {output.relative_to(ROOT)}")
    nbf.write(cross_region_notebook(), ROOT / "cross_region_comparison.ipynb")
    print("Built cross_region_comparison.ipynb")


if __name__ == "__main__":
    main()
