"""Create the compact comparison figure used in the Overleaf report."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "overleaf_report" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

metrics = pd.read_csv(ROOT / "results" / "forecast_metrics.csv")
models = [
    "ARIMA stationary: (2,0,1) on Δ24",
    "ARIMA non-stationary: (2,1,1)",
    "SARIMA stationary: (1,0,1)(1,0,1,24) on Δ24",
    "SARIMA non-stationary: (1,0,1)(1,1,1,24)",
]
labels = ["ARIMA\nstationary", "ARIMA\nraw", "SARIMA\nstationary", "SARIMA\nraw"]
regions = ["IN-NO", "IN-WE", "IN-SO"]
colors = ["#1d4ed8", "#d97706", "#15803d"]

table = (
    metrics[metrics["Model"].isin(models)]
    .pivot(index="Model", columns="Region", values="RMSE")
    .reindex(models)
)

fig, ax = plt.subplots(figsize=(7.2, 2.8))
x = range(len(models))
width = 0.23
for offset, region, color in zip([-width, 0, width], regions, colors):
    bars = ax.bar([value + offset for value in x], table[region], width,
                  label=region, color=color, edgecolor="white", linewidth=0.5)
    for bar, value in zip(bars, table[region]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2.2, f"{value:.1f}",
                ha="center", va="bottom", fontsize=7, color="#1f2937")

ax.set_xticks(list(x), labels)
ax.set_ylabel("Test RMSE (gCO2eq/kWh)")
ax.set_ylim(0, 122)
ax.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
fig.tight_layout(pad=0.7)
fig.savefig(OUTPUT / "model_rmse_comparison.pdf", bbox_inches="tight")
plt.close(fig)

print(OUTPUT / "model_rmse_comparison.pdf")
