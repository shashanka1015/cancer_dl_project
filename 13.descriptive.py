import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUT_DIR  = "data/processed"
PLOT_DIR = "outputs/figures"
os.makedirs(PLOT_DIR, exist_ok=True)

# Load the smoothed complete dataset
df = pd.read_csv("data/processed/complete_data_smooth.csv")
print(f"Dataset shape: {df.shape}")
print(f"Years: {sorted(df['year'].unique())}")
print(f"Counties: {df['county_fips'].nunique()}")

# Define outcome and predictor groups for organised reporting
OUTCOMES = {
    "mortality_lung_per100k":        "Lung cancer mortality (raw)",
    "mortality_lung_smoothed":       "Lung cancer mortality (smoothed)",
    "mortality_breast_per100k":      "Breast cancer mortality (raw)",
    "mortality_breast_smoothed":     "Breast cancer mortality (smoothed)",
    "mortality_colorectal_per100k":  "Colorectal cancer mortality (raw)",
    "mortality_colorectal_smoothed": "Colorectal cancer mortality (smoothed)",
}

POLLUTION = {
    "pm25_satellite_lag10": "PM2.5 satellite 10yr lag (ug/m3)",
    "pm25_ground_lag10":    "PM2.5 ground 10yr lag (ug/m3)",
    "no2_satellite_lag10":  "NO2 satellite 10yr lag (ppb)",
    "ozone_lag10":          "Ozone 10yr lag (ppm)",
}

SOCIOECONOMIC = {
    "smoking_pct":          "Adult smoking prevalence",
    "obesity_pct":          "Adult obesity rate",
    "unemployment_pct":     "Unemployment rate",
    "child_poverty_pct":    "Children in poverty",
    "median_income":        "Median household income (USD)",
    "uninsured_adults_pct": "Uninsured adults rate",
    "rural_pct":            "Percent rural population",
    "pct_black":            "Percent Non-Hispanic Black",
    "pct_hispanic":         "Percent Hispanic",
}

OUTCOME    = "mortality_lung_smoothed"
PREDICTORS = list(POLLUTION.keys()) + list(SOCIOECONOMIC.keys())

# Step 1: Summary statistics table
print("\nSummary statistics")
all_vars = {**OUTCOMES, **POLLUTION, **SOCIOECONOMIC}
rows = []
for col, label in all_vars.items():
    if col not in df.columns:
        continue
    s        = df[col].describe()
    n_miss   = df[col].isna().sum()
    pct_miss = round(n_miss / len(df) * 100, 1)
    rows.append({
        "Variable":  label,
        "N":         int(s["count"]),
        "Missing %": pct_miss,
        "Mean":      round(s["mean"], 3),
        "SD":        round(s["std"],  3),
        "Min":       round(s["min"],  3),
        "Median":    round(s["50%"],  3),
        "Max":       round(s["max"],  3),
    })

df_summary = pd.DataFrame(rows)
print(df_summary.to_string(index=False))

out_summary = os.path.join(OUT_DIR, "descriptive_summary.csv")
df_summary.to_csv(out_summary, index=False)
print(f"\nSaved summary table: {out_summary}")

# Step 2: Temporal trends — national mean per year
print("\nTemporal trends (national mean by year)")
trend_cols = [
    "mortality_lung_smoothed",
    "pm25_satellite_lag10",
    "no2_satellite_lag10",
    "smoking_pct",
]
df_trend = df.groupby("year")[trend_cols].mean().round(3)
print(df_trend.to_string())

# Step 3: Pearson correlations for all three cancer outcomes
predictor_cols = [c for c in PREDICTORS if c in df.columns]

for cancer_col, cancer_label in [
    ("mortality_lung_smoothed",       "Lung cancer"),
    ("mortality_breast_smoothed",     "Breast cancer"),
    ("mortality_colorectal_smoothed", "Colorectal cancer"),
]:
    if cancer_col not in df.columns:
        continue
    print(f"\n{'='*65}")
    print(f"Pearson correlations with {cancer_label} mortality (smoothed)")
    print(f"{'='*65}")
    corr_series = (
        df[predictor_cols + [cancer_col]]
        .corr()[cancer_col]
        .drop(cancer_col)
        .sort_values(ascending=False)
    )
    for var, r in corr_series.items():
        bar  = "#" * int(abs(r) * 20)
        sign = "+" if r >= 0 else "-"
        print(f"  {var:30s}: {r:+.3f}  {sign}{bar}")

# Step 4: Distribution plots for all outcome variables
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Cancer Mortality Rate Distributions (per 100,000)", fontsize=14)

plot_pairs = [
    ("mortality_lung_per100k",        "Lung (raw)"),
    ("mortality_lung_smoothed",       "Lung (smoothed)"),
    ("mortality_breast_per100k",      "Breast (raw)"),
    ("mortality_breast_smoothed",     "Breast (smoothed)"),
    ("mortality_colorectal_per100k",  "Colorectal (raw)"),
    ("mortality_colorectal_smoothed", "Colorectal (smoothed)"),
]

for ax, (col, title) in zip(axes.flatten(), plot_pairs):
    if col in df.columns:
        data = df[col].dropna()
        ax.hist(data, bins=50, color="#1B7A8C", edgecolor="white", linewidth=0.3)
        ax.axvline(data.mean(), color="#D85A30", linewidth=1.5,
                   label=f"Mean: {data.mean():.1f}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Rate per 100,000")
        ax.set_ylabel("County-year count")
        ax.legend(fontsize=9)

plt.tight_layout()
dist_path = os.path.join(PLOT_DIR, "01_outcome_distributions.png")
plt.savefig(dist_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {dist_path}")

# Step 5: Temporal trends plot
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("National Mean Temporal Trends 2010-2020", fontsize=14)

trend_plot = [
    ("mortality_lung_smoothed", "Lung cancer mortality\n(smoothed, per 100k)", "#1B7A8C"),
    ("pm25_satellite_lag10",    "PM2.5 satellite\n10-year lag (ug/m3)",        "#D85A30"),
    ("no2_satellite_lag10",     "NO2 satellite\n10-year lag (ppb)",            "#534AB7"),
    ("smoking_pct",             "Adult smoking\nprevalence",                   "#15803D"),
]

for ax, (col, title, color) in zip(axes.flatten(), trend_plot):
    if col in df.columns:
        yearly = df.groupby("year")[col].mean()
        ax.plot(yearly.index, yearly.values, color=color,
                linewidth=2, marker="o", markersize=5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Year")
        ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
trend_path = os.path.join(PLOT_DIR, "02_temporal_trends.png")
plt.savefig(trend_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {trend_path}")

# Step 6: PM2.5 and smoking vs all three cancer outcomes
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("PM2.5 satellite lag and smoking vs cancer mortality (smoothed)", fontsize=13)

cancer_outcomes = [
    ("mortality_lung_smoothed",       "Lung mortality (per 100k)"),
    ("mortality_breast_smoothed",     "Breast mortality (per 100k)"),
    ("mortality_colorectal_smoothed", "Colorectal mortality (per 100k)"),
]

for col_idx, (y_col, y_label) in enumerate(cancer_outcomes):
    for row_idx, (x_col, x_label) in enumerate([
        ("pm25_satellite_lag10", "PM2.5 satellite lag (ug/m3)"),
        ("smoking_pct",          "Smoking prevalence"),
    ]):
        ax    = axes[row_idx, col_idx]
        valid = df[[x_col, y_col]].dropna()
        ax.scatter(valid[x_col], valid[y_col],
                   alpha=0.08, s=4, color="#1B7A8C")
        z   = np.polyfit(valid[x_col], valid[y_col], 1)
        xln = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)
        ax.plot(xln, np.poly1d(z)(xln), color="#D85A30", linewidth=1.5)
        r   = valid[[x_col, y_col]].corr().iloc[0, 1]
        ax.set_title(f"{y_label[:20]} — r={r:.3f}", fontsize=10)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)

plt.tight_layout()
scatter_path = os.path.join(PLOT_DIR, "03_pollution_cancer_scatter.png")
plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {scatter_path}")

# Step 7: Distribution plots for all pollution predictor variables
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("Pollution Predictor Distributions (10-year lagged means)", fontsize=13)

poll_plot = [
    ("pm25_satellite_lag10", "PM2.5 satellite lag (ug/m3)", "#1B7A8C"),
    ("pm25_ground_lag10",    "PM2.5 ground lag (ug/m3)",    "#2BA8BF"),
    ("no2_satellite_lag10",  "NO2 satellite lag (ppb)",     "#534AB7"),
    ("ozone_lag10",          "Ozone lag (ppm)",              "#15803D"),
]

for ax, (col, title, color) in zip(axes.flatten(), poll_plot):
    if col in df.columns:
        data = df[col].dropna()
        ax.hist(data, bins=50, color=color, edgecolor="white", linewidth=0.3)
        ax.axvline(data.mean(), color="#D85A30", linewidth=1.5,
                   label=f"Mean: {data.mean():.3f}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("County-year count")
        ax.legend(fontsize=9)

plt.tight_layout()
poll_dist_path = os.path.join(PLOT_DIR, "04_pollution_distributions.png")
plt.savefig(poll_dist_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {poll_dist_path}")

# Step 8: Distribution plots for all socioeconomic predictor variables
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
fig.suptitle("Socioeconomic Predictor Distributions", fontsize=13)

soc_plot = [
    ("smoking_pct",          "Adult smoking prevalence",     "#D85A30"),
    ("obesity_pct",          "Adult obesity rate",            "#D85A30"),
    ("unemployment_pct",     "Unemployment rate",             "#D85A30"),
    ("child_poverty_pct",    "Children in poverty",           "#534AB7"),
    ("median_income",        "Median household income (USD)", "#534AB7"),
    ("uninsured_adults_pct", "Uninsured adults rate",         "#534AB7"),
    ("rural_pct",            "Percent rural population",      "#15803D"),
    ("pct_black",            "Percent Non-Hispanic Black",    "#15803D"),
    ("pct_hispanic",         "Percent Hispanic",              "#15803D"),
]

for ax, (col, title, color) in zip(axes.flatten(), soc_plot):
    if col in df.columns:
        data = df[col].dropna()
        ax.hist(data, bins=50, color=color, edgecolor="white", linewidth=0.3)
        ax.axvline(data.mean(), color="#1B7A8C", linewidth=1.5,
                   label=f"Mean: {data.mean():.3f}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("County-year count")
        ax.legend(fontsize=9)

plt.tight_layout()
soc_dist_path = os.path.join(PLOT_DIR, "05_socioeconomic_distributions.png")
plt.savefig(soc_dist_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {soc_dist_path}")

# Step 9: Correlation heatmap for all predictors and lung cancer outcome
fig, ax = plt.subplots(figsize=(13, 10))

all_pred_cols = [c for c in PREDICTORS if c in df.columns]
corr_matrix   = df[all_pred_cols + [OUTCOME]].corr()

im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

labels = [SOCIOECONOMIC.get(c, POLLUTION.get(c, c)) for c in corr_matrix.columns]
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(labels, fontsize=9)

for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        val = corr_matrix.values[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7, color="black" if abs(val) < 0.6 else "white")

ax.set_title("Correlation matrix — all predictors and lung cancer outcome", fontsize=12)
plt.tight_layout()
heatmap_path = os.path.join(PLOT_DIR, "06_correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {heatmap_path}")

# Step 10: Boxplots of lung cancer mortality by year
fig, ax = plt.subplots(figsize=(13, 6))
years_list   = sorted(df["year"].unique())
data_by_year = [
    df[df["year"] == y]["mortality_lung_smoothed"].dropna().values
    for y in years_list
]
bp = ax.boxplot(data_by_year, labels=years_list, patch_artist=True,
                medianprops=dict(color="#D85A30", linewidth=2))
for patch in bp["boxes"]:
    patch.set_facecolor("#1B7A8C")
    patch.set_alpha(0.6)
ax.set_title("Lung cancer mortality by year (smoothed, per 100k)", fontsize=12)
ax.set_xlabel("Year")
ax.set_ylabel("Mortality rate per 100,000")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
box_path = os.path.join(PLOT_DIR, "07_lung_mortality_boxplot.png")
plt.savefig(box_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {box_path}")

# Step 10b: Temporal trends for all three cancer outcomes
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Cancer mortality trends 2010-2020 (smoothed national mean)", fontsize=13)

trend_cancers = [
    ("mortality_lung_smoothed",       "Lung cancer",       "#1B7A8C"),
    ("mortality_breast_smoothed",     "Breast cancer",     "#534AB7"),
    ("mortality_colorectal_smoothed", "Colorectal cancer", "#D85A30"),
]

for ax, (col, label, color) in zip(axes, trend_cancers):
    yearly = df.groupby("year")[col].mean()
    ax.plot(yearly.index, yearly.values, color=color,
            linewidth=2, marker="o", markersize=5)
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mortality rate per 100,000")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
trend3_path = os.path.join(PLOT_DIR, "10_all_cancer_trends.png")
plt.savefig(trend3_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {trend3_path}")

# Step 11: Cross-cancer correlations
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Cross-cancer correlations (smoothed rates)", fontsize=13)

cross_pairs = [
    ("mortality_breast_smoothed",     "Breast cancer mortality (per 100k)"),
    ("mortality_colorectal_smoothed", "Colorectal cancer mortality (per 100k)"),
]

for ax, (col, ylabel) in zip(axes, cross_pairs):
    valid = df[[col, "mortality_lung_smoothed"]].dropna()
    ax.scatter(valid["mortality_lung_smoothed"], valid[col],
               alpha=0.08, s=4, color="#534AB7")
    z   = np.polyfit(valid["mortality_lung_smoothed"], valid[col], 1)
    xln = np.linspace(valid["mortality_lung_smoothed"].min(),
                      valid["mortality_lung_smoothed"].max(), 100)
    ax.plot(xln, np.poly1d(z)(xln), color="#D85A30", linewidth=1.5)
    r = valid[["mortality_lung_smoothed", col]].corr().iloc[0, 1]
    ax.set_title(f"r = {r:.3f}", fontsize=11)
    ax.set_xlabel("Lung cancer mortality (per 100k)")
    ax.set_ylabel(ylabel)

plt.tight_layout()
cross_path = os.path.join(PLOT_DIR, "08_cross_cancer_correlation.png")
plt.savefig(cross_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {cross_path}")

print("\nAll descriptive figures saved.")
print(f"Figures saved to: {os.path.abspath(PLOT_DIR)}")

# Step 12: Choropleth maps of key variables using county shapefile

# Load shapefile and merge with 2015 data as a representative year
counties_map  = gpd.read_file("data/shapefiles/tl_2020_us_county.shp")
excluded      = ["02", "15", "60", "66", "69", "72", "78"]
counties_map  = counties_map[~counties_map["STATEFP"].isin(excluded)].copy()
counties_map  = counties_map.to_crs("EPSG:4326")
counties_map["county_fips"] = counties_map["GEOID"].astype(int)

df_2015 = df[df["year"] == 2015].copy()
counties_map = counties_map.merge(df_2015, on="county_fips", how="left")

# Variables to map
map_vars = [
    ("mortality_lung_per100k",  "Lung cancer mortality 2015\n(per 100,000)",  "YlOrRd"),
    ("pm25_satellite_lag10",    "PM2.5 satellite 10yr lag\n(ug/m3)",          "YlOrBr"),
    ("no2_satellite_lag10",     "NO2 satellite 10yr lag\n(ppb)",              "Purples"),
    ("smoking_pct",             "Adult smoking prevalence 2015",               "Oranges"),
    ("median_income",           "Median household income 2015\n(USD)",        "Greens"),
    ("rural_pct",               "Percent rural population",                   "Blues"),
]

fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle("Geographic Distribution of Key Variables (2015)", fontsize=15)

for ax, (col, title, cmap) in zip(axes.flatten(), map_vars):
    if col not in counties_map.columns:
        ax.axis("off")
        continue

    # Plot counties with missing data in light gray
    counties_map[counties_map[col].isna()].plot(
        ax=ax, color="#DDDDDD", linewidth=0
    )

    # Plot counties with valid data using choropleth
    counties_map[counties_map[col].notna()].plot(
        column=col, ax=ax, cmap=cmap,
        linewidth=0.05, edgecolor="white",
        legend=True,
        legend_kwds={
            "shrink": 0.6,
            "label":  title,
            "orientation": "horizontal",
            "pad": 0.01,
        }
    )
    ax.set_title(title, fontsize=11)
    ax.axis("off")

plt.tight_layout()
map_path = os.path.join(PLOT_DIR, "09_choropleth_maps.png")
plt.savefig(map_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {map_path}")

# Step 13: Lung cancer mortality choropleth alone at higher resolution
fig, ax = plt.subplots(figsize=(16, 9))

counties_map[counties_map["mortality_lung_per100k"].isna()].plot(
    ax=ax, color="#DDDDDD", linewidth=0
)
counties_map[counties_map["mortality_lung_per100k"].notna()].plot(
    column="mortality_lung_per100k", ax=ax,
    cmap="YlOrRd", linewidth=0.05, edgecolor="white",
    legend=True,
    legend_kwds={
        "shrink": 0.5,
        "label":  "Lung cancer mortality per 100,000",
        "orientation": "horizontal",
        "pad": 0.02,
    }
)
ax.set_title("County-level lung cancer mortality 2015 (per 100,000)", fontsize=14)
ax.axis("off")
plt.tight_layout()
lung_map_path = os.path.join(PLOT_DIR, "09b_lung_mortality_map.png")
plt.savefig(lung_map_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {lung_map_path}")

print("\nAll figures complete.")