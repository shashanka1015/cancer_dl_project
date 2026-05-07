import os
import numpy as np
import pandas as pd
import geopandas as gpd
import libpysal
import spreg
from esda.moran import Moran
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUT_DIR  = "data/processed"
PLOT_DIR = "outputs/figures"
os.makedirs(PLOT_DIR, exist_ok=True)

# Spatial Lag Model (SLM) and Spatial Error Model (SEM) as classical baselines
# These are equivalent to spdep in R: lagsarlm() and errorsarlm()
# Reference: Anselin (1988) Spatial Econometrics
# We run these on lung cancer only (primary outcome, lowest missingness)
# Results provide the benchmark R2 and RMSE the DL model must beat

# Predictor variables used in spatial models
# Same features that will go into the DNN branch of the DL model
OUTCOMES = {
    "mortality_lung_smoothed":        "Lung cancer",
    "mortality_breast_smoothed":      "Breast cancer",
    "mortality_colorectal_smoothed":  "Colorectal cancer",
}

PREDICTORS = [
    "pm25_satellite_lag10",
    "no2_satellite_lag10",
    "smoking_pct",
    "obesity_pct",
    "unemployment_pct",
    "child_poverty_pct",
    "median_income",
    "uninsured_adults_pct",
    "rural_pct",
    "pct_black",
    "pct_hispanic",
]

# Step 1: Load smoothed dataset and county shapefile
print("Loading data...")
df = pd.read_csv("data/processed/complete_data_smooth.csv")
print(f"  Dataset: {df.shape}")

counties  = gpd.read_file("data/shapefiles/tl_2020_us_county.shp")
excluded  = ["02", "15", "60", "66", "69", "72", "78"]
counties  = counties[~counties["STATEFP"].isin(excluded)].copy()
counties  = counties.to_crs("EPSG:4326")
counties["county_fips"] = counties["GEOID"].astype(int)

# Keep only counties in cancer data
cancer_fips = df["county_fips"].unique()
counties    = counties[counties["county_fips"].isin(cancer_fips)].copy()
counties    = counties.reset_index(drop=True)
print(f"  Counties: {len(counties)}")

# Step 2: Build spatial weights matrix
print("\nBuilding spatial weights matrix...")
w          = libpysal.weights.Queen.from_dataframe(counties, use_index=False)
w.id_order = list(range(len(counties)))
w.transform = "r"
print(f"  Weights: {w.n} units, avg {w.mean_neighbors:.1f} neighbors")

# Step 3: Run Moran's I, SLM and SEM for each cancer outcome
all_results = []

for OUTCOME, outcome_label in OUTCOMES.items():
    print(f"\n{'='*60}")
    print(f"Outcome: {outcome_label} ({OUTCOME})")
    print(f"{'='*60}")

    # Moran's I on 2015 cross-section
    print(f"\nMoran's I test (2015)...")
    df_2015   = df[df["year"] == 2015].copy()
    df_ord    = counties[["county_fips"]].merge(
        df_2015, on="county_fips", how="left"
    )
    y_2015    = df_ord[OUTCOME].fillna(
        df_ord[OUTCOME].median()
    ).values

    mi = Moran(y_2015, w)
    print(f"  Moran's I: {mi.I:.4f}  p = {mi.p_sim:.4f}")
    if mi.p_sim < 0.05:
        print("  Significant spatial autocorrelation confirmed")
    else:
        print("  No significant spatial autocorrelation")

    # Moran's I scatter plot
    fig, ax = plt.subplots(figsize=(7, 6))
    y_std   = (y_2015 - y_2015.mean()) / y_2015.std()
    wy_std  = libpysal.weights.lag_spatial(w, y_std)
    ax.scatter(y_std, wy_std, alpha=0.2, s=6, color="#1B7A8C")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    z   = np.polyfit(y_std, wy_std, 1)
    xln = np.linspace(y_std.min(), y_std.max(), 100)
    ax.plot(xln, np.poly1d(z)(xln), color="#D85A30", linewidth=2)
    ax.set_xlabel(f"{outcome_label} mortality (standardised)")
    ax.set_ylabel("Spatial lag (standardised)")
    ax.set_title(
        f"Moran's I = {mi.I:.3f}  p = {mi.p_sim:.4f}  "
        f"({outcome_label}, 2015)"
    )
    plt.tight_layout()
    cancer_tag  = OUTCOME.replace("mortality_", "").replace("_smoothed", "")
    moran_path  = os.path.join(PLOT_DIR, f"04_morans_{cancer_tag}.png")
    plt.savefig(moran_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {moran_path}")

    # SLM and SEM per year
    print(f"\nRunning OLS, SLM, SEM per year...")
    imputer  = SimpleImputer(strategy="median")

    for year in sorted(df["year"].unique()):
        df_yr  = df[df["year"] == year].copy()
        df_ord = counties[["county_fips"]].merge(
            df_yr, on="county_fips", how="left"
        )

        # Use only features with at least one non-missing value this year
        available_preds = [
            p for p in PREDICTORS
            if df_ord[p].notna().sum() > 0
        ]

        X_raw = df_ord[available_preds].values.astype(float)
        X     = imputer.fit_transform(X_raw)
        y     = df_ord[OUTCOME].fillna(
            df_ord[OUTCOME].median()
        ).values.astype(float)

        ols = spreg.OLS(
            y.reshape(-1, 1), X,
            w=w,
            name_y=OUTCOME,
            name_x=available_preds
        )

        slm = spreg.ML_Lag(
            y.reshape(-1, 1), X,
            w=w,
            name_y=OUTCOME,
            name_x=available_preds
        )

        sem = spreg.ML_Error(
            y.reshape(-1, 1), X,
            w=w,
            name_y=OUTCOME,
            name_x=available_preds
        )

        rmse_ols = np.sqrt(mean_squared_error(y, ols.predy.flatten()))
        rmse_slm = np.sqrt(mean_squared_error(y, slm.predy.flatten()))
        rmse_sem = np.sqrt(mean_squared_error(y, sem.predy.flatten()))

        all_results.append({
            "cancer":      outcome_label,
            "year":        year,
            "ols_r2":      round(ols.r2,    4),
            "ols_rmse":    round(rmse_ols,  3),
            "slm_r2":      round(slm.pr2,   4),
            "slm_rmse":    round(rmse_slm,  3),
            "slm_rho":     round(slm.rho,   4),
            "sem_r2":      round(sem.pr2,   4),
            "sem_rmse":    round(rmse_sem,  3),
            "sem_lambda":  round(sem.lam,   4),
        })

        print(f"  {year} {outcome_label[:4]:4s}:  "
              f"OLS R2={ols.r2:.3f}  "
              f"SLM R2={slm.pr2:.3f} rho={slm.rho:.3f}  "
              f"SEM R2={sem.pr2:.3f} lam={sem.lam:.3f}")

# Step 4: Summary across all outcomes
df_results = pd.DataFrame(all_results)
print(f"\n{'='*60}")
print("Mean performance by cancer type across all years:")
print(f"{'='*60}")
for cancer in df_results["cancer"].unique():
    sub = df_results[df_results["cancer"] == cancer]
    print(f"\n  {cancer}:")
    print(f"    OLS:  R2={sub['ols_r2'].mean():.3f}  "
          f"RMSE={sub['ols_rmse'].mean():.2f}")
    print(f"    SLM:  R2={sub['slm_r2'].mean():.3f}  "
          f"RMSE={sub['slm_rmse'].mean():.2f}  "
          f"rho={sub['slm_rho'].mean():.3f}")
    print(f"    SEM:  R2={sub['sem_r2'].mean():.3f}  "
          f"RMSE={sub['sem_rmse'].mean():.2f}  "
          f"lambda={sub['sem_lambda'].mean():.3f}")

# Step 5: Performance plot for all three outcomes
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Spatial Model Performance by Cancer Type", fontsize=14)

for col_idx, cancer in enumerate(df_results["cancer"].unique()):
    sub   = df_results[df_results["cancer"] == cancer]
    years = sub["year"].values

    axes[0, col_idx].plot(years, sub["ols_r2"], label="OLS",
        color="#888780", linewidth=1.5, linestyle="--", marker="o", markersize=4)
    axes[0, col_idx].plot(years, sub["slm_r2"], label="SLM",
        color="#1B7A8C", linewidth=2, marker="o", markersize=4)
    axes[0, col_idx].plot(years, sub["sem_r2"], label="SEM",
        color="#D85A30", linewidth=2, marker="o", markersize=4)
    axes[0, col_idx].set_title(f"{cancer} — R²", fontsize=11)
    axes[0, col_idx].set_xlabel("Year")
    axes[0, col_idx].set_ylabel("R²")
    axes[0, col_idx].legend(fontsize=8)
    axes[0, col_idx].grid(axis="y", alpha=0.3)

    axes[1, col_idx].plot(years, sub["ols_rmse"], label="OLS",
        color="#888780", linewidth=1.5, linestyle="--", marker="o", markersize=4)
    axes[1, col_idx].plot(years, sub["slm_rmse"], label="SLM",
        color="#1B7A8C", linewidth=2, marker="o", markersize=4)
    axes[1, col_idx].plot(years, sub["sem_rmse"], label="SEM",
        color="#D85A30", linewidth=2, marker="o", markersize=4)
    axes[1, col_idx].set_title(f"{cancer} — RMSE", fontsize=11)
    axes[1, col_idx].set_xlabel("Year")
    axes[1, col_idx].set_ylabel("RMSE (per 100,000)")
    axes[1, col_idx].legend(fontsize=8)
    axes[1, col_idx].grid(axis="y", alpha=0.3)

plt.tight_layout()
perf_path = os.path.join(PLOT_DIR, "09_spatial_model_performance.png")
plt.savefig(perf_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {perf_path}")

# Step 6: Save full results table
res_path = os.path.join(OUT_DIR, "spatial_model_results.csv")
df_results.to_csv(res_path, index=False)
print(f"Saved: {res_path}")