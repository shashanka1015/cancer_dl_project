import os
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# Step 1: Load all five processed files
print("Loading processed files...")

df_acag = pd.read_csv("data/processed/acag_county_pm25.csv")
df_epa  = pd.read_csv("data/processed/epa_county_pollution.csv")
df_mort = pd.read_csv("data/processed/cancer_mortality_model_ready.csv")
df_chr  = pd.read_csv("data/processed/chr_county_socioeconomic.csv")
df_no2  = pd.read_csv("data/processed/no2_satellite_county.csv")

print(f"  ACAG satellite PM2.5:  {df_acag.shape}")
print(f"  EPA ground pollution:  {df_epa.shape}")
print(f"  Cancer mortality:      {df_mort.shape}")
print(f"  CHR socioeconomic:     {df_chr.shape}")
print(f"  NO2 satellite:         {df_no2.shape}")

# Step 2: Drop high-missingness CHR columns identified during processing
# These variables were unavailable for most of the 2010-2020 period
CHR_DROP = [
    "poverty_pct",
    "male_65plus_pct",
    "female_65plus_pct",
    "income_inequality",
    "pct_white",
    "uninsured_pct",
    "primary_care_ratio",    # Scaling inconsistency across CHR years
]

df_chr = df_chr.drop(columns=[c for c in CHR_DROP if c in df_chr.columns])
print(f"\nCHR after dropping sparse columns: {df_chr.shape}")

# Step 3: Compute 10-year lagged pollution features
# For each cancer mortality year T, we average pollution from T-10 to T-1
# Biological rationale: lung cancer latency period is 10-30 years
# The LSTM will learn which years within the window matter most
print("\nComputing 10-year lagged pollution features...")

CANCER_YEARS = range(2010, 2021)
LAG_YEARS    = 10

lag_records = []

for cancer_year in CANCER_YEARS:
    poll_years = list(range(cancer_year - LAG_YEARS, cancer_year))

    # ACAG satellite PM2.5 lag - complete coverage all counties
    acag_window = df_acag[df_acag["year"].isin(poll_years)]
    acag_lag    = (
        acag_window.groupby("county_fips")["pm25_satellite"]
        .mean().reset_index()
        .rename(columns={"pm25_satellite": "pm25_satellite_lag10"})
    )

    # EPA ground PM2.5, ozone lag - sparser monitor coverage
    epa_window = df_epa[df_epa["year"].isin(poll_years)]
    epa_lag    = (
        epa_window.groupby("county_fips")[["pm25", "ozone"]]
        .mean().reset_index()
        .rename(columns={
            "pm25":  "pm25_ground_lag10",
            "ozone": "ozone_lag10",
        })
    )

    # NO2 satellite lag - available 2000 and 2005-2019
    # Years 2001-2004 are absent from the dataset (pre-OMI satellite era)
    # Mean is computed over whichever years in the window are available
    no2_window = df_no2[df_no2["year"].isin(poll_years)]
    no2_lag    = (
        no2_window.groupby("county_fips")["no2_satellite_ppb"]
        .mean().reset_index()
        .rename(columns={"no2_satellite_ppb": "no2_satellite_lag10"})
    )

    # Merge all pollution lags for this cancer year
    poll_lag = acag_lag.merge(epa_lag, on="county_fips", how="outer")
    poll_lag = poll_lag.merge(no2_lag, on="county_fips", how="outer")
    poll_lag["year"] = cancer_year
    lag_records.append(poll_lag)

df_pollution_lag = pd.concat(lag_records, ignore_index=True)
print(f"  Lagged pollution table: {df_pollution_lag.shape}")

# Step 4: Merge all datasets into one complete panel
# Base: cancer mortality (defines which county-years exist)
# Left join pollution lag, CHR, on county_fips and year
print("\nMerging all data sources...")

df_merged = df_mort.copy()
if "county_name" in df_merged.columns:
    df_merged = df_merged.drop(columns=["county_name"])

df_merged = df_merged.merge(df_pollution_lag, on=["county_fips", "year"], how="left")
print(f"  After pollution merge: {df_merged.shape}")

df_merged = df_merged.merge(df_chr, on=["county_fips", "year"], how="left")
print(f"  After CHR merge:       {df_merged.shape}")

df_merged = df_merged.sort_values(["county_fips", "year"]).reset_index(drop=True)

# Step 5: Final column summary
print(f"\nFinal dataset shape: {df_merged.shape}")
print(f"\nAll columns ({len(df_merged.columns)}):")
for col in df_merged.columns:
    n_miss = df_merged[col].isna().sum()
    pct    = round(n_miss / len(df_merged) * 100, 1)
    print(f"  {col:35s}: {pct}% missing")

# Step 6: Key variable summaries
print("\nKey variable summaries:")
key_cols = [
    "pm25_satellite_lag10", "pm25_ground_lag10",
    "no2_satellite_lag10",  "ozone_lag10",
    "mortality_lung_per100k", "smoking_pct",
]
for col in key_cols:
    if col in df_merged.columns:
        s = df_merged[col].describe()
        print(f"\n  {col}:")
        print(f"    mean={s['mean']:.3f}  std={s['std']:.3f}  "
              f"min={s['min']:.3f}  max={s['max']:.3f}")

# Step 7: Save as complete_data.csv
out_path = os.path.join(OUT_DIR, "complete_data.csv")
df_merged.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Shape: {df_merged.shape}")

print("\nFirst 5 rows:")
print(df_merged.head(5).to_string())