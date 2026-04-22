import os
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# Step 1: Load all four processed files
print("Loading processed files...")

df_acag = pd.read_csv("data/processed/acag_county_pm25.csv")
df_epa  = pd.read_csv("data/processed/epa_county_pollution.csv")
df_mort = pd.read_csv("data/processed/cancer_mortality_model_ready.csv")
df_chr  = pd.read_csv("data/processed/chr_county_socioeconomic.csv")

print(f"  ACAG:    {df_acag.shape}")
print(f"  EPA:     {df_epa.shape}")
print(f"  Cancer:  {df_mort.shape}")
print(f"  CHR:     {df_chr.shape}")

# Step 2: Drop high-missingness columns from CHR that were flagged earlier
# These variables either changed codes between years or were unavailable
# for most of the 2010-2020 period
CHR_DROP = [
    "poverty_pct",         # 98.3% missing
    "male_65plus_pct",     # 98.3% missing
    "female_65plus_pct",   # 98.3% missing
    "income_inequality",   # 36.4% missing
    "pct_white",           # 27.3% missing
    "uninsured_pct",       # 18.2% missing — uninsured_adults_pct is better
]
df_chr = df_chr.drop(columns=[c for c in CHR_DROP if c in df_chr.columns])
print(f"\nCHR after dropping high-missingness columns: {df_chr.shape}")

# Step 3: Compute 10-year lagged pollution features
# For each cancer mortality year T, we use mean pollution from T-10 to T-1
# Biological rationale: lung cancer latency period is 10-30 years
# The LSTM will learn which years within the window matter most
#
# Cancer years: 2010 to 2020
# Pollution window for 2010: mean of 2000 to 2009
# Pollution window for 2020: mean of 2010 to 2019
print("\nComputing 10-year lagged pollution features...")

CANCER_YEARS = range(2010, 2021)
LAG_YEARS    = 10

lag_records = []

for cancer_year in CANCER_YEARS:
    # The pollution years we average over for this cancer year
    poll_years = list(range(cancer_year - LAG_YEARS, cancer_year))

    # ACAG lagged PM2.5 — covers all counties
    acag_window = df_acag[df_acag["year"].isin(poll_years)]
    acag_lag    = (
        acag_window.groupby("county_fips")["pm25_satellite"]
        .mean()
        .reset_index()
        .rename(columns={"pm25_satellite": "pm25_satellite_lag10"})
    )

    # EPA lagged concentrations — sparser coverage
    epa_window = df_epa[df_epa["year"].isin(poll_years)]
    epa_lag    = (
        epa_window.groupby("county_fips")[["pm25", "no2", "ozone", "so2"]]
        .mean()
        .reset_index()
        .rename(columns={
            "pm25":  "pm25_ground_lag10",
            "no2":   "no2_lag10",
            "ozone": "ozone_lag10",
            "so2":   "so2_lag10",
        })
    )

    # Merge ACAG and EPA lags together
    poll_lag = acag_lag.merge(epa_lag, on="county_fips", how="outer")
    poll_lag["year"] = cancer_year
    lag_records.append(poll_lag)

df_pollution_lag = pd.concat(lag_records, ignore_index=True)
print(f"  Lagged pollution table: {df_pollution_lag.shape}")
print(f"  Years covered: {sorted(df_pollution_lag['year'].unique())}")

# Step 4: Merge everything together
# Base: cancer mortality (defines which county-years exist)
# Left join pollution lag (all cancer counties, fill missing with NaN)
# Left join CHR (socioeconomic features for the same year)
print("\nMerging all data sources...")

df_merged = df_mort.copy()

# Drop county_name to keep the table clean — FIPS is our key
if "county_name" in df_merged.columns:
    df_merged = df_merged.drop(columns=["county_name"])

# Merge lagged pollution
df_merged = df_merged.merge(
    df_pollution_lag,
    on=["county_fips", "year"],
    how="left"
)
print(f"  After pollution merge: {df_merged.shape}")

# Merge CHR socioeconomic features
df_merged = df_merged.merge(
    df_chr,
    on=["county_fips", "year"],
    how="left"
)
print(f"  After CHR merge: {df_merged.shape}")

# Step 5: Sort and check the result
df_merged = df_merged.sort_values(["county_fips", "year"]).reset_index(drop=True)

print(f"\nFinal merged dataset shape: {df_merged.shape}")
print(f"Columns ({len(df_merged.columns)}):")
for col in df_merged.columns:
    print(f"  {col}")

# Step 6: Missingness report
print("\nMissing values per column:")
for col in df_merged.columns:
    n_miss = df_merged[col].isna().sum()
    pct    = round(n_miss / len(df_merged) * 100, 1)
    bar    = "#" * int(pct / 5)
    print(f"  {col:35s}: {pct:5.1f}%  {bar}")

# Step 7: Summary statistics for key variables
print("\nKey variable summaries:")
key_cols = [
    "pm25_satellite_lag10", "pm25_ground_lag10",
    "mortality_lung_per100k", "mortality_breast_per100k",
    "mortality_colorectal_per100k", "smoking_pct", "child_poverty_pct"
]
for col in key_cols:
    if col in df_merged.columns:
        s = df_merged[col].describe()
        print(f"\n  {col}:")
        print(f"    mean={s['mean']:.3f}  std={s['std']:.3f}  "
              f"min={s['min']:.3f}  max={s['max']:.3f}")

# Step 8: Save
out_path = os.path.join(OUT_DIR, "merged_panel.csv")
df_merged.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Shape: {df_merged.shape}")

print("\nFirst 5 rows:")
print(df_merged.head(5).to_string())

# Step 8b: Drop pollution features with excessive missingness
# NO2 and SO2 ground monitors are extremely sparse (>80% missing)
# They cover too few counties to be reliable model inputs
# PM2.5 satellite (0.4%), PM2.5 ground (70.9%), and ozone (63.9%)
# are retained as the three pollution features
POLL_DROP = ["no2_lag10", "so2_lag10"]
df_merged = df_merged.drop(columns=POLL_DROP)
print(f"\nAfter dropping sparse pollution columns: {df_merged.shape}")
print(f"Final pollution features retained: pm25_satellite_lag10, pm25_ground_lag10, ozone_lag10")