import os
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

CANCER_DIR = "data/cancer_raw"
OUT_DIR    = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# All eight cancer types with their filenames and short labels
# The label becomes the column name in the final merged dataset
CANCER_FILES = {
    "lung":        "mortality_lung_C34_2010_2020.csv",
    "colorectal":  "mortality_colorectal_C18C20_2010_2020.csv",
    "bladder":     "mortality_bladder_C67_2010_2020.csv",
    "prostate":    "mortality_prostate_C61_2010_2020.csv",
    "leukemia":    "mortality_leukemia_C91C95_2010_2020.csv",
    "kidney":      "mortality_kidney_C64C66_2010_2020.csv",
    "breast":      "mortality_breast_C50_2010_2020.csv",
    "nhl":         "mortality_nhl_C82C85_2010_2020.csv",
}


def read_wonder_file(filepath, cancer_label):
    # CDC WONDER CSV files have two sections:
    # 1. Data rows where the Notes column is empty
    # 2. A footer section with metadata notes where Notes column is filled
    # Read everything and keep data rows

    df_raw = pd.read_csv(filepath, dtype=str)

    # Data rows have an empty Notes column
    # Footer rows have text in Notes
    df = df_raw[df_raw["Notes"].isna()].copy()

    # Drop rows where County Code is missing (total/subtotal rows)
    df = df[df["County Code"].notna()].copy()

    # Rename columns to simpler names
    df = df.rename(columns={
        "County Code":      "county_fips",
        "County":           "county_name",
        "Year":             "year",
        "Deaths":           "deaths_raw",
        "Population":       "population",
        "Age Adjusted Rate": "age_adj_rate",
    })

    # Keep only the columns required columns
    df = df[["county_fips", "county_name", "year",
             "deaths_raw", "population", "age_adj_rate"]].copy()

    # Convert year and population to numbers
    df["year"]       = pd.to_numeric(df["year"], errors="coerce")
    df["population"] = pd.to_numeric(df["population"], errors="coerce")

    # Deaths column contains numbers or "Suppressed" for very small counts
    # Suppressed means fewer than 10 deaths in that county-year
    # Flag these separately to handle with Bayesian smoothing later
    df["suppressed"] = df["deaths_raw"].str.strip() == "Suppressed"
    df["deaths"]     = pd.to_numeric(df["deaths_raw"], errors="coerce")

    # Age adjusted rate contains numbers or "Unreliable" for small counts
    # Unreliable means the death count exists but the rate is statistically unstable
    # Flag these and calculate crude rate from raw counts instead
    df["rate_unreliable"] = df["age_adj_rate"].str.strip() == "Unreliable"
    df["age_adj_rate"]    = pd.to_numeric(df["age_adj_rate"], errors="coerce")

    # Calculate crude mortality rate per 100,000 population from raw counts
    # This is more stable than using the CDC-provided crude rate because we can control exactly how we handle small numbers
    df["crude_rate_per100k"] = (df["deaths"] / df["population"]) * 100000

    # Add cancer type label
    df["cancer_type"] = cancer_label

    # Clean up FIPS codes — ensure they are always 5 characters with leading zeros
    # Some counties like Alabama (01001) need the leading zero preserved
    df["county_fips"] = df["county_fips"].str.strip().str.zfill(5)

    return df


# Step 1: Read and clean all eight cancer files
print("Reading and cleaning all cancer mortality files...")
all_cancer_dfs = []

for cancer_label, filename in CANCER_FILES.items():
    filepath = os.path.join(CANCER_DIR, filename)

    if not os.path.exists(filepath):
        print(f"  Missing: {filename} -- skipping")
        continue

    df = read_wonder_file(filepath, cancer_label)
    all_cancer_dfs.append(df)

    n_rows        = len(df)
    n_suppressed  = df["suppressed"].sum()
    n_unreliable  = df["rate_unreliable"].sum()
    n_counties    = df["county_fips"].nunique()
    n_years       = df["year"].nunique()

    print(f"  {cancer_label:12s}: {n_rows:6,} rows | "
          f"{n_counties:4} counties | "
          f"{n_years} years | "
          f"{n_suppressed:4} suppressed | "
          f"{n_unreliable:4} unreliable rates")

# Step 2: Combine all cancer types into one long-format table
# Long format: one row per county-year-cancer_type combination
print("\nCombining all cancer types into long format...")
df_long = pd.concat(all_cancer_dfs, ignore_index=True)
print(f"Long format shape: {df_long.shape}")

# Step 3: Create wide format table — one row per county-year
# Each cancer type becomes its own column for mortality rate
# This is the format the model will use for training
print("\nPivoting to wide format (one row per county-year)...")

df_wide = df_long.pivot_table(
    index   = ["county_fips", "county_name", "year"],
    columns = "cancer_type",
    values  = "crude_rate_per100k",
    aggfunc = "mean"
).reset_index()

# Rename mortality columns to be clearly labelled
df_wide.columns.name = None
rate_cols = [c for c in df_wide.columns
             if c not in ["county_fips", "county_name", "year"]]
df_wide = df_wide.rename(
    columns={c: f"mortality_{c}_per100k" for c in rate_cols}
)

df_wide = df_wide.sort_values(["county_fips", "year"]).reset_index(drop=True)

print(f"Wide format shape: {df_wide.shape}")
print(f"Columns: {df_wide.columns.tolist()}")

# Step 4: Check how complete the data is
# Some county-years will be missing because CDC suppressed counts below 10
print("\nMissing value summary (suppressed county-years):")
for col in df_wide.columns:
    if col.startswith("mortality_"):
        n_missing = df_wide[col].isna().sum()
        pct       = round(n_missing / len(df_wide) * 100, 1)
        print(f"  {col:45s}: {n_missing:5,} missing ({pct}%)")

# Step 5: Save both formats
long_path = os.path.join(OUT_DIR, "cancer_mortality_long.csv")
wide_path = os.path.join(OUT_DIR, "cancer_mortality_wide.csv")

df_long.to_csv(long_path, index=False)
df_wide.to_csv(wide_path, index=False)

print(f"\nSaved long format: {long_path}")
print(f"Saved wide format: {wide_path}")

# Step 6: Quick check on lung cancer values
# Lung cancer rates should be roughly 30-80 per 100,000 for US counties
print("\nLung cancer mortality rate summary:")
lung_col = "mortality_lung_per100k"
if lung_col in df_wide.columns:
    print(df_wide[lung_col].describe().round(2))

print("\nFirst few rows of wide format:")
print(df_wide.head(10).to_string())

# Step 7: Flag which cancer types have sufficient data for modelling
# Cancer types with more than 65% missing values at county-year level are excluded because Bayesian smoothing cannot reliably impute rates when the majority of observations are suppressed by CDC
print("\nData sufficiency assessment for modelling:")

threshold = 65.0
sufficient = []
insufficient = []

for col in df_wide.columns:
    if col.startswith("mortality_"):
        pct_missing = df_wide[col].isna().mean() * 100
        cancer_name = col.replace("mortality_", "").replace("_per100k", "")
        if pct_missing <= threshold:
            sufficient.append(cancer_name)
            print(f"  INCLUDE  {cancer_name:15s}: {pct_missing:.1f}% missing")
        else:
            insufficient.append(cancer_name)
            print(f"  EXCLUDE  {cancer_name:15s}: {pct_missing:.1f}% missing")

print(f"\nCancers included in model: {sufficient}")
print(f"Cancers excluded from model: {insufficient}")

# Save a filtered version with only sufficient cancer types
keep_cols = ["county_fips", "county_name", "year"] + \
            [f"mortality_{c}_per100k" for c in sufficient]
df_model = df_wide[keep_cols].copy()

model_path = os.path.join(OUT_DIR, "cancer_mortality_model_ready.csv")
df_model.to_csv(model_path, index=False)
print(f"\nModel-ready file saved: {model_path}")
print(f"Shape: {df_model.shape}")