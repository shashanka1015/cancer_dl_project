import os
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

CHR_DIR = "data/chr_raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

VARIABLES = {
    "v009_rawvalue": "smoking_pct",
    "v011_rawvalue": "obesity_pct",
    "v023_rawvalue": "unemployment_pct",
    "v118_rawvalue": "poverty_pct",
    "v024_rawvalue": "child_poverty_pct",
    "v063_rawvalue": "median_income",
    "v044_rawvalue": "income_inequality",
    "v085_rawvalue": "uninsured_pct",
    "v003_rawvalue": "uninsured_adults_pct",
    "v058_rawvalue": "rural_pct",
    "v051_rawvalue": "population",
    "v054_rawvalue": "pct_black",
    "v056_rawvalue": "pct_hispanic",
    "v126_rawvalue": "pct_white",
    "v004_rawvalue": "primary_care_ratio",
    "v017_rawvalue": "male_65plus_pct",
    "v032_rawvalue": "female_65plus_pct",
}


def read_chr_year(year):
    filepath = os.path.join(CHR_DIR, f"chr_analytic_{year}.csv")

    if not os.path.exists(filepath):
        print(f"  {year}: file not found, skipping")
        return None

    # CHR files have a two-row header before data begins
    # Row 1 (index 0): long descriptive labels e.g. "Adult smoking raw value"
    # Row 2 (index 1): short variable codes e.g. "v009_rawvalue"
    # Use index 1 to have short codes as column headers
    both_rows   = pd.read_csv(filepath, nrows=2, header=None)
    short_codes = both_rows.iloc[1].tolist()

    # Read the actual data, skipping both header rows
    df          = pd.read_csv(filepath, skiprows=2, header=None, dtype=str)
    df.columns  = short_codes

    # Remove state-level summary rows where countycode is 0 or missing
    # State rows have countycode = 0, county rows have a real county code
    if "countycode" in df.columns:
        df = df[pd.to_numeric(df["countycode"], errors="coerce") > 0].copy()

    # Build 5-digit FIPS code from state and county code columns
    if "statecode" in df.columns and "countycode" in df.columns:
        df["county_fips"] = (
            df["statecode"].str.strip().str.zfill(2) +
            df["countycode"].str.strip().str.zfill(3)
        )
    elif "fipscode" in df.columns:
        df["county_fips"] = df["fipscode"].str.strip().str.zfill(5)
    else:
        print(f"  {year}: cannot find FIPS columns, skipping")
        return None

    # Identify which of our target variables are present in this year
    available    = {k: v for k, v in VARIABLES.items() if k in df.columns}
    missing_vars = [k for k in VARIABLES if k not in df.columns]

    if missing_vars:
        print(f"  {year}: variables not found: {missing_vars}")

    # Keep only FIPS and the available target variables
    cols_to_keep = ["county_fips"] + list(available.keys())
    df           = df[cols_to_keep].copy()
    df           = df.rename(columns=available)

    # Convert all feature columns to numeric
    for col in df.columns:
        if col != "county_fips":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove non-contiguous states and territories
    excluded = ("02", "15", "60", "66", "69", "72", "78")
    df       = df[~df["county_fips"].str.startswith(excluded)].copy()

    df["year"] = year
    return df


# Step 1: Process all years
print("Processing County Health Rankings files 2010 to 2020...")
all_years = []

for year in range(2010, 2021):
    df_year = read_chr_year(year)
    if df_year is not None:
        n_features = df_year.shape[1] - 2
        print(f"  {year}: {len(df_year)} counties, {n_features} features")
        all_years.append(df_year)

# Step 2: Combine all years
print("\nCombining all years...")
df_chr = pd.concat(all_years, ignore_index=True)
df_chr = df_chr.sort_values(["county_fips", "year"]).reset_index(drop=True)
print(f"Combined shape: {df_chr.shape}")

# Step 3: Check missingness
print("\nMissing values per feature:")
for col in df_chr.columns:
    if col not in ["county_fips", "year"]:
        n_miss = df_chr[col].isna().sum()
        pct    = round(n_miss / len(df_chr) * 100, 1)
        print(f"  {col:25s}: {n_miss:5,} missing ({pct}%)")

# Step 4: Sanity checks
print("\nSmoking prevalence summary (expect 0.10 to 0.35):")
if "smoking_pct" in df_chr.columns:
    print(df_chr["smoking_pct"].describe().round(4))

print("\nPoverty rate summary (expect 0.05 to 0.50):")
if "poverty_pct" in df_chr.columns:
    print(df_chr["poverty_pct"].describe().round(4))

# Step 5: Save
out_path = os.path.join(OUT_DIR, "chr_county_socioeconomic.csv")
df_chr.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Shape: {df_chr.shape}")

print("\nFirst few rows:")
print(df_chr.head(5).to_string())

# Step 4b: Drop variables with excessive missingness
# These variables either changed codes between years or were not
# collected consistently across the 2010 to 2020 period
# child_poverty_pct serves as our poverty proxy (0% missing)
DROP_COLS = [
    "poverty_pct",         # 98.3% missing - v118 only exists in 2020
    "male_65plus_pct",     # 98.3% missing - age group codes changed
    "female_65plus_pct",   # 98.3% missing - age group codes changed
    "income_inequality",   # 36.4% missing - only from 2015 onwards
    "pct_white",           # 27.3% missing - only from 2013 onwards
    "uninsured_pct",       # 18.2% missing - uninsured_adults_pct is better
]

cols_to_drop = [c for c in DROP_COLS if c in df_chr.columns]
df_chr       = df_chr.drop(columns=cols_to_drop)

print(f"\nAfter dropping high-missingness variables:")
print(f"Final shape: {df_chr.shape}")
print(f"Final columns: {df_chr.columns.tolist()}")