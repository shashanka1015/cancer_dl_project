import os
import glob
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

CONC_DIR = "data/epa_raw"
OUT_DIR  = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# Air pollution from 2000 to 2019 creates a 10-year exposure lag
# for cancer mortality predictions covering 2010 to 2020
# Example: predict 2015 mortality using mean PM2.5 from 2005 to 2014
YEARS = range(2000, 2020)

# Pollutants we want and the exact Parameter Name string in the EPA files
POLLUTANT_FILTERS = {
    "pm25":  "PM2.5 - Local Conditions",
    "no2":   "Nitrogen dioxide (NO2)",
    "ozone": "Ozone",
    "so2":   "Sulfur dioxide",
}

# For ozone we only want the 8-hour standard, not the 1-hour standard
# The 8-hour standard is the current regulatory standard for health effects
OZONE_SAMPLE_DURATION = "8-HR RUN AVG BEGIN HOUR"

# For PM2.5 we only want the 24-hour local conditions measurement
PM25_SAMPLE_DURATION = "24 HOUR"


def process_one_year(year):
    # Find the concentration file for this year
    pattern = os.path.join(
        CONC_DIR,
        f"annual_conc_by_monitor_{year}",
        f"annual_conc_by_monitor_{year}.csv"
    )
    matches = glob.glob(pattern)

    if not matches:
        print(f"  {year}: file not found, skipping")
        return None

    filepath = matches[0]

    # Read only the columns we need to save memory
    usecols = [
        "State Code", "County Code", "Parameter Name",
        "Sample Duration", "Arithmetic Mean", "Completeness Indicator"
    ]
    df = pd.read_csv(filepath, usecols=usecols, dtype=str)

    # Remove rows where State Code or County Code are not numeric
    # EPA files sometimes contain repeated header rows mid-file
    # with values like 'CC' or 'State Code' in those columns
    df = df[df["State Code"].str.isnumeric()].copy()
    df = df[df["County Code"].str.isnumeric()].copy()

    # Build FIPS code directly with string operations
    # FIPS is 5 digits: 2-digit state + 3-digit county
    df["county_fips"] = (
        df["State Code"].str.zfill(2) + df["County Code"].str.zfill(3)
    )

    # Convert arithmetic mean to float
    df["Arithmetic Mean"] = pd.to_numeric(df["Arithmetic Mean"], errors="coerce")

    # Keep only monitors with complete data
    df = df[df["Completeness Indicator"] == "Y"].copy()

    # Exclude non-contiguous states and territories
    # 02 = Alaska, 15 = Hawaii, 60+ = territories
    excluded_prefixes = ("02", "15", "60", "66", "69", "72", "78")
    df = df[~df["county_fips"].str.startswith(excluded_prefixes)].copy()

    pollutant_dfs = []

    for pollutant_key, param_name in POLLUTANT_FILTERS.items():

        # Filter to this pollutant
        df_poll = df[df["Parameter Name"] == param_name].copy()

        # For ozone, keep only the 8-hour standard
        if pollutant_key == "ozone":
            df_poll = df_poll[
                df_poll["Sample Duration"] == OZONE_SAMPLE_DURATION
            ].copy()

        # For PM2.5, keep only the 24-hour local conditions measurement
        if pollutant_key == "pm25":
            df_poll = df_poll[
                df_poll["Sample Duration"] == PM25_SAMPLE_DURATION
            ].copy()

        if df_poll.empty:
            continue

        # Average across all monitors within the same county
        # This gives one annual mean value per county per pollutant
        county_mean = (
            df_poll.groupby("county_fips")["Arithmetic Mean"]
            .mean()
            .reset_index()
        )
        county_mean.columns = ["county_fips", pollutant_key]
        pollutant_dfs.append(county_mean)

    if not pollutant_dfs:
        return None

    # Merge all pollutants for this year into one table
    df_year = pollutant_dfs[0]
    for other_df in pollutant_dfs[1:]:
        df_year = df_year.merge(other_df, on="county_fips", how="outer")

    df_year["year"] = year
    return df_year


# Step 1: Process all years
print("Processing EPA concentration files year by year...")
all_years = []

for year in YEARS:
    df_year = process_one_year(year)
    if df_year is not None:
        all_years.append(df_year)
        n_counties = len(df_year)
        print(f"  {year}: {n_counties} counties with at least one pollutant")

# Step 2: Combine all years
print("\nCombining all years...")
df_epa = pd.concat(all_years, ignore_index=True)
df_epa = df_epa.sort_values(["county_fips", "year"]).reset_index(drop=True)

print(f"Combined shape: {df_epa.shape}")
print(f"\nMissing values per pollutant:")
for col in ["pm25", "no2", "ozone", "so2"]:
    if col in df_epa.columns:
        n_miss = df_epa[col].isna().sum()
        pct    = round(n_miss / len(df_epa) * 100, 1)
        print(f"  {col:8s}: {n_miss:6,} missing ({pct}%)")

print(f"\nPM2.5 ground level summary:")
print(df_epa["pm25"].describe().round(3))

# Step 3: Save
out_path = os.path.join(OUT_DIR, "epa_county_pollution.csv")
df_epa.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

print("\nFirst few rows:")
print(df_epa.head(10).to_string())