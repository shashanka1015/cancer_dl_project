import os
import numpy as np
import pandas as pd
import geopandas as gpd
import libpysal
from esda.smoothing import Spatial_Empirical_Bayes

os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = "data/processed"

# Spatial Empirical Bayes smoothing for cancer mortality rates
# This method shrinks each county's observed rate toward a spatially weighted
# local mean, stabilizing estimates in counties with small death counts
# Reference: Assuncao and Correa (1992); Anselin et al. (2006)
# Computationally efficient alternative to full BYM MCMC which would require
# hours per model fit at 3,108 county scale — not practical for a course project
# The SEB achieves the same scientific goal: borrowing spatial strength

# Step 1: Load the original unsmoothed complete dataset
# Always read from complete_data_original.csv to avoid accumulating
# smoothed columns from previous runs
print("Loading data...")
df = pd.read_csv("data/processed/complete_data_original.csv")
print(f"  Complete data: {df.shape}")

counties = gpd.read_file("data/shapefiles/tl_2020_us_county.shp")
excluded = ["02", "15", "60", "66", "69", "72", "78"]
counties = counties[~counties["STATEFP"].isin(excluded)].copy()
counties = counties.to_crs("EPSG:4326")
counties["county_fips"] = counties["GEOID"].astype(int)
print(f"  Shapefile counties: {len(counties)}")

# Keep only counties that appear in the cancer mortality data
# Using all 3108 shapefile counties introduces ~980 phantom counties
# with zero deaths that pull the spatial mean toward zero during SEB
cancer_fips = df["county_fips"].unique()
counties    = counties[counties["county_fips"].isin(cancer_fips)].copy()
counties    = counties.reset_index(drop=True)
print(f"  Counties with cancer data: {len(counties)}")

# Step 2: Build spatial weights matrix from cancer-data counties only
# Queen contiguity: counties sharing any boundary point are neighbors
# This is standard in spatial epidemiology for US county data
print("\nBuilding spatial weights matrix...")
w           = libpysal.weights.Queen.from_dataframe(counties, use_index=False)
w.id_order  = list(range(len(counties)))
w.transform = "r"
print(f"  Weights matrix: {w.n} units")
print(f"  Average neighbors per county: {w.mean_neighbors:.1f}")

# Step 3: Apply SEB smoothing for each cancer type and year
# SEB requires event counts (deaths) and population at risk
# We back-calculate deaths from crude rate and population
# crude_rate_per100k = (deaths / population) * 100000
# Therefore deaths = crude_rate * population / 100000

CANCER_COLS = {
    "mortality_lung_per100k":       "lung",
    "mortality_breast_per100k":     "breast",
    "mortality_colorectal_per100k": "colorectal",
}

print("\nApplying Spatial Empirical Bayes smoothing...")
print("One line per cancer type per year:")

smoothed_records = []
years = sorted(df["year"].unique())

for year in years:
    df_year = df[df["year"] == year].copy()

    # Merge with cancer-data counties only — no phantom counties
    # SEB requires data ordered the same way as the weights matrix
    df_ordered = counties[["county_fips"]].merge(
        df_year, on="county_fips", how="left"
    )

    for mort_col, cancer_name in CANCER_COLS.items():

        pop    = df_ordered["population"].values.astype(float)
        rate   = df_ordered[mort_col].values.astype(float)
        deaths = rate * pop / 100000

        # For counties with missing population use 10,000 as a conservative proxy
        # This allows spatial smoothing to proceed without producing zeros
        # Counties using the proxy are flagged and reverted to raw rate afterward
        proxy_used = ~np.isfinite(pop) | (pop <= 0)
        pop_seb    = np.where(proxy_used, 10000.0, pop)
        deaths_seb = np.where(np.isfinite(deaths), deaths, 0.0)

        # Apply Spatial Empirical Bayes smoothing
        # flatten() ensures 1D array regardless of what shape esda returns
        seb           = Spatial_Empirical_Bayes(deaths_seb, pop_seb, w)
        smoothed_rate = np.array(seb.r).flatten() * 100000

        # Where proxy population was used revert to the raw observed rate
        # This prevents inaccurate smoothing for counties with unknown population
        # Where both smoothed and raw rate are unavailable use 0
        final_rate = np.where(proxy_used, rate, smoothed_rate)
        final_rate = np.where(np.isfinite(final_rate), final_rate, 0.0)

        result = pd.DataFrame({
            "county_fips":   df_ordered["county_fips"].values.flatten(),
            "year":          year,
            "cancer_type":   cancer_name,
            "smoothed_rate": final_rate,
        })
        result = result.rename(
            columns={"smoothed_rate": f"mortality_{cancer_name}_smoothed"}
        )
        smoothed_records.append(result)

    print(f"  {year}: smoothed lung, breast, colorectal")

# Step 4: Combine smoothed rates and pivot to wide format
print("\nCombining smoothed rates...")
df_smoothed_long = pd.concat(smoothed_records, ignore_index=True)

df_smoothed_wide = df_smoothed_long.pivot_table(
    index   = ["county_fips", "year"],
    columns = "cancer_type",
    values  = [c for c in df_smoothed_long.columns
               if c.startswith("mortality_") and c.endswith("_smoothed")],
    aggfunc = "mean"
).reset_index()

df_smoothed_wide.columns = [
    "_".join(c).strip("_") if isinstance(c, tuple) else c
    for c in df_smoothed_wide.columns
]

df_smoothed_wide.columns = [
    col.replace("mortality_lung_smoothed_lung", "mortality_lung_smoothed")
       .replace("mortality_breast_smoothed_breast", "mortality_breast_smoothed")
       .replace("mortality_colorectal_smoothed_colorectal", "mortality_colorectal_smoothed")
    for col in df_smoothed_wide.columns
]

print(f"Smoothed wide table: {df_smoothed_wide.shape}")
print(f"Columns: {df_smoothed_wide.columns.tolist()}")

# Step 5: Merge smoothed rates into the original complete dataset
print("\nMerging smoothed rates into complete dataset...")
df_final = df.merge(df_smoothed_wide, on=["county_fips", "year"], how="left")
print(f"Final shape: {df_final.shape}")

# Step 6: Check — compare raw vs smoothed lung cancer rates
# Smoothed values should be close to raw rates but slightly pulled toward the local spatial mean — not zeros
print("\nSanity check — raw vs smoothed lung cancer (first 10 rows):")
lung_smoothed_col = [c for c in df_final.columns if "lung_smoothed" in c]
if lung_smoothed_col:
    check_cols = ["county_fips", "year",
                  "mortality_lung_per100k", lung_smoothed_col[0]]
    print(df_final[check_cols].head(10).to_string())
    print(f"\nSmoothed lung cancer summary ({lung_smoothed_col[0]}):")
    print(df_final[lung_smoothed_col[0]].describe().round(2))
else:
    print("Smoothed lung column not found in final dataframe")

# Step 7: Save to complete_data_smooth.csv
out_path = os.path.join(OUT_DIR, "complete_data_smooth.csv")
df_final.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Shape: {df_final.shape}")
print(f"Columns ({len(df_final.columns)}):")
for col in df_final.columns:
    print(f"  {col}")