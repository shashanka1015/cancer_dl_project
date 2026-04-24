import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterstats import zonal_stats

os.chdir(os.path.dirname(os.path.abspath(__file__)))

NO2_DIR = "data/no2_raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# Step 1: Inspect the 2000 file to confirm structure before processing all years
test_file = os.path.join(NO2_DIR, "2000_final_1km.tif")
print(f"Inspecting: {test_file}")

with rasterio.open(test_file) as src:
    print(f"CRS:        {src.crs}")
    print(f"Resolution: {src.res} degrees per pixel")
    print(f"Bounds:     {src.bounds}")
    print(f"NoData:     {src.nodata}")
    us_window = from_bounds(
        left=-125, bottom=24, right=-66, top=50,
        transform=src.transform
    )
    data  = src.read(1, window=us_window)
    valid = data[data != src.nodata]
    valid = valid[valid > 0]
    print(f"US mean:    {valid.mean():.4f} ppb")

# Step 2: Load county boundaries already downloaded during ACAG processing
# Exclude Alaska, Hawaii, and territories
print("\nLoading county boundaries...")
counties = gpd.read_file("data/shapefiles/tl_2020_us_county.shp")
excluded = ["02", "15", "60", "66", "69", "72", "78"]
counties = counties[~counties["STATEFP"].isin(excluded)].copy()
counties = counties.to_crs("EPSG:4326")
print(f"Counties loaded: {len(counties)}")

# Step 3: Compute county-level mean NO2 for each available year
# Years available: 2000 and 2005-2019
# Years 2001-2004 do not exist in the Anenberg/Mohegh dataset
# because the OMI satellite was not launched until July 2004
# This gap is noted as a limitation in the methods section
# For cancer years 2010-2014, the lagged NO2 mean uses available years only

YEAR_FILES = {
    2000: "2000_final_1km.tif",
    2005: "2005_final_1km.tif",
    2006: "2006_final_1km.tif",
    2007: "2007_final_1km.tif",
    2008: "2008_final_1km.tif",
    2009: "2009_final_1km.tif",
    2010: "2010_final_1km.tif",
    2011: "2011_final_1km.tif",
    2012: "2012_final_1km.tif",
    2013: "2013_final_1km.tif",
    2014: "2014_final_1km.tif",
    2015: "2015_final_1km.tif",
    2016: "2016_final_1km.tif",
    2017: "2017_final_1km.tif",
    2018: "2018_final_1km.tif",
    2019: "2019_final_1km.tif",
}

all_years = []

for year, filename in YEAR_FILES.items():
    filepath = os.path.join(NO2_DIR, filename)

    if not os.path.exists(filepath):
        print(f"  {year}: file not found, skipping")
        continue

    print(f"  Processing {year}...", end=" ", flush=True)

    with rasterio.open(filepath) as src:
        nodata    = src.nodata
        us_window = from_bounds(
            left=-125, bottom=24, right=-66, top=50,
            transform=src.transform
        )
        us_data      = src.read(1, window=us_window)
        us_transform = src.window_transform(us_window)

        # Replace nodata with NaN so rasterstats ignores those pixels
        us_data                  = us_data.astype("float32")
        us_data[us_data == nodata] = np.nan

    # zonal_stats computes mean of raster pixels within each county polygon
    # This is faster than converting pixels to points and doing a spatial join
    # all_touched=True ensures small counties still get values
    stats = zonal_stats(
        counties,
        us_data,
        affine=us_transform,
        stats=["mean"],
        nodata=np.nan,
        all_touched=True
    )

    df_year                      = counties[["GEOID"]].copy()
    df_year                      = df_year.rename(columns={"GEOID": "county_fips"})
    df_year["no2_satellite_ppb"] = [s["mean"] for s in stats]
    df_year["year"]              = year

    n_valid  = df_year["no2_satellite_ppb"].notna().sum()
    mean_val = df_year["no2_satellite_ppb"].mean()
    print(f"{n_valid} counties, mean = {mean_val:.2f} ppb")
    all_years.append(df_year)

# Step 4: Combine all years and save
print("\nCombining all years...")
df_no2 = pd.concat(all_years, ignore_index=True)
df_no2 = df_no2.sort_values(["county_fips", "year"]).reset_index(drop=True)

print(f"Combined shape: {df_no2.shape}")
print(f"Years covered:  {sorted(df_no2['year'].unique())}")
print(f"Missing values: {df_no2['no2_satellite_ppb'].isna().sum()}")

print("\nNO2 summary statistics (ppb):")
print(df_no2["no2_satellite_ppb"].describe().round(3))

out_path = os.path.join(OUT_DIR, "no2_satellite_county.csv")
df_no2.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Shape: {df_no2.shape}")

print("\nFirst few rows:")
print(df_no2.head(10).to_string())