import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
import glob
import os
import requests

os.chdir(os.path.dirname(os.path.abspath(__file__)))
NC_DIR   = "data/acag_raw/NA_annual"
OUT_DIR  = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# Step 2: Download US county boundary shapefile from the Census Bureau
# This gives us the polygon for every county to spatially match satellite pixels
SHAPEFILE_DIR = "data/shapefiles"
os.makedirs(SHAPEFILE_DIR, exist_ok=True)
SHAPEFILE_ZIP  = os.path.join(SHAPEFILE_DIR, "us_counties.zip")
SHAPEFILE_PATH = os.path.join(SHAPEFILE_DIR, "tl_2020_us_county.shp")

if not os.path.exists(SHAPEFILE_PATH):
    print("Downloading US county shapefile from Census Bureau...")
    url = "https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/tl_2020_us_county.zip"
    response = requests.get(url, timeout=120)
    with open(SHAPEFILE_ZIP, "wb") as f:
        f.write(response.content)
    import zipfile
    with zipfile.ZipFile(SHAPEFILE_ZIP, "r") as z:
        z.extractall(SHAPEFILE_DIR)
    print("Shapefile downloaded and extracted.")
else:
    print("Shapefile already exists, skipping download.")

# Step 3: Load county boundaries and keep only the 48 contiguous states
# FIPS codes: 01-56 excluding 02 (Alaska) and 15 (Hawaii) and territories
print("\nLoading county boundaries...")
counties = gpd.read_file(SHAPEFILE_PATH)
excluded = ["02", "15", "60", "66", "69", "72", "78"]
counties = counties[~counties["STATEFP"].isin(excluded)].copy()
counties = counties.to_crs("EPSG:4326")
print(f"Counties loaded: {len(counties)}")

# Step 4: For each annual NetCDF file, extract PM2.5 values per county
# Method: convert satellite grid to points, spatial join to county polygons,
# then take the mean of all points that fall within each county boundary
all_years = []
nc_files  = sorted(glob.glob(os.path.join(NC_DIR, "*.nc")))

for nc_file in nc_files:

    # Extract year from filename: V5NA05.HybridPM25.NorthAmerica.2000001-2000364.nc
    basename = os.path.basename(nc_file)
    year     = int(basename.split(".")[3][:4])
    print(f"\nProcessing year {year}...")

    # Open the NetCDF and load the PM2.5 grid
    ds   = xr.open_dataset(nc_file)
    pm25 = ds["GWRPM25"]

    # Clip to contiguous US bounding box to reduce memory before spatial join
    # Contiguous US: lat 24-50, lon -125 to -66
    pm25_us = pm25.sel(
        lat=slice(24.0, 50.0),
        lon=slice(-125.0, -66.0)
    )

    # Convert the 2D grid to a flat table of (lat, lon, pm25_value)
    lats, lons = np.meshgrid(pm25_us.lat.values, pm25_us.lon.values, indexing="ij")
    values     = pm25_us.values

    # Remove pixels with missing or negative values before spatial join
    valid  = np.isfinite(values) & (values >= 0)
    df_pts = pd.DataFrame({
        "lat":  lats[valid],
        "lon":  lons[valid],
        "pm25": values[valid]
    })

    # Convert to GeoDataFrame so we can do a spatial join with county polygons
    gdf_pts = gpd.GeoDataFrame(
        df_pts,
        geometry=gpd.points_from_xy(df_pts["lon"], df_pts["lat"]),
        crs="EPSG:4326"
    )

    # Spatial join: assign each satellite pixel to the county it falls inside
    joined = gpd.sjoin(gdf_pts, counties[["GEOID", "geometry"]], how="inner", predicate="within")

    # Average all pixels within the same county
    county_pm25 = joined.groupby("GEOID")["pm25"].mean().reset_index()
    county_pm25.columns = ["county_fips", "pm25_satellite"]
    county_pm25["year"] = year

    all_years.append(county_pm25)
    ds.close()
    print(f"  Year {year} done: {len(county_pm25)} counties with PM2.5 values")

# Step 5: Combine all years into one table and save
print("\nCombining all years...")
df_final = pd.concat(all_years, ignore_index=True)
df_final = df_final[["county_fips", "year", "pm25_satellite"]]
df_final = df_final.sort_values(["county_fips", "year"]).reset_index(drop=True)

out_path = os.path.join(OUT_DIR, "acag_county_pm25.csv")
df_final.to_csv(out_path, index=False)

print(f"Saved: {out_path}")
print(f"Shape: {df_final.shape}")
print("\nFirst few rows:")
print(df_final.head(10))