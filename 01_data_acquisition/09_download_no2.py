import os
import requests
import zipfile

os.chdir(os.path.dirname(os.path.abspath(__file__)))

NO2_DIR = "data/no2_raw"
os.makedirs(NO2_DIR, exist_ok=True)

# Anenberg, Mohegh et al. (2022) global surface NO2 at 1km resolution
# Reference: Anenberg et al. (2022), Lancet Planetary Health
# doi: 10.1016/S2542-5196(21)00255-2
# Data hosted on figshare: https://figshare.com/articles/dataset/12968114
# NetCDF files hosted on GWU Box: https://gwu.box.com/s/4wey4183hb0nlp3dg4wl02chca3nqnl9

# Years available: 1990, 1995, 2000, and 2005-2020
# We need 2000-2019 for our 10-year pollution lag
# Missing years 2001-2004 will be noted as a limitation
AVAILABLE_YEARS = [2000] + list(range(2005, 2020))

# Step 1: Try to download via figshare API
# The figshare dataset ID is 12968114
FIGSHARE_API = "https://api.figshare.com/v2/articles/12968114"

print("Checking figshare...")
response = requests.get(FIGSHARE_API, timeout=30)

if response.status_code == 200:
    data  = response.json()
    files = data.get("files", [])
    print(f"Found {len(files)} files on figshare:")
    for f in files:
        print(f"  {f['name']}  ({round(f['size']/1e6, 1)} MB)  {f['download_url']}")
else:
    print(f"Figshare API returned {response.status_code}")
    print("Files must be downloaded manually from:")
    print("https://gwu.box.com/s/4wey4183hb0nlp3dg4wl02chca3nqnl9")

# Step 2: Download the files we need for 2000-2019 lag window
# Files are GeoTIFF format (.tif) not NetCDF
# Years 2001-2004 are not in this dataset — noted as a limitation
# Each file is global at 1km resolution, roughly 290-620 MB each

DOWNLOAD_FILES = {
    "2000_final_1km.tif": "https://ndownloader.figshare.com/files/24705152",
    "2005_final_1km.tif": "https://ndownloader.figshare.com/files/24705155",
    "2006_final_1km.tif": "https://ndownloader.figshare.com/files/24705158",
    "2007_final_1km.tif": "https://ndownloader.figshare.com/files/24705161",
    "2008_final_1km.tif": "https://ndownloader.figshare.com/files/24705164",
    "2009_final_1km.tif": "https://ndownloader.figshare.com/files/24705167",
    "2010_final_1km.tif": "https://ndownloader.figshare.com/files/24705170",
    "2011_final_1km.tif": "https://ndownloader.figshare.com/files/24705173",
    "2012_final_1km.tif": "https://ndownloader.figshare.com/files/24705176",
    "2013_final_1km.tif": "https://ndownloader.figshare.com/files/24705185",
    "2014_final_1km.tif": "https://ndownloader.figshare.com/files/24705191",
    "2015_final_1km.tif": "https://ndownloader.figshare.com/files/24705194",
    "2016_final_1km.tif": "https://ndownloader.figshare.com/files/24705197",
    "2017_final_1km.tif": "https://ndownloader.figshare.com/files/24705203",
    "2018_final_1km.tif": "https://ndownloader.figshare.com/files/26064341",
    "2019_final_1km.tif": "https://ndownloader.figshare.com/files/26064344",
}

print(f"\nFiles to download: {len(DOWNLOAD_FILES)}")
print(f"Estimated total size: ~7 GB")
print(f"Saving to: {os.path.abspath(NO2_DIR)}")
print()

for filename, url in DOWNLOAD_FILES.items():
    save_path = os.path.join(NO2_DIR, filename)

    if os.path.exists(save_path):
        size_mb = round(os.path.getsize(save_path) / 1e6, 1)
        print(f"  Already exists: {filename} ({size_mb} MB)")
        continue

    print(f"  Downloading {filename}...", flush=True)
    response = requests.get(url, stream=True, timeout=300)

    if response.status_code == 200:
        downloaded = 0
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                print(f"    {round(downloaded/1e6)} MB downloaded", end="\r", flush=True)
        size_mb = round(os.path.getsize(save_path) / 1e6, 1)
        print(f"  Saved: {filename} ({size_mb} MB)              ")
    else:
        print(f"  Failed ({response.status_code}): {filename}")

print("\nAll NO2 files downloaded.")