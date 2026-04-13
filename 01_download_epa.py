import os
import requests
import zipfile

# Path to save files.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Configuration
YEARS = range(2000, 2024)  # 2000 to 2023 inclusive
SAVE_DIR = "data/epa_raw"
os.makedirs(SAVE_DIR, exist_ok=True)

# Two file types per year
FILE_TYPES = {
    "aqi_by_county": "annual_aqi_by_county_{year}.zip",
    "conc_by_monitor": "annual_conc_by_monitor_{year}.zip",
}

BASE_URL = "https://aqs.epa.gov/aqsweb/airdata/"

# Download loop
for year in YEARS:
    for file_key, file_template in FILE_TYPES.items():

        filename = file_template.format(year=year)
        url = BASE_URL + filename
        save_path = os.path.join(SAVE_DIR, filename)

        # Skip if already downloaded
        if os.path.exists(save_path):
            print(f"  Already exists, skipping: {filename}")
            continue

        print(f"Downloading {filename}...")
        response = requests.get(url, timeout=60)

        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"  Saved {filename}")
        else:
            # Some early years may not have all file types
            print(f"  Not found ({response.status_code}): {filename}")

# Unzip all downloaded files
for fname in os.listdir(SAVE_DIR):
    if fname.endswith(".zip"):
        zip_path = os.path.join(SAVE_DIR, fname)
        extract_dir = os.path.join(SAVE_DIR, fname.replace(".zip", ""))

        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)
            print(f"Unzipped: {fname}")

print("\nAll EPA files downloaded and unzipped.")