import os
import boto3
import requests as req
from botocore import UNSIGNED
from botocore.config import Config

# Save relative to this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = "data/acag_raw"
os.makedirs(SAVE_DIR, exist_ok=True)

# Connect to public AWS S3 bucket (no account needed)
# Source: https://registry.opendata.aws/surface-pm2-5-v6gl/
# Bucket name: v6.gl.02.04  |  Region: us-west-2
# addressing_style path fixes SSL error caused by dots in bucket name
s3 = boto3.client(
    "s3",
    region_name="us-west-2",
    config=Config(
        signature_version=UNSIGNED,
        s3={"addressing_style": "path"}
    )
)
BUCKET = "v6.gl.02.04"

# Step 1: List top-level bucket structure
print("\nTop-level folders:")
response = s3.list_objects_v2(Bucket=BUCKET, Delimiter="/")
for prefix in response.get("CommonPrefixes", []):
    print(" ", prefix["Prefix"])

print("\nTop-level files:")
for obj in response.get("Contents", []):
    print(f"  {obj['Key']}  ({round(obj['Size']/1e6, 1)} MB)")

# Step 2: Inspect both resolution folders
for folder in ["V6.GL.02.04/", "V6.GL.02.04-0p10/"]:
    print(f"\n{folder}")
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=BUCKET, Prefix=folder):
        for obj in page.get("Contents", []):
            print(f"  {obj['Key']}  ({round(obj['Size']/1e6, 1)} MB)")
            count += 1
            if count >= 10:
                print("  first 10 shown only")
                break
        if count >= 10:
            break

# Step 3: Confirm North America files exist in the 0.1 degree folder
# NA = North America, AS = Asia
print("\nNorth America files in 0.1 degree folder")
paginator = s3.get_paginator("list_objects_v2")
count = 0
for page in paginator.paginate(
    Bucket=BUCKET, Prefix="V6.GL.02.04-0p10/NA/Annual/"
):
    for obj in page.get("Contents", []):
        print(f"  {obj['Key']}  ({round(obj['Size']/1e6, 1)} MB)")
        count += 1
        if count >= 10:
            print("  first 10 shown only")
            break
    if count >= 10:
        break
print(f"Total files previewed: {count}")

# Step 4: Download all NA annual files for 2000 to 2023
# Path-style URL puts the bucket name in the path, not the hostname
# This avoids the SSL certificate error caused by dots in the bucket name
# https://s3.amazonaws.com/v6.gl.02.04  (no SSL issue)
print("\nDownloading North America PM2.5 files 2000 to 2023")

YEARS     = range(2000, 2024)
LOCAL_DIR = os.path.join(SAVE_DIR, "NA_annual")
os.makedirs(LOCAL_DIR, exist_ok=True)

BASE_URL = (
    "https://s3.us-west-2.amazonaws.com/"
    "v6.gl.02.04/"
    "V6.GL.02.04-0p10/NA/Annual/"
)

for year in YEARS:
    filename  = f"V6GL02.04.CNNPM25.0p10.NA.{year}01-{year}12.nc"
    save_path = os.path.join(LOCAL_DIR, filename)

    if os.path.exists(save_path):
        print(f"  Already exists, skipping: {filename}")
        continue

    print(f"  Downloading {filename}", flush=True)
    response = req.get(BASE_URL + filename, stream=True, timeout=120)

    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        print(f"  Saved {round(os.path.getsize(save_path)/1e6, 1)} MB")
    else:
        print(f"  Download failed with status {response.status_code}")
        break

print("\nAll ACAG NA files downloaded.")
print(f"Saved to: {os.path.abspath(LOCAL_DIR)}")

# Note on ACAG satellite PM2.5 download attempts
# Three programmatic download methods for the ACAG V6.GL.02.04 dataset were attempted

# Attempt 1: boto3 download_file()
# Failed because download_file() internally calls HeadObject before downloading
# The bucket policy blocks HeadObject for anonymous (unsigned) users
# Error: 403 Forbidden on HeadObject operation

# Attempt 2: boto3 get_object()
# Failed because the bucket policy blocks GetObject for anonymous users entirely
# Even though list_objects_v2 works fine, actual file retrieval is restricted
# Error: AccessDenied on GetObject operation

# Attempt 3: Direct HTTPS requests with path-style URL
# Path-style was needed because the bucket name "v6.gl.02.04" contains dots
# Dots in bucket names break SSL when used as subdomains
# However the server still returned 403 Forbidden on the file request
# The bucket allows public listing but not public downloading

# Conclusion: The AWS bucket is configured as list-public but download-private
# This is a deliberate policy by the ACAG team to track usage via credentialed access
# Manual download from the website is the correct approach for this dataset

# Manual download instructions followed:
# 1. https://sites.wustl.edu/acag/datasets/surface-pm2-5/
# 2. Annual mean total PM2.5 [ug/m3] at 0.01° × 0.01°:
# 3. Downloaded [NetCDF]
# 5. Folder for manually downloaded ACAG files:cancer_dl_project/data/acag_raw/NA_annual/