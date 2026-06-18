import os
import requests
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

CHR_DIR = "data/chr_raw"
os.makedirs(CHR_DIR, exist_ok=True)

# County Health Rankings analytic CSV files 2010 to 2020
# These contain county-level socioeconomic and health behavior measures
# URL pattern changes slightly for some years as shown below
# Source: https://www.countyhealthrankings.org/health-data/methodology-and-sources/data-documentation/national-data-documentation-2010-2023

CHR_URLS = {
    2010: "https://www.countyhealthrankings.org/sites/default/files/analytic_data2010.csv",
    2011: "https://www.countyhealthrankings.org/sites/default/files/analytic_data2011.csv",
    2012: "https://www.countyhealthrankings.org/sites/default/files/analytic_data2012.csv",
    2013: "https://www.countyhealthrankings.org/sites/default/files/analytic_data2013.csv",
    2014: "https://www.countyhealthrankings.org/sites/default/files/analytic_data2014.csv",
    2015: "https://www.countyhealthrankings.org/sites/default/files/analytic_data2015.csv",
    2016: "https://www.countyhealthrankings.org/sites/default/files/analytic_data2016.csv",
    2017: "https://www.countyhealthrankings.org/sites/default/files/analytic_data2017.csv",
    2018: "https://www.countyhealthrankings.org/sites/default/files/analytic_data2018_0.csv",
    2019: "https://www.countyhealthrankings.org/sites/default/files/analytic_data2019.csv",
    2020: "https://www.countyhealthrankings.org/sites/default/files/media/document/analytic_data2020_0.csv",
}

# Step 1: Download all CHR files
print("Downloading County Health Rankings analytic files 2010 to 2020...")

for year, url in CHR_URLS.items():
    save_path = os.path.join(CHR_DIR, f"chr_analytic_{year}.csv")

    if os.path.exists(save_path):
        print(f"  {year}: already exists, skipping")
        continue

    print(f"  {year}: downloading...", end=" ", flush=True)
    response = requests.get(url, timeout=60)

    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        size_kb = round(os.path.getsize(save_path) / 1024, 1)
        print(f"saved ({size_kb} KB)")
    else:
        print(f"failed (status {response.status_code})")

# Step 2: Inspect one file to understand the column structure
# CHR analytic CSVs have a two-row header:
# Row 1: variable codes (e.g. v009_rawvalue)
# Row 2: variable labels (e.g. "Adult smoking")
# Actual data starts at row 3
# Read the first two rows separately to build a code-to-label mapping

print("\nInspecting 2020 file structure...")
sample_file = os.path.join(CHR_DIR, "chr_analytic_2020.csv")

if os.path.exists(sample_file):
    codes  = pd.read_csv(sample_file, nrows=1, header=None)
    labels = pd.read_csv(sample_file, nrows=2, header=None).iloc[1]

    print("\nFirst 30 columns (code -> label):")
    for i, (code, label) in enumerate(zip(codes.iloc[0], labels)):
        if i >= 30:
            break
        print(f"  {str(code):35s}: {label}")

    print("\nAll columns containing 'smoke', 'poverty', 'unemploy', 'obes', 'insur', 'income':")
    for code, label in zip(codes.iloc[0], labels):
        label_str = str(label).lower()
        if any(kw in label_str for kw in
               ["smoke", "poverty", "unemploy", "obes", "insur", "income",
                "education", "rural", "physician", "65"]):
            print(f"  {str(code):35s}: {label}")

# Step 3: Corrected inspection - the two rows are swapped
# Row 1 (index 0) = descriptive labels like "Adult smoking raw value"
# Row 2 (index 1) = short codes like "v009_rawvalue"
# Search the descriptive labels

print("\nCorrected column search:")
header_df = pd.read_csv(sample_file, nrows=2, header=None)

descriptive = header_df.iloc[0]  # Full descriptions
short_codes = header_df.iloc[1]  # Short variable codes

keywords = ["smoke", "poverty", "unemploy", "obes", "insur",
            "income", "educat", "rural", "physician", "age 65",
            "hispanic", "black", "white", "population"]

for desc, code in zip(descriptive, short_codes):
    desc_str = str(desc).lower()
    if any(kw in desc_str for kw in keywords):
        print(f"  {str(code):35s}: {desc}")