"""
download_gdp.py (OPTIONAL)
Download GDP per capita data from World Bank API.
The data file gdp_2019_clean.csv is already included in data/raw/.
Run this only if you need to re-download or update the GDP data.
"""

import json, pandas as pd, urllib.request, os, re

# Download country list from World Bank
print("Downloading World Bank country list...")
url_countries = 'https://api.worldbank.org/v2/country?format=json&per_page=300'
req = urllib.request.Request(url_countries, headers={'User-Agent': 'ResearchBot/1.0'})
countries = json.loads(urllib.request.urlopen(req, timeout=30).read())

# Build ISO2 -> ISO3 mapping
iso_map = {}
for c in countries[1]:
    iso2 = c.get('iso2Code', '')
    iso3 = c['id']
    if iso2 and len(iso2) == 2 and iso2.isalpha() and len(iso3) == 3:
        iso_map[iso2] = iso3

print(f"  Found {len(iso_map)} ISO2->ISO3 mappings")

# Extract valid country ISO2 codes
country_codes = []
for c in countries[1]:
    code = c['id']
    region = c.get('region', {}).get('value', '')
    if region != 'Aggregates' and len(code) == 3 and code.isalpha() and code.isupper():
        # Get ISO2
        iso2 = c.get('iso2Code', '')
        if iso2 and len(iso2) == 2:
            country_codes.append(iso2)

print(f"  Found {len(country_codes)} countries")

# Download GDP for all countries in batches
batch_size = 60
all_gdp = []

for i in range(0, len(country_codes), batch_size):
    batch = country_codes[i:i+batch_size]
    codes_str = ';'.join(batch)
    url = f'https://api.worldbank.org/v2/country/{codes_str}/indicator/NY.GDP.PCAP.CD?format=json&per_page=2000&date=2019'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'ResearchBot/1.0'})
        data = json.loads(urllib.request.urlopen(req, timeout=30).read())
        for r in data[1]:
            if r['value'] is not None:
                all_gdp.append({
                    'iso_code': r['country']['id'],
                    'country': r['country']['value'],
                    'gdp_per_capita_2019': float(r['value'])
                })
        print(f"  Batch {i//batch_size+1}: {len(batch)} countries, {len(all_gdp)} records so far")
    except Exception as e:
        print(f"  Batch {i//batch_size+1} failed: {e}")

# Convert ISO2 to ISO3
gdp_df = pd.DataFrame(all_gdp)
gdp_df['iso_code_3'] = gdp_df['iso_code'].map(iso_map)
gdp_clean = gdp_df.dropna(subset=['iso_code_3'])[['iso_code_3', 'gdp_per_capita_2019']].copy()
gdp_clean.rename(columns={'iso_code_3': 'iso_code'}, inplace=True)

# Save
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gdp_clean.to_csv(os.path.join(BASE, 'data', 'raw', 'gdp_2019_clean.csv'), index=False)
print(f"\nSaved {len(gdp_clean)} countries to data/raw/gdp_2019_clean.csv")
