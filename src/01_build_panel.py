"""
01_build_panel.py
Build cross-country panel dataset from public data sources.
Run this script first to generate data/processed/panel.csv.

Data sources:
  - Wellcome Global Monitor (vaccine attitudes, 119 countries)
  - OWID COVID-19 vaccinations (uptake time series, 204 countries)
  - WHO/UNICEF DTP3 coverage (baseline infrastructure, 218 countries)
  - World Bank GDP per capita (2019, 94 countries)
"""

import pandas as pd
import numpy as np
import os

# Paths relative to this script's location
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, 'data', 'raw')
PROC = os.path.join(BASE, 'data', 'processed')
os.makedirs(PROC, exist_ok=True)

# Aggregated regions to exclude
AGGREGATES = [
    'OWID_AFR', 'OWID_ASI', 'OWID_EUR', 'OWID_EUN', 'OWID_INT',
    'OWID_KOS', 'OWID_NAM', 'OWID_OCE', 'OWID_SAM', 'OWID_WRL',
    'OWID_LIC', 'OWID_LMC', 'OWID_UMC', 'OWID_HIC', 'OWID_CYN'
]

print("=" * 60)
print("01: BUILDING PANEL DATASET")
print("=" * 60)

# ------------------------------------------------------------------
# 1. COVID-19 vaccination data (primary outcome)
# ------------------------------------------------------------------
vax = pd.read_csv(
    os.path.join(RAW, 'owid_vaccinations.csv'),
    parse_dates=['date'], low_memory=False
)
vax = vax[~vax['iso_code'].isin(AGGREGATES)]

# Peak uptake per country
vax_max = vax.groupby('iso_code').agg(
    location=('location', 'first'),
    max_vax_pct=('people_vaccinated_per_hundred', 'max'),
    max_fully_pct=('people_fully_vaccinated_per_hundred', 'max'),
    n_vax_days=('date', 'count'),
).reset_index()
vax_max = vax_max[vax_max['n_vax_days'] > 30]
print(f"  Vaccination data: {len(vax_max)} countries with >30 days of data")

# ------------------------------------------------------------------
# 2. Wellcome Global Monitor (vaccine attitudes)
# ------------------------------------------------------------------
wgm_safe   = pd.read_csv(os.path.join(RAW, 'wellcome_vaccine_safe.csv'))
wgm_imp    = pd.read_csv(os.path.join(RAW, 'wellcome_vaccine_important.csv'))
wgm_dsafe  = pd.read_csv(os.path.join(RAW, 'wellcome_disagree_safe.csv'))
wgm_deff   = pd.read_csv(os.path.join(RAW, 'wellcome_disagree_effective.csv'))

wgm_safe.rename(columns={'Share that agrees vaccines are safe': 'agree_safe'}, inplace=True)
wgm_imp.rename(columns={'Share that agrees vaccines are important for children': 'agree_important'}, inplace=True)
wgm_dsafe.rename(columns={'Share that disagrees vaccines are safe': 'disagree_safe'}, inplace=True)
wgm_deff.rename(columns={'Share that disagree that vaccines are effective': 'disagree_effective'}, inplace=True)

wgm = wgm_safe[['Entity', 'Code', 'Year', 'agree_safe']].merge(
    wgm_imp[['Entity', 'Code', 'Year', 'agree_important']],
    on=['Entity', 'Code', 'Year'], how='outer'
)
wgm = wgm.merge(wgm_dsafe[['Entity', 'Code', 'Year', 'disagree_safe']],
                on=['Entity', 'Code', 'Year'], how='outer')
wgm = wgm.merge(wgm_deff[['Entity', 'Code', 'Year', 'disagree_effective']],
                on=['Entity', 'Code', 'Year'], how='outer')

wgm['hesitant_safe']       = 100 - wgm['agree_safe']
wgm['hesitant_important']  = 100 - wgm['agree_important']

# Latest available WGM data per country
wgm_latest = wgm.sort_values('Year', ascending=False).groupby('Entity').first().reset_index()
wgm_latest.rename(columns={
    'agree_safe': 'wgm_agree_safe',
    'agree_important': 'wgm_agree_important',
    'hesitant_safe': 'wgm_hesitant_safe',
    'hesitant_important': 'wgm_hesitant_important',
    'disagree_safe': 'wgm_disagree_safe',
    'disagree_effective': 'wgm_disagree_effective',
    'Year': 'wgm_year'
}, inplace=True)
wgm_latest = wgm_latest[['Code', 'Entity', 'wgm_year',
                          'wgm_agree_safe', 'wgm_agree_important',
                          'wgm_hesitant_safe', 'wgm_hesitant_important',
                          'wgm_disagree_safe', 'wgm_disagree_effective']]

# Pre-pandemic (2019) and post-pandemic (2023) subsets
wgm_2019 = wgm[wgm['Year'] == 2019][['Code', 'agree_safe', 'hesitant_safe']].copy()
wgm_2019.rename(columns={'agree_safe': 'wgm_2019_agree', 'hesitant_safe': 'wgm_2019_hesitant'}, inplace=True)

wgm_2023 = wgm[wgm['Year'] == 2023][['Code', 'agree_safe', 'hesitant_safe']].copy()
wgm_2023.rename(columns={'agree_safe': 'wgm_2023_agree', 'hesitant_safe': 'wgm_2023_hesitant'}, inplace=True)

print(f"  Wellcome G.M.: {wgm['Entity'].nunique()} countries, {len(wgm)} country-years")

# ------------------------------------------------------------------
# 3. DTP3 coverage (baseline vaccination infrastructure)
# ------------------------------------------------------------------
dtp = pd.read_csv(os.path.join(RAW, 'dtp3_coverage.csv'))
dtp.rename(columns={'Diphtheria/tetanus/pertussis (DTP3)': 'dtp3'}, inplace=True)
dtp_2019 = dtp[dtp['Year'] == 2019][['Entity', 'Code', 'dtp3']].copy()
dtp_2019.rename(columns={'dtp3': 'dtp3_2019'}, inplace=True)
print(f"  DTP3 coverage: {len(dtp_2019)} countries (2019)")

# ------------------------------------------------------------------
# 4. GDP per capita (World Bank)
# ------------------------------------------------------------------
gdp = pd.read_csv(os.path.join(RAW, 'gdp_2019_clean.csv'))
gdp.rename(columns={'gdp_per_capita_2019': 'gdp_2019'}, inplace=True)
print(f"  GDP data: {len(gdp)} countries")

# ------------------------------------------------------------------
# 5. Merge into country-level panel
# ------------------------------------------------------------------
panel = vax_max[['iso_code', 'location', 'max_vax_pct', 'max_fully_pct', 'n_vax_days']].copy()

panel = panel.merge(wgm_latest, left_on='iso_code', right_on='Code', how='left', suffixes=('', '_wgm'))
panel = panel.merge(wgm_2019,  left_on='iso_code', right_on='Code', how='left', suffixes=('', '_2019'))
panel = panel.merge(wgm_2023,  left_on='iso_code', right_on='Code', how='left', suffixes=('', '_2023'))
panel = panel.merge(dtp_2019[['Code', 'dtp3_2019']], left_on='iso_code', right_on='Code', how='left', suffixes=('', '_dtp'))
panel = panel.merge(gdp[['iso_code', 'gdp_2019']], on='iso_code', how='left')

# Derived variables
panel['hesitancy_change'] = panel['wgm_2023_hesitant'] - panel['wgm_2019_hesitant']
panel['log_gdp'] = np.log(panel['gdp_2019'])

# Remove duplicate columns
panel = panel.loc[:, ~panel.columns.duplicated()]

# Save
panel_final = panel[['iso_code', 'location',
                      'max_vax_pct', 'max_fully_pct', 'n_vax_days',
                      'wgm_year', 'wgm_agree_safe', 'wgm_hesitant_safe',
                      'wgm_disagree_safe', 'wgm_disagree_effective',
                      'wgm_2019_agree', 'wgm_2019_hesitant',
                      'wgm_2023_agree', 'wgm_2023_hesitant',
                      'dtp3_2019', 'gdp_2019', 'log_gdp',
                      'hesitancy_change']]
panel_final.to_csv(os.path.join(PROC, 'panel.csv'), index=False)

print(f"\n  Final panel: {len(panel_final)} countries")
print(f"  With full data (hesitancy + vax + DTP3 + GDP): "
      f"{panel_final.dropna(subset=['wgm_hesitant_safe','max_vax_pct','dtp3_2019','gdp_2019']).shape[0]}")
print(f"  Saved to: {os.path.join(PROC, 'panel.csv')}")
print("Done.")
