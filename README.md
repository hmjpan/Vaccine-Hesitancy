# Vaccine Hesitancy and COVID-19 Vaccination Uptake

Cross-country analysis combining Wellcome Global Monitor, OWID, WHO/UNICEF, and World Bank data to examine whether self-reported vaccine hesitancy predicts COVID-19 vaccination uptake.

## Data Sources

All data are publicly available:

| Source | File | Description |
|--------|------|-------------|
| Wellcome Global Monitor (via OWID) | `data/raw/wellcome_vaccine_safe.csv` | % agreeing vaccines are safe (119 countries) |
| OWID COVID-19 Vaccinations | `data/raw/owid_vaccinations.csv` | Daily vaccination time series (204 countries) |
| WHO/UNICEF WUENIC | `data/raw/dtp3_coverage.csv` | DTP3 coverage 2019 (218 countries) |
| World Bank API | `data/raw/gdp_2019_clean.csv` | GDP per capita 2019 (94 countries) |

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

Run scripts in order:

```bash
# 1. Build cross-country panel dataset
python src/01_build_panel.py

# 2. Run empirical analysis (regression + longitudinal)
python src/02_analysis.py

# 3. Generate figures
python src/03_visualize.py
```

## Output

- `results/tables/` - Regression results, descriptive statistics, longitudinal analysis
- `results/figures/` - Publication-quality figures (PNG + PDF)

## Key Findings

- Vaccine hesitancy alone explains only 7% of cross-country variation in COVID-19 vaccination uptake (r = -0.26, p = 0.096, N = 42)
- After controlling for GDP, hesitancy becomes a significant predictor (beta = -0.83, p = 0.001, R2 = 0.62)
- GDP per capita is the dominant predictor of uptake
- Hesitancy increased in 93% of countries from 2019 to 2023 (mean +12.5 pp, p < 0.001)

## Requirements

- Python >= 3.9
- Dependencies: pandas, numpy, scipy, matplotlib

## License

MIT
