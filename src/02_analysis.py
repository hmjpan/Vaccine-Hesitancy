"""
02_analysis.py
Empirical analysis: regression models + longitudinal hesitancy change.
Run after 01_build_panel.py.

Methods:
  - OLS regression with 5,000 bootstrap replications
  - Paired t-test and Wilcoxon signed-rank test for longitudinal change
  - All standard errors are bootstrap-based
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, ttest_1samp, wilcoxon
from numpy.linalg import lstsq
import os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, 'data', 'processed')
TABLES = os.path.join(BASE, 'results', 'tables')
os.makedirs(TABLES, exist_ok=True)

# ------------------------------------------------------------------
# 0. Load panel
# ------------------------------------------------------------------
panel = pd.read_csv(os.path.join(PROC, 'panel.csv'))

# Primary analysis sample: complete cases
df = panel.dropna(subset=['wgm_hesitant_safe', 'max_vax_pct',
                           'dtp3_2019', 'gdp_2019']).copy()

# Longitudinal sample
ch = panel.dropna(subset=['wgm_2019_hesitant', 'wgm_2023_hesitant']).copy()
ch['delta'] = ch['wgm_2023_hesitant'] - ch['wgm_2019_hesitant']

y = df['max_vax_pct'].values        # outcome: peak vaccination rate (%)
Xh = df['wgm_hesitant_safe'].values # hesitancy (%)
Xd = df['dtp3_2019'].values         # DTP3 coverage (%)
Xg = np.log(df['gdp_2019'].values)  # log(GDP)

N = len(y)
N_long = len(ch)

print("=" * 60)
print("02: EMPIRICAL ANALYSIS")
print("=" * 60)
print(f"Primary sample: N = {N}")
print(f"Longitudinal sample: N = {N_long}")

# ------------------------------------------------------------------
# 1. Bivariate correlations
# ------------------------------------------------------------------
rh, ph = pearsonr(Xh, y)
r_dtp, p_dtp = pearsonr(Xd, y)
r_gdp, p_gdp = pearsonr(Xg, y)

print(f"\n1) Bivariate correlations with vaccination uptake:")
print(f"   Hesitancy  -->  uptake: r = {rh:+.2f}, p = {ph:.4f}")
print(f"   DTP3       -->  uptake: r = {r_dtp:+.2f}, p = {p_dtp:.4f}")
print(f"   log(GDP)   -->  uptake: r = {r_gdp:+.2f}, p = {p_gdp:.4f}")

# ------------------------------------------------------------------
# 2. Fully adjusted OLS with bootstrap SEs
# ------------------------------------------------------------------
np.random.seed(42)
n_boot = 5000
X4 = np.column_stack([np.ones(N), Xh, Xd, Xg])
coeffs_boot = []
for _ in range(n_boot):
    idx = np.random.choice(N, N, replace=True)
    try:
        c, _, _, _ = lstsq(X4[idx], y[idx], rcond=None)
        coeffs_boot.append(c)
    except:
        pass
cb = np.array(coeffs_boot)
bm = cb.mean(axis=0)  # point estimates
bs = cb.std(axis=0)   # bootstrap standard errors
bp = 2 * np.minimum((cb <= 0).mean(axis=0), (cb >= 0).mean(axis=0))

y_pred = X4 @ bm
r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

print(f"\n2) Fully adjusted model (bootstrap SEs, {n_boot} reps):")
print(f"   {'Variable':<22} {'Coef':>8} {'SE':>8} {'t':>6} {'p':>8}")
print(f"   {'-'*52}")
labels = ['Intercept', 'Hesitancy (per pp)', 'DTP3 (per pp)', 'log(GDP)']
for lbl, b, s, pval in zip(labels, bm, bs, bp):
    print(f"   {lbl:<22} {b:>8.2f} {s:>8.2f} {b/max(s,1e-10):>6.1f} {pval:>8.3f}")
print(f"   R-squared = {r2:.3f}, N = {N}")

# Save regression table
reg_table = pd.DataFrame({
    'Variable': labels,
    'Coefficient': bm,
    'SE': bs,
    't': bm / np.maximum(bs, 1e-10),
    'p': bp,
})
reg_table.to_csv(os.path.join(TABLES, 'regression_results.csv'), index=False)

# ------------------------------------------------------------------
# 3. Longitudinal analysis: hesitancy change 2019 --> 2023
# ------------------------------------------------------------------
t_stat, t_p = ttest_1samp(ch['delta'], 0)
w_stat, w_p = wilcoxon(ch['wgm_2023_hesitant'], ch['wgm_2019_hesitant'])
n_inc = (ch['delta'] > 0).sum()
n_dec = (ch['delta'] < 0).sum()

print(f"\n3) Longitudinal change (2019 --> 2023):")
print(f"   Mean change:           {ch['delta'].mean():+.2f} pp")
print(f"   Median change:         {ch['delta'].median():+.2f} pp")
print(f"   95% CI:                [{ch['delta'].mean()-1.96*ch['delta'].sem():.1f}, "
      f"{ch['delta'].mean()+1.96*ch['delta'].sem():.1f}]")
print(f"   Paired t-test:         t = {t_stat:.2f}, p = {t_p:.1e}")
print(f"   Wilcoxon signed-rank:  p = {w_p:.1e}")
print(f"   Increased hesitancy:   {n_inc}/{N_long} ({n_inc/N_long*100:.0f}%)")
print(f"   Decreased hesitancy:   {n_dec}/{N_long} ({n_dec/N_long*100:.0f}%)")

# Top changes
print(f"\n   Top 5 increases:")
for _, row in ch.nlargest(5, 'delta').iterrows():
    loc = row.get('location', row['iso_code'])
    print(f"     {loc}: {row['wgm_2019_hesitant']:.1f}% -> {row['wgm_2023_hesitant']:.1f}% "
          f"(+{row['delta']:.1f} pp)")

# Save longitudinal results
long_table = pd.DataFrame({
    'Metric': ['N', 'Mean change (pp)', 'Median change (pp)',
               '95% CI lower', '95% CI upper',
               'Paired t', 't p-value',
               'Increased (%)', 'Decreased (%)'],
    'Value': [
        N_long,
        f"{ch['delta'].mean():+.2f}",
        f"{ch['delta'].median():+.2f}",
        f"{ch['delta'].mean()-1.96*ch['delta'].sem():.1f}",
        f"{ch['delta'].mean()+1.96*ch['delta'].sem():.1f}",
        f"{t_stat:.2f}",
        f"{t_p:.1e}",
        f"{n_inc} ({n_inc/N_long*100:.0f}%)",
        f"{n_dec} ({n_dec/N_long*100:.0f}%)",
    ]
})
long_table.to_csv(os.path.join(TABLES, 'longitudinal_results.csv'), index=False)

# ------------------------------------------------------------------
# 4. Descriptive statistics
# ------------------------------------------------------------------
desc_vars = {
    'vaccination_rate_pct': df['max_vax_pct'],
    'hesitancy_pct': df['wgm_hesitant_safe'],
    'dtp3_coverage_pct': df['dtp3_2019'],
    'gdp_per_capita_usd': df['gdp_2019'],
    'distrust_disagree_safe_pct': df['wgm_disagree_safe'],
}
desc = pd.DataFrame({
    'Variable': list(desc_vars.keys()),
    'N': [len(v.dropna()) for v in desc_vars.values()],
    'Mean': [v.dropna().mean() for v in desc_vars.values()],
    'SD': [v.dropna().std() for v in desc_vars.values()],
    'Min': [v.dropna().min() for v in desc_vars.values()],
    'Max': [v.dropna().max() for v in desc_vars.values()],
})
desc.to_csv(os.path.join(TABLES, 'descriptive_stats.csv'), index=False)

print(f"\nResults saved to: {TABLES}")
print("Done.")
