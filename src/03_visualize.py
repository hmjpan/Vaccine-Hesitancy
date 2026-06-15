"""
03_visualize.py
Generate publication-quality figures from the analysis results.
Run after 02_analysis.py.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from numpy.linalg import lstsq
import os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, 'data', 'processed')
FIGS = os.path.join(BASE, 'results', 'figures')
os.makedirs(FIGS, exist_ok=True)

panel = pd.read_csv(os.path.join(PROC, 'panel.csv'))
df = panel.dropna(subset=['wgm_hesitant_safe', 'max_vax_pct',
                           'dtp3_2019', 'gdp_2019']).copy()
df['log_gdp'] = np.log(df['gdp_2019'])

N = len(df)
print(f"Analysis sample: N = {N}")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ================================================================
# FIGURE 1: Bivariate associations
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

panels = [
    (axes[0], df['wgm_hesitant_safe'], df['max_vax_pct'],
     'Vaccine Hesitancy (%)', 'COVID-19 Vaccination Rate (%)',
     '(a) Unadjusted', '#e76f51'),
    (axes[1], df['log_gdp'], df['max_vax_pct'],
     'log(GDP per capita, 2019)', 'COVID-19 Vaccination Rate (%)',
     '(b) Economic Development', '#2a9d8f'),
    (axes[2], df['dtp3_2019'], df['max_vax_pct'],
     'DTP3 Coverage 2019 (%)', 'COVID-19 Vaccination Rate (%)',
     '(c) Baseline Infrastructure', '#264653'),
]

for ax, x, y, xlbl, ylbl, title, color in panels:
    r, p = pearsonr(x, y)
    ax.scatter(x, y, alpha=0.6, s=40, color=color, edgecolors='white', linewidth=0.5)
    z = np.polyfit(x, y, 1)
    xp = np.linspace(x.min() - 0.05*(x.max()-x.min()), x.max() + 0.05*(x.max()-x.min()), 100)
    ax.plot(xp, np.polyval(z, xp), 'k-', linewidth=1.5)
    ax.set_xlabel(xlbl); ax.set_ylabel(ylbl)
    ax.set_title(title, fontweight='bold')
    p_str = '<0.001' if p < 0.001 else f'p={p:.3f}'
    ax.annotate(f'r = {r:.2f}\n{p_str}\nn = {N}',
                xy=(0.05, 0.05), xycoords='axes fraction', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle('Figure 1: Bivariate Associations with COVID-19 Vaccination Uptake',
             fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig1_bivariate.png'))
fig.savefig(os.path.join(FIGS, 'fig1_bivariate.pdf'))
print("Figure 1 saved")

# ================================================================
# FIGURE 2: Standardized regression coefficients (forest plot)
# ================================================================
fig, ax = plt.subplots(figsize=(8, 3.5))

from scipy.stats import zscore
np.random.seed(42)
n_boot = 5000
y_std = zscore(df['max_vax_pct'].values)
X_std = np.column_stack([
    zscore(df['wgm_hesitant_safe']),
    zscore(df['dtp3_2019']),
    zscore(df['log_gdp']),
])
cb = []
for _ in range(n_boot):
    idx = np.random.choice(N, N, replace=True)
    try:
        c, _, _, _ = lstsq(np.column_stack([np.ones(N), X_std[idx]]), y_std[idx], rcond=None)
        cb.append(c)
    except: pass
cb = np.array(cb)
means = cb.mean(axis=0)
ci_low = np.percentile(cb, 2.5, axis=0)
ci_high = np.percentile(cb, 97.5, axis=0)

labels = ['Vaccine Hesitancy', 'DTP3 Coverage', 'log(GDP per capita)']
colors = ['#e76f51', '#264653', '#2a9d8f']
ypos = [2, 1, 0]

for i in range(3):
    ax.errorbar(means[i+1], ypos[i],
                xerr=[[means[i+1]-ci_low[i+1]], [ci_high[i+1]-means[i+1]]],
                fmt='o', color=colors[i], markersize=10, capsize=4, linewidth=2,
                markeredgecolor='black', markeredgewidth=0.5)
    ax.axhline(y=ypos[i], color='gray', linestyle=':', alpha=0.3)

ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_yticks(ypos)
ax.set_yticklabels(labels)
ax.set_xlabel('Standardized Regression Coefficient (95% CI)')
ax.set_title('Figure 2: Adjusted Predictors of Vaccination Uptake', fontweight='bold')

y_pred = np.column_stack([np.ones(N), X_std]) @ means
r2_val = 1 - np.sum((y_std - y_pred)**2) / np.sum((y_std - y_std.mean())**2)
ax.text(0.95, 0.95, f'R2 = {r2_val:.2f}\nN = {N}',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig2_coefficients.png'))
fig.savefig(os.path.join(FIGS, 'fig2_coefficients.pdf'))
print("Figure 2 saved")

# ================================================================
# FIGURE 3: Longitudinal hesitancy change
# ================================================================
ch = panel.dropna(subset=['wgm_2019_hesitant', 'wgm_2023_hesitant']).copy()
ch['delta'] = ch['wgm_2023_hesitant'] - ch['wgm_2019_hesitant']
ch = ch.sort_values('delta')
N_long = len(ch)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Lollipop chart
ax = axes[0]
ypos = range(N_long)
colors_a = ['#e76f51' if d > 0 else '#2a9d8f' for d in ch['delta']]
ax.barh(ypos, ch['delta'], color=colors_a, edgecolor='white', linewidth=0.3, height=0.7)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.axvline(x=ch['delta'].mean(), color='black', linestyle='--', linewidth=1.5,
           label=f"Mean: +{ch['delta'].mean():.1f} pp")
ax.legend(fontsize=9)
ax.set_yticks([])
ax.set_xlabel('Change in Hesitancy 2019-2023 (pp)')
ax.set_title('(a) Country-Level Changes', fontweight='bold')

# Label extreme countries
for i, (_, row) in enumerate(ch.iterrows()):
    if abs(row['delta']) > 15 or i < 3 or i > N_long - 4:
        loc = row.get('location', row['iso_code'])
        offset = 0.5 if row['delta'] > 0 else -0.5
        ax.text(row['delta'] + offset, ypos[i], loc, fontsize=6, va='center',
                ha='left' if row['delta'] > 0 else 'right')

# Panel B: Distribution
ax = axes[1]
ax.hist(ch['delta'], bins=12, color='#457b9d', edgecolor='white', alpha=0.8)
ax.axvline(x=ch['delta'].mean(), color='red', linestyle='--', linewidth=2)
ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)
ax.set_xlabel('Change in Hesitancy 2019-2023 (pp)')
ax.set_ylabel('Number of Countries')
ax.set_title('(b) Distribution', fontweight='bold')
ax.annotate(f"Mean: +{ch['delta'].mean():.1f} pp\n"
            f"Increased: {(ch['delta']>0).sum()}/{N_long}\n"
            f"Decreased: {(ch['delta']<0).sum()}/{N_long}",
            xy=(0.02, 0.95), xycoords='axes fraction', fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle('Figure 3: Global Increase in Vaccine Hesitancy (2019-2023)',
             fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig3_hesitancy_change.png'))
fig.savefig(os.path.join(FIGS, 'fig3_hesitancy_change.pdf'))
print("Figure 3 saved")

# ================================================================
# FIGURE 4: WGM longitudinal trends
# ================================================================
wgm_long = pd.read_csv(os.path.join(BASE, 'data', 'raw', 'wellcome_vaccine_safe.csv'))
wgm_long.rename(columns={'Share that agrees vaccines are safe': 'agree_safe'}, inplace=True)
wgm_long['hesitant_safe'] = 100 - wgm_long['agree_safe']

fig, ax = plt.subplots(figsize=(9, 4.5))

wgm_avg = wgm_long.groupby('Year')['hesitant_safe'].agg(['mean', 'std', 'count']).reset_index()
wgm_avg['se'] = wgm_avg['std'] / np.sqrt(wgm_avg['count'])

ax.errorbar(wgm_avg['Year'], wgm_avg['mean'], yerr=1.96*wgm_avg['se'],
            fmt='o-', color='#e76f51', linewidth=2.5, markersize=9, capsize=5,
            markeredgecolor='black', markeredgewidth=0.5, label='Global mean')

highlight = ['South Korea', 'Romania', 'Bulgaria', 'France', 'United States']
colors_h = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#457b9d']
for entity, c in zip(highlight, colors_h):
    cdata = wgm_long[wgm_long['Entity'] == entity].sort_values('Year')
    if len(cdata) >= 2:
        ax.plot(cdata['Year'], cdata['hesitant_safe'], 'o-', color=c, linewidth=1,
                markersize=5, alpha=0.8, label=entity)

ax.axvspan(2020, 2022, alpha=0.1, color='red')
ax.text(2021, 58, 'COVID-19\nPandemic', ha='center', fontsize=9, color='red', fontstyle='italic')
ax.set_xlabel('Year'); ax.set_ylabel('Vaccine Hesitancy (%)')
ax.set_title('Figure 4: Trends in Vaccine Hesitancy (Wellcome Global Monitor)', fontweight='bold')
ax.legend(loc='upper left', fontsize=8, ncol=2)
ax.set_xlim(2014, 2026); ax.set_ylim(0, 60)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig4_trends.png'))
fig.savefig(os.path.join(FIGS, 'fig4_trends.pdf'))
print("Figure 4 saved")

# ================================================================
# FIGURE 5: Game-theoretic framework (conceptual illustration)
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel A: Net benefit functions
ax = axes[0]
h_grid = np.linspace(0.01, 0.99, 500)
scenarios = [
    {'r': 0.85, 'alpha': 1.0, 'omega': 0.3, 'c_v': 0.6,
     'label': 'Low cost (rich country)', 'color': '#2a9d8f'},
    {'r': 0.75, 'alpha': 1.0, 'omega': 0.3, 'c_v': 0.8,
     'label': 'High cost (poor country)', 'color': '#e76f51'},
    {'r': 0.80, 'alpha': 1.5, 'omega': 0.2, 'c_v': 0.7,
     'label': 'Strong free-rider', 'color': '#264653'},
]

for s in scenarios:
    du = s['r'] * (1 - s['alpha'] * h_grid) - s['c_v'] + s['omega'] * h_grid**2
    ax.plot(h_grid, du, linewidth=2, color=s['color'], label=s['label'], alpha=0.8)
    crossings = np.where(np.diff(np.sign(du)))[0]
    for idx in crossings:
        ax.plot(h_grid[idx], 0, 'o', color=s['color'], markersize=8,
                markeredgecolor='black', markeredgewidth=0.5)

ax.axhline(y=0, color='black', linewidth=1)
ax.set_xlabel('Vaccination Rate (h)')
ax.set_ylabel('Net Benefit')
ax.set_title('(a) Net Benefit Functions', fontweight='bold')
ax.legend(fontsize=8, loc='lower left')
ax.set_ylim(-0.6, 0.6)

# Panel B: Equilibrium landscape
ax = axes[1]
alpha_ill = 1.0; omega_ill = 0.3
r_vals = df['wgm_agree_safe'].values / 100
h_vals = df['max_vax_pct'].values / 100
h_other = r_vals * alpha_ill / omega_ill - h_vals

ax.scatter(h_vals, h_other, alpha=0.5, s=30, color='#457b9d', edgecolors='white', linewidth=0.5)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.fill_between([0, 1], [0, 0], [1, 1], alpha=0.08, color='green')
ax.set_xlabel('Observed Uptake (h)')
ax.set_ylabel('Alternative Equilibrium')
ax.set_title('(b) Equilibrium Landscape (illustrative)', fontweight='bold')
ax.set_xlim(0, 1); ax.set_ylim(-0.5, 2.0)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
ax.axhline(y=1, color='gray', linestyle=':', alpha=0.3)
ax.annotate('Dual equilibrium\nregion', xy=(0.5, 0.2), ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax.annotate(f'(alpha/omega = {alpha_ill/omega_ill:.1f})',
            xy=(0.95, 0.05), xycoords='axes fraction', fontsize=9, ha='right')

fig.suptitle('Figure 5: Game-Theoretic Framework (Conceptual Illustration)',
             fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig5_game_theory.png'))
fig.savefig(os.path.join(FIGS, 'fig5_game_theory.pdf'))
print("Figure 5 saved")

print(f"\nAll figures saved to: {FIGS}")
print("Done.")
