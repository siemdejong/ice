"""
Siem de Jong
Plot R critical, <R> and <R>_k for comparison selected csv file.
Append x to a file to mark for exclusion.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
Polynomial = np.polynomial.Polynomial
from scipy.optimize import curve_fit

def fit_func(t, a):
    return a * t

plt.style.use(['science'])
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Palatino Linotype"],  # specify font here
    "font.size": 30})          # specify font size here

space_scale = 86.7*10**(-9) #m

df = pd.DataFrame(columns=['times', 'mean_r3_A_div_l', 'A', 'l', 'r'])
exp_df = pd.read_csv(r'E:\Ice\analysis\0uM_X_10%_0\0uM_X_10%_0.csv')

df['times'] = exp_df['times']
df['mean_r3_A_div_l'] = exp_df['mean_r3_A_div_l'] * space_scale**3 * 1e18
df['Rcr3'] = (2 * exp_df['A'] / exp_df['l'])**3 * space_scale**3 * 1e18
df['r_k3'] = exp_df['r_k3'] * space_scale**3 * 1e18

fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(df['times'], df['Rcr3'], label=r'$R_{\mathrm{cr}}^3 = \langle 2\cdot A/l\rangle^3$', color='m', marker='D')
ax.scatter(df['times'], df['mean_r3_A_div_l'], label=r'$\langle R\rangle^3 = (2\cdot\langle A\rangle /\langle l\rangle)^3$', color='c', marker='x')
ax.scatter(df['times'], df['r_k3'], label=r'$\langle R \rangle_{\kappa}^3 = 1/\langle \kappa\rangle^3$', color='y', marker='8')

ax.set_ylabel(r'$R^3$ [$\mathrm{\mu m}^3$]')
ax.set_xlabel('Time [s]')

# Fits
popt2, pcov2 = curve_fit(fit_func, df['times'], df['Rcr3'])
popt1, pcov1 = curve_fit(fit_func, df['times'], df['mean_r3_A_div_l'])
popt3, pcov3 = curve_fit(fit_func, df['times'], df['r_k3'])
ax.plot(df['times'], fit_func(df['times'], popt1[0]), color='c')
ax.plot(df['times'], fit_func(df['times'], popt2[0]), color='m')
ax.plot(df['times'], fit_func(df['times'], popt3[0]), color='y')

ax.legend()

fig.savefig(r'E:\Ice\analysis\R_comparison.pdf', bbox_inches='tight')