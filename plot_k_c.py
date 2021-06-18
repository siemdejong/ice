"""
Siem de Jong
Plot k(c) for 30% sucrose, rQAE WT and T18N.
Append x to a file to mark for exclusion.
Fitting had to be done using fit_data.py for this file to work.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
from glob import glob
from tkinter import filedialog
from tkinter import *
import os
import matplotlib.font_manager as font_manager
from scipy.optimize.minpack import curve_fit

plt.style.use(['science'])
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Palatino Linotype"],  # specify font here
    "font.size": 30})          # specify font size here

def extract_Q(path):
    """Extract ice volume fraction information and fit results from csv file."""
    df = pd.DataFrame(columns=['sucrose_conc', 'IBP', 'IBP_conc', 'k_opt', 'k_err'])

    for csv_file_path in glob(os.path.join(path, '*[!test]', '*[!x].csv')):
        exp_df = pd.read_csv(csv_file_path, index_col='index').dropna()
        exp_name = os.path.splitext(os.path.basename(csv_file_path))[0].split('_')
        data = {
            'IBP_conc': int(exp_name[0][:-2]),
            'IBP': exp_name[1],
            'sucrose_conc': int(exp_name[2][:-1]),
            'k_opt': float(exp_df['r_kd_A_div_l_opt'].sample()),
            'k_err': float(exp_df['r_kd_A_div_l_err'].sample())
        }
        df = df.append(data, ignore_index=True)
    
    return df

def sigmoidal_func(c, kd0, ci, s):
    return kd0 - kd0 / (1 + np.exp((ci - c) / s))

def plot_Q(df, output_plot_dir):
    """Plot the Q over time with different lines for different IBP concentration."""
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(1, 2)
    axs = gs.subplots()

    df = df.sort_values('IBP_conc').loc[df['sucrose_conc'].eq(30)]

    xdata1, ydata1 = [], []
    xdata2, ydata2 = [], []
    for _, data in df.iterrows():
        if data['IBP'] == 'X':
            xdata1.append(data['IBP_conc'])
            xdata2.append(data['IBP_conc'])
            ydata1.append(data['k_opt']*space_scale**3*1e18 * 60)
            ydata2.append(data['k_opt']*space_scale**3*1e18 * 60)
            for ax in axs:
                ax.errorbar(data['IBP_conc'], data['k_opt']*space_scale**3*1e18 * 60, yerr=data['k_err']*space_scale**3*1e18 * 60, color='k', marker='x', ms=15, capsize=3)
        elif data['IBP'] == 'WT':
            axs[0].errorbar(data['IBP_conc'], data['k_opt']*space_scale**3*1e18 * 60, yerr=data['k_err']*space_scale**3*1e18 * 60, color='k', marker='x', ms=15, capsize=3)
            xdata1.append(data['IBP_conc'])
            ydata1.append(data['k_opt']*space_scale**3*1e18 * 60)
        elif data['IBP'] == 'T18N':
            axs[1].errorbar(data['IBP_conc'], data['k_opt']*space_scale**3*1e18 * 60, yerr=data['k_err']*space_scale**3*1e18 * 60, color='k', marker='x', ms=15, capsize=3)
            xdata2.append(data['IBP_conc']*space_scale**3*1e18 * 60)
            ydata2.append(data['k_opt']*space_scale**3*1e18 * 60)
    popt1, pcov1 = curve_fit(sigmoidal_func, xdata1, ydata1, p0=(3, 1.5, 1), bounds=((-np.inf, -np.inf, 0.1), (np.inf, np.inf, np.inf)), maxfev=100000)
    popt2, pcov2 = curve_fit(sigmoidal_func, xdata2, ydata2, p0=(3, 1.5, 1), bounds=((-np.inf, -np.inf, 0.1), (np.inf, np.inf, np.inf)), maxfev=100000)

    # Plot the fits.
    # axs[0].plot(xdata1, sigmoidal_func(xdata1, *popt1), 'r')
    # axs[1].plot(xdata2, sigmoidal_func(xdata2, *popt2), 'r')

    # Plot the fits using values calculated by origin
    # xdata1 = np.linspace(xdata1[0], xdata1[-1], 1000)
    # xdata2 = np.linspace(xdata2[0], xdata2[-1], 1000)
    xs = np.linspace(0, 10, 10000)
    axs[0].plot(xs, sigmoidal_func(xs, 2.96565, 0.59325, 0.00264), 'r')
    # axs[1].plot(xdata1, sigmoidal_func(np.array(xdata1), 2.96565, 0.59325, 0.00264), 'r')
    axs[1].plot(xs, sigmoidal_func(xs, 2.96565, 0.59325, 0.00264), 'r')

    axs[0].plot(0.59325, sigmoidal_func(0.59325, 2.96565, 0.59325, 0.00264), ms=10, marker='o', markerfacecolor='none', markeredgewidth=2, markeredgecolor='red')
    axs[1].plot(0.59325, sigmoidal_func(0.59325, 2.96565, 0.59325, 0.00264), ms=10, marker='o', markerfacecolor='none', markeredgewidth=2, markeredgecolor='red')

    axs[0].set_xlabel(r"[rQAE WT] [\textmu M]")#, fontsize=30)
    axs[1].set_xlabel(r"[rQAE T18N] [\textmu M]")#, fontsize=30)
    axs[0].set_ylabel(r"$k_d$ [\textmu m$^3 \mathrm{min}^{-1}$]")#, fontsize=30)
    # axs[1].set_ylabel(r"$k_d$ [\textmu m$^3 \mathrm{min}^{-1}$]")#, fontsize=30)
    # axs[0].legend()#fontsize=20)

    fig.savefig(os.path.join(output_plot_dir, 'k_c.pdf'), bbox_inches='tight')

    plt.show()

space_scale = 86.7*10**(-9) #m
if __name__ == '__main__':
    # root = Tk() # File dialog
    # INPUT_FOLDER_NAME =  filedialog.askdirectory(title = "Select directory")
    # root.destroy()
    INPUT_FOLDER_NAME = r'E:\Ice\analysis'
    OUTPUT_FOLDER_NAME = INPUT_FOLDER_NAME

    df = extract_Q(INPUT_FOLDER_NAME)
    plot_Q(df, OUTPUT_FOLDER_NAME)