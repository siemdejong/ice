"""
Siem de Jong
Plot summary of k_d accompanying the *critical radius* cubed.
Fitting using fit_data.py has to be performed first.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from tkinter import filedialog
from tkinter import *
import os
import matplotlib.font_manager as font_manager

plt.style.use(['science', 'scatter'])
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Palatino Linotype"],  # specify font here
    "font.size": 30})          # specify font size here

def extract_k(path):
    """Extract ice volume fraction information and fit results from csv file."""
    df = pd.DataFrame(columns=['sucrose_conc', 'IBP', 'IBP_conc', 'r_kd_opt', 'r_kd_err'])

    for csv_file_path in glob(os.path.join(path, '*[!test]', '*[!x].csv')):
        exp_df = pd.read_csv(csv_file_path, index_col='index').dropna()
        exp_name = os.path.splitext(os.path.basename(csv_file_path))[0].split('_')
        data = {
            'IBP_conc': int(exp_name[0][:-2]),
            'IBP': exp_name[1],
            'sucrose_conc': int(exp_name[2][:-1]),
            'k_opt': float(exp_df['r_kd_opt'].sample()),
            'k_err': float(exp_df['r_kd_err'].sample())
        }
        df = df.append(data, ignore_index=True)

    return df

def plot_k(df, output_plot_dir):
    """Plot the Q over time with different lines for different IBP concentration."""

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(1, 2)
    axs = gs.subplots()

    # Sort data on IBP concentration.
    df = df.sort_values('IBP_conc')

    # Filter desired data.
    df_X_0 = df.loc[df['IBP'].isin(['X'])].loc[df['IBP_conc'].eq(0)].reset_index()
    df_WT_1 = df.loc[df['IBP'].isin(['WT', 'X'])].loc[df['IBP_conc'].eq(1)].reset_index()
    df_T18N_1 = df.loc[df['IBP'].isin(['T18N', 'X'])].loc[df['IBP_conc'].eq(1)].reset_index()
    df_WT_4 = df.loc[df['IBP'].isin(['WT', 'X'])].loc[df['IBP_conc'].eq(4)].reset_index()
    df_T18N_4 = df.loc[df['IBP'].isin(['T18N', 'X'])].loc[df['IBP_conc'].eq(4)].reset_index()
    df_WT_10 = df.loc[df['IBP'].isin(['WT', 'X'])].loc[df['IBP_conc'].eq(10)].reset_index()
    df_T18N_10 = df.loc[df['IBP'].isin(['T18N', 'X'])].loc[df['IBP_conc'].eq(10)].reset_index()

    # Plot the data.
    for data, ax in zip([df_X_0, df_X_0], axs):
        ax.errorbar(data['sucrose_conc'],
                    data['k_opt']*space_scale**3*1e18*60,
                    yerr=data['k_err']*space_scale**3*1e18*60,
                    label="0uM", fmt='o', capsize=3, color='tab:blue', ms = 15)
    for data, ax in zip([df_WT_1, df_T18N_1], axs):
        ax.errorbar(data['sucrose_conc'],
                    data['k_opt']*space_scale**3*1e18*60,
                    yerr=data['k_err']*space_scale**3*1e18*60,
                    label="1uM", fmt='x', capsize=3, color='tab:green', ms = 15)        
    for data, ax in zip([df_WT_4, df_T18N_4], axs):
        ax.errorbar(data['sucrose_conc'],
                    data['k_opt']*space_scale**3*1e18*60,
                    yerr=data['k_err']*space_scale**3*1e18*60,
                    label="4uM", fmt='v', capsize=3, color='tab:red', ms = 15)
    for data, ax in zip([df_WT_10, df_T18N_10], axs):
        ax.errorbar(data['sucrose_conc'],
                    data['k_opt']*space_scale**3*1e18*60,
                    yerr=data['k_err']*space_scale**3*1e18*60,
                    label="10uM", fmt='s', capsize=3, color='tab:orange', ms = 15)

    # Settings for the axes.
    for title, ax in zip(['WT', 'T18N'], axs):
        ax.set_yscale('symlog', linthresh=1e-2) # around 0 a linear scale (because log(0)=-inf)
        ax.set_title(title)
        # ax.set_yticks(np.arange(0, 1.1, .1))
        ax.set_xticks(np.arange(10, 40, 10))
        # ax.set_yticks([-0.00001, -0.0001, -0.001, -0.01, -0.1, 0])
        ax.set_xticklabels(np.arange(10, 40, 10))
        # ax.set_yticklabels([-0.00001, -0.0001, -0.001, -0.01, -0.1, 0], fontsize=20, **pfont)
        
        ax.tick_params(axis='y', which='major', labelsize=20)
        ax.set_ylim([-0.5e-1, 100])
    axs[0].set_xlabel(r"[Sucrose] [\% w/w]")
    axs[1].set_xlabel(r"[Sucrose] [\% w/w]")
    axs[0].set_ylabel(r"$k_{d,\mathrm{cr}}$ [\textmu m$^3$ min$^{-1}$]")
    axs[1].legend(prop={'size': 20}, loc='lower right')

    # from matplotlib.pyplot import gca
    # a = gca()
    # a.set_xticklabels(a.get_xticks(), font)
    # a.set_yticklabels(a.get_yticks(), font)

    fig.savefig(os.path.join(output_plot_dir, 'kcr_summary.pdf'), bbox_inches='tight')

    plt.show()

space_scale = 86.7*10**(-9) #m

if __name__ == '__main__':
    # root = Tk() # File dialog
    # INPUT_FOLDER_NAME =  filedialog.askdirectory(title = "Select directory")
    # root.destroy()
    INPUT_FOLDER_NAME = r'E:\Ice\analysis'
    OUTPUT_FOLDER_NAME = INPUT_FOLDER_NAME

    df = extract_k(INPUT_FOLDER_NAME)
    plot_k(df, OUTPUT_FOLDER_NAME)