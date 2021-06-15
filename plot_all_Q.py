"""
Siem de Jong
Plot Q for all selected sets.
Append x to a file to mark for exclusion.
Fitting had to be done using fit_data.py for this file to work.
"""

from math import exp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from tkinter import filedialog
from tkinter import *
import os
from fit_data import jmak_func, linear_func
from fit_data import exp_decrease_func

plt.style.use(['science'])
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Palatino Linotype"],  # specify font here
    "font.size": 30})          # specify font size here

def extract_Q(path):
    """Extract number of crystals information and fit results from csv file."""
    dataframes = []
    for csv_file_path in glob(os.path.join(path, '*[!test]', '*[!x].csv')):
        df = pd.DataFrame(columns=['sucrose_conc', 'IBP', 'IBP_conc', 'times', 'Q_a_opt', 'Q_t0_opt', 'Q_tau_opt', 'ROI_area'])

        exp_df = pd.read_csv(csv_file_path, index_col='index').dropna()
        exp_name = os.path.splitext(os.path.basename(csv_file_path))[0].split('_')
    
        df['times'] = exp_df['times']
        df['Q'] = exp_df['Q']
        df['Q_opt'] = exp_df['Q_opt']
        df['Q_t0_opt'] = exp_df['Q_t0_opt']
        df['Q_tau_opt'] = exp_df['Q_tau_opt']
        df['ROI_area'] = exp_df['ROI_area']
        df['IBP_conc'] = int(exp_name[0][:-2])
        df['IBP'] = exp_name[1]
        df['sucrose_conc'] = int(exp_name[2][:-1])

        dataframes.append(df)
    
    return dataframes

def plot_all_Q(dfs, output_plot_dir):
    """Plot the N per area A over time with different lines for different IBP concentration."""

    # Initialize figure.
    fig = plt.figure(figsize=(15, 20))
    gs = fig.add_gridspec(3, 2)
    axs = gs.subplots()

    # Plot stored dataframes.
    for df in dfs:
        df['Q_opt'] = jmak_func(df.times, df.Q_opt, df.Q_t0_opt, df.Q_tau_opt, 1)
        if df['sucrose_conc'].iloc[0] == 10:
            if df['IBP'].iloc[0] == 'X':
                axs[0][0].scatter(df['times'], df['Q'], label=r"0 $\mu$M", s=15, marker='o')
                axs[0][0].plot(df['times'], df['Q_opt'])
                axs[0][1].scatter(df['times'], df['Q'], label=r"0 $\mu$M", s=15, marker='o')
                axs[0][1].plot(df['times'], df['Q_opt'])
            elif df['IBP'].iloc[0] == 'WT':
                if df['IBP_conc'].iloc[0] == 1:
                    axs[0][0].scatter(df['times'], df['Q'], label=r"1 \textmu M", s=15, marker='x')
                elif df['IBP_conc'].iloc[0] == 4:
                    axs[0][0].scatter(df['times'], df['Q'], label=r"4 \textmu M", s=15, marker='v')
                elif df['IBP_conc'].iloc[0] == 10:
                    axs[0][0].scatter(df['times'], df['Q'], label=r"10 \textmu M", s=15, marker='s')
                axs[0][0].plot(df['times'], df['Q_opt'])
            elif df['IBP'].iloc[0] == 'T18N':
                if df['IBP_conc'].iloc[0] == 1:
                    axs[0][1].scatter(df['times'], df['Q'], label=r"1 \textmu M", s=15, marker='x')
                elif df['IBP_conc'].iloc[0] == 4:
                    axs[0][1].scatter(df['times'], df['Q'], label=r"4 \textmu M", s=15, marker='v')
                elif df['IBP_conc'].iloc[0] == 10:
                    axs[0][1].scatter(df['times'], df['Q'], label=r"10 \textmu M", s=15, marker='s')
                axs[0][1].plot(df['times'], df['Q_opt'])

        elif df['sucrose_conc'].iloc[0] == 20:
            if df['IBP'].iloc[0] == 'X':
                axs[1][0].scatter(df['times'], df['Q'], label=r"0 \textmu M", s=15, marker='o')
                axs[1][0].plot(df['times'], df['Q_opt'])
                axs[1][1].scatter(df['times'], df['Q'], label=r"0 \textmu M", s=15, marker='o')
                axs[1][1].plot(df['times'], df['Q_opt'])
            elif df['IBP'].iloc[0] == 'WT':
                if df['IBP_conc'].iloc[0] == 1:
                    axs[1][0].scatter(df['times'], df['Q'], label=r"1 \textmu M", s=15, marker='x')
                elif df['IBP_conc'].iloc[0] == 4:
                    axs[1][0].scatter(df['times'], df['Q'], label=r"4 \textmu M", s=15, marker='v')
                elif df['IBP_conc'].iloc[0] == 10:
                    axs[1][0].scatter(df['times'], df['Q'], label=r"10 \textmu M", s=15, marker='s')
                axs[1][0].plot(df['times'], df['Q_opt'])
            elif df['IBP'].iloc[0] == 'T18N':
                if df['IBP_conc'].iloc[0] == 1:
                    axs[1][1].scatter(df['times'], df['Q'], label=r"1 \textmu M", s=15, marker='x')
                elif df['IBP_conc'].iloc[0] == 4:
                    axs[1][1].scatter(df['times'], df['Q'], label=r"4 \textmu M", s=15, marker='v')
                elif df['IBP_conc'].iloc[0] == 10:
                    axs[1][1].scatter(df['times'], df['Q'], label=r"10 \textmu M", s=15, marker='s')
                axs[1][1].plot(df['times'], df['Q_opt'])

        elif df['sucrose_conc'].iloc[0] == 30:
            if df['IBP'].iloc[0] == 'X':
                axs[2][0].scatter(df['times'], df['Q'], label=r"0 \textmu M", s=15, marker='o')
                axs[2][0].plot(df['times'], df['Q_opt'])
                axs[2][1].scatter(df['times'], df['Q'], label=r"0 \textmu M", s=15, marker='o')
                axs[2][1].plot(df['times'], df['Q_opt'])
            elif df['IBP'].iloc[0] == 'WT':
                if df['IBP_conc'].iloc[0] == 1:
                    axs[2][0].scatter(df['times'], df['Q'], label=r"1 \textmu M", s=15, marker='x')
                elif df['IBP_conc'].iloc[0] == 4:
                    axs[2][0].scatter(df['times'], df['Q'], label=r"4 \textmu M", s=15, marker='v')
                elif df['IBP_conc'].iloc[0] == 10:
                    axs[2][0].scatter(df['times'], df['Q'], label=r"10 \textmu M", s=15, marker='s')
                axs[2][0].plot(df['times'], df['Q_opt'])
            elif df['IBP'].iloc[0] == 'T18N':
                if df['IBP_conc'].iloc[0] == 1:
                    axs[2][1].scatter(df['times'], df['Q'], label=r"1 \textmu M", s=15, marker='x')
                elif df['IBP_conc'].iloc[0] == 4:
                    axs[2][1].scatter(df['times'], df['Q'], label=r"4 \textmu M", s=15, marker='v')
                elif df['IBP_conc'].iloc[0] == 10:
                    axs[2][1].scatter(df['times'], df['Q'], label=r"10 \textmu M", s=15, marker='s')
                axs[2][1].plot(df['times'], df['Q_opt'])
    
    # Order legend (https://stackoverflow.com/a/46160465/8797886)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0,2,3,1]
    axs[0][1].legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right', fontsize=20, markerscale=3)

    for ax in axs.flat:
        ax.set_ylim([0, 1])
    
    # Place axis labels
    axs[0][0].set_ylabel(r"Q")
    axs[1][0].set_ylabel(r"Q")
    axs[2][0].set_ylabel(r"Q")

    axs[2][0].set_xlabel("Time [s]")
    axs[2][1].set_xlabel("Time [s]")

    axs[0][1].yaxis.set_label_position('right')
    axs[1][1].yaxis.set_label_position('right')
    axs[2][1].yaxis.set_label_position('right')
    axs[0][1].set_ylabel(r"[sucrose] 10\% w/w")
    axs[1][1].set_ylabel(r"[sucrose] 20\% w/w")
    axs[2][1].set_ylabel(r"[sucrose] 30\% w/w")

    axs[0][0].xaxis.set_label_position('top')
    axs[0][1].xaxis.set_label_position('top')
    axs[0][0].set_xlabel("WT")
    axs[0][1].set_xlabel("T18N")

    # axs[0][0].set_yscale('log')

    fig.savefig(os.path.join(output_plot_dir, 'Q_all.pdf'), bbox_inches='tight')
    plt.show()


space_scale = 86.7*10**(-9) #m

if __name__ == '__main__':
    # root = Tk() # File dialog
    # INPUT_FOLDER_NAME =  filedialog.askdirectory(title = "Select directory")
    # root.destroy()
    INPUT_FOLDER_NAME = r'E:\Ice\analysis'
    OUTPUT_FOLDER_NAME = INPUT_FOLDER_NAME

    df = extract_Q(INPUT_FOLDER_NAME)
    plot_all_Q(df, OUTPUT_FOLDER_NAME)