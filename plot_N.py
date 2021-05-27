from math import exp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from tkinter import filedialog
from tkinter import *
import os
from fit_data import linear_func
from fit_data import exp_decrease_func

def calculate_N_per_A(df, A):
    """Calculate the amount of crystals per area A [um]."""
    N_per_A = df['N'] / df['ROI_area'] / space_scale**2 / 1e12 * A
    return N_per_A

def extract_N(path):
    """Extract number of crystals information and fit results from csv file."""
    dataframes = []
    for csv_file_path in glob(os.path.join(path, '*[!test]', '*[!x].csv')):
        df = pd.DataFrame(columns=['sucrose_conc', 'IBP', 'IBP_conc', 'times', 'r3', 'N0_opt', 'N0_err', 'N_t0_opt', 'N_t0_err', 'N_tau_opt', 'N_tau_err', 'N_end_opt', 'N_end_err' 'ROI_area'])

        exp_df = pd.read_csv(csv_file_path, index_col='index').dropna()
        exp_name = os.path.splitext(os.path.basename(csv_file_path))[0].split('_')
    
        df['times'] = exp_df['times']
        df['N'] = exp_df['N']
        df['N0_opt'] = exp_df['N0_opt']
        df['N0_err'] =  exp_df['N0_err']
        df['N_t0_opt'] = exp_df['N_t0_opt']
        df['N_t0_err'] = exp_df['N_t0_err']
        df['N_tau_opt'] = exp_df['N_tau_opt']
        df['N_tau_err'] = exp_df['N_tau_err']
        df['N_end_opt'] = exp_df['N_end_opt']
        df['N_end_err'] = exp_df['N_end_err']
        df['ROI_area'] = exp_df['ROI_area']
        df['IBP_conc'] = int(exp_name[0][:-2])
        df['IBP'] = exp_name[1]
        df['sucrose_conc'] = int(exp_name[2][:-1])

        dataframes.append(df)
    
    return dataframes

def plot_N_per_A(dfs, output_plot_dir, A):
    """Plot the N per area A over time with different lines for different IBP concentration."""

    def change_scale(value, A=100):
        """Change the N per area scale where A is the area in um"""
        return value / df['ROI_area'] / space_scale**2 / 1e12 * A

    # Initialize figure.
    fig = plt.figure()
    gs = fig.add_gridspec(2, 3)
    axs = gs.subplots()

    # Plot stored dataframes.
    for df in dfs:
        df['N_per_A'] = calculate_N_per_A(df, A)
        df['N_per_A_est'] = change_scale(exp_decrease_func(df.times, df.N0_opt, df.N_tau_opt, df.N_end_opt))
        if df['sucrose_conc'].iloc[0] == 10:
            if df['IBP'].iloc[0] == 'X':
                axs[0][0].scatter(df['times'], df['N_per_A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                # axs[0][0].plot(df['times'], linear_func(df['times'], df['N0_opt'], df['N0_opt']))
                # axs[0][0].plot(df.times, df['N_per_A'])
                axs[0][0].plot(df.times, df['N_per_A_est'])
                axs[1][0].scatter(df['times'], df['N_per_A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                # axs[1][0].plot(df['times'], linear_func(df['times'], df['Nt_opt'], df['N0_opt']))
                axs[1][0].plot(df['times'], df['N_per_A_est'])
            elif df['IBP'].iloc[0] == 'WT':
                axs[0][0].scatter(df['times'], df['N_per_A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                # axs[0][0].plot(df['times'], linear_func(df['times'], df['Nt_opt'], df['N0_opt']))
                axs[0][0].plot(df['times'], df['N_per_A_est'])
            elif df['IBP'].iloc[0] == 'T18N':
                axs[1][0].scatter(df['times'], df['N_per_A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                # axs[1][0].plot(df['times'], linear_func(df['times'], df['Nt_opt'], df['N0_opt']))
                axs[1][0].plot(df['times'], df['N_per_A_est'])
        elif df['sucrose_conc'].iloc[0] == 20:
            if df['IBP'].iloc[0] == 'X':
                axs[0][1].scatter(df['times'], df['N_per_A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                # axs[0][1].plot(df['times'], linear_func(df['times'], df['Nt_opt'], df['N0_opt']))
                axs[0][1].plot(df['times'], df['N_per_A_est'])
                axs[1][1].scatter(df['times'], df['N_per_A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                # axs[1][1].plot(df['times'], linear_func(df['times'], df['Nt_opt'], df['N0_opt']))
                axs[1][1].plot(df['times'], df['N_per_A_est'])
            elif df['IBP'].iloc[0] == 'WT':
                axs[0][1].scatter(df['times'], df['N_per_A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                # axs[0][1].plot(df['times'], linear_func(df['times'], df['Nt_opt'], df['N0_opt']))
                axs[0][1].plot(df['times'], df['N_per_A_est'])
            elif df['IBP'].iloc[0] == 'T18N':
                axs[1][1].scatter(df['times'], df['N_per_A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                # axs[1][1].plot(df['times'], linear_func(df['times'], df['Nt_opt'], df['N0_opt']))
                axs[1][1].plot(df['times'], df['N_per_A_est'])
        elif df['sucrose_conc'].iloc[0] == 30:
            if df['IBP'].iloc[0] == 'X':
                axs[0][2].scatter(df['times'], df['N_per_A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                # axs[0][2].plot(df['times'], linear_func(df['times'], df['Nt_opt'], df['N0_opt']))
                axs[0][2].plot(df['times'], df['N_per_A_est'])
                axs[1][2].scatter(df['times'], df['N_per_A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                # axs[1][2].plot(df['times'], linear_func(df['times'], df['Nt_opt'], df['N0_opt']))
                axs[1][2].plot(df['times'], df['N_per_A_est'])
            elif df['IBP'].iloc[0] == 'WT':
                axs[0][2].scatter(df['times'], df['N_per_A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                # axs[0][2].plot(df['times'], linear_func(df['times'], df['Nt_opt'], df['N0_opt']))
                axs[0][2].plot(df['times'], df['N_per_A_est'])
            elif df['IBP'].iloc[0] == 'T18N':
                axs[1][2].scatter(df['times'], df['N_per_A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                # axs[1][2].plot(df['times'], linear_func(df['times'], df['Nt_opt'], df['N0_opt']))
                axs[1][2].plot(df['times'], df['N_per_A_est'])
    
    # Order legend (https://stackoverflow.com/a/46160465/8797886)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0,2,3,1]
    axs[0][2].legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    
    # Place axis labels
    axs[0][0].set_ylabel(f"N per {A}$\mu$m$^2$")
    axs[1][0].set_ylabel(f"N per {A}$\mu$m$^2$")

    axs[1][0].set_xlabel("Time [s]")
    axs[1][1].set_xlabel("Time [s]")
    axs[1][2].set_xlabel("Time [s]")

    axs[0][2].yaxis.set_label_position('right')
    axs[1][2].yaxis.set_label_position('right')
    axs[0][2].set_ylabel("WT")
    axs[1][2].set_ylabel("T18N")

    axs[0][0].xaxis.set_label_position('top')
    axs[0][1].xaxis.set_label_position('top')
    axs[0][2].xaxis.set_label_position('top')
    axs[0][0].set_xlabel("10% w/w sucrose")
    axs[0][1].set_xlabel("20% w/w sucrose")
    axs[0][2].set_xlabel("30% w/w sucrose")

    # axs[0][0].set_yscale('log')

    fig.savefig(os.path.join(output_plot_dir, 'N summary'), bbox_inches='tight')
    plt.show()


space_scale = 86.7*10**(-9) #m

if __name__ == '__main__':
    # root = Tk() # File dialog
    # INPUT_FOLDER_NAME =  filedialog.askdirectory(title = "Select directory")
    # root.destroy()
    INPUT_FOLDER_NAME = r'E:\Ice\analysis'
    OUTPUT_FOLDER_NAME = INPUT_FOLDER_NAME

    df = extract_N(INPUT_FOLDER_NAME)
    plot_N_per_A(df, OUTPUT_FOLDER_NAME, 100)