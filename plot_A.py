"""
Siem de Jong
Plot the time evolution of the areas of selected files.
Append x to files to mark them for exclusion.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from tkinter import filedialog
from tkinter import *
import os
from fit_data import linear_func

def extract_A(path):
    """Extract ice volume fraction information and fit results from csv file."""
    dataframes = []
    for csv_file_path in glob(os.path.join(path, '*[!test]', '*[!x].csv')):
        df = pd.DataFrame(columns=['sucrose_conc', 'IBP', 'IBP_conc', 'times', 'A', 'At_opt', 'At_err', 'A0_opt', 'A_err'])

        exp_df = pd.read_csv(csv_file_path, index_col='index').dropna()
        exp_name = os.path.splitext(os.path.basename(csv_file_path))[0].split('_')
    
        df['times'] = exp_df['times']
        df['A'] = exp_df['A']
        df['At_opt'] = exp_df['At_opt']
        df['At_err'] =  exp_df['At_err']
        df['A0_opt'] = exp_df['A0_opt']
        df['A0_err'] = exp_df['A0_err']
        df['IBP_conc'] = int(exp_name[0][:-2])
        df['IBP'] = exp_name[1]
        df['sucrose_conc'] = int(exp_name[2][:-1])

        dataframes.append(df)
    
    return dataframes

def plot_A(dfs, output_plot_dir):
    """Plot the Q over time with different lines for different IBP concentration."""
    # Initialize figure.
    fig = plt.figure()
    gs = fig.add_gridspec(2, 3)
    axs = gs.subplots()

    # Plot stored dataframes.
    for df in dfs:
        df['A'] = df['A'] * space_scale**2 * 1e12
        df['A0_opt'] = df['A0_opt'] * space_scale**2 * 1e12
        df['At_opt'] = df['At_opt'] * space_scale**2 * 1e12
        if df['sucrose_conc'].iloc[0] == 10:
            if df['IBP'].iloc[0] == 'X':
                axs[0][0].scatter(df['times'], df['A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                axs[0][0].plot(df['times'], linear_func(df['times'], df['At_opt'], df['A0_opt']))
                axs[1][0].scatter(df['times'], df['A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                axs[1][0].plot(df['times'], linear_func(df['times'], df['At_opt'], df['A0_opt']))
            elif df['IBP'].iloc[0] == 'WT':
                axs[0][0].scatter(df['times'], df['A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                axs[0][0].plot(df['times'], linear_func(df['times'], df['At_opt'], df['A0_opt']))
            elif df['IBP'].iloc[0] == 'T18N':
                axs[1][0].scatter(df['times'], df['A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                axs[1][0].plot(df['times'], linear_func(df['times'], df['At_opt'], df['A0_opt']))
        elif df['sucrose_conc'].iloc[0] == 20:
            if df['IBP'].iloc[0] == 'X':
                axs[0][1].scatter(df['times'], df['A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                axs[0][1].plot(df['times'], linear_func(df['times'], df['At_opt'], df['A0_opt']))
                axs[1][1].scatter(df['times'], df['A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                axs[1][1].plot(df['times'], linear_func(df['times'], df['At_opt'], df['A0_opt']))
            elif df['IBP'].iloc[0] == 'WT':
                axs[0][1].scatter(df['times'], df['A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                axs[0][1].plot(df['times'], linear_func(df['times'], df['At_opt'], df['A0_opt']))
            elif df['IBP'].iloc[0] == 'T18N':
                axs[1][1].scatter(df['times'], df['A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                axs[1][1].plot(df['times'], linear_func(df['times'], df['At_opt'], df['A0_opt']))
        elif df['sucrose_conc'].iloc[0] == 30:
            if df['IBP'].iloc[0] == 'X':
                axs[0][2].scatter(df['times'], df['A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                axs[0][2].plot(df['times'], linear_func(df['times'], df['At_opt'], df['A0_opt']))
                axs[1][2].scatter(df['times'], df['A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                axs[1][2].plot(df['times'], linear_func(df['times'], df['At_opt'], df['A0_opt']))
            elif df['IBP'].iloc[0] == 'WT':
                axs[0][2].scatter(df['times'], df['A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                axs[0][2].plot(df['times'], linear_func(df['times'], df['At_opt'], df['A0_opt']))
            elif df['IBP'].iloc[0] == 'T18N':
                axs[1][2].scatter(df['times'], df['A'], label=f"{df['IBP_conc'].iloc[0]} $\mu$M", s=0.5)
                axs[1][2].plot(df['times'], linear_func(df['times'], df['At_opt'], df['A0_opt']))
    
    # Order legend (https://stackoverflow.com/a/46160465/8797886)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0,2,3,1]
    axs[0][2].legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    
    # Place axis labels
    axs[0][0].set_ylabel("Area [$\mu$m$^2$]")
    axs[1][0].set_ylabel("Area [$\mu$m$^2$]")

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

    fig.savefig(os.path.join(output_plot_dir, 'area summary'), bbox_inches='tight')
    plt.show()


space_scale = 86.7*10**(-9) #m

if __name__ == '__main__':
    # root = Tk() # File dialog
    # INPUT_FOLDER_NAME =  filedialog.askdirectory(title = "Select directory")
    # root.destroy()
    INPUT_FOLDER_NAME = r'E:\Ice\analysis'
    OUTPUT_FOLDER_NAME = INPUT_FOLDER_NAME

    df = extract_A(INPUT_FOLDER_NAME)
    plot_A(df, OUTPUT_FOLDER_NAME)