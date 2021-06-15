"""
Siem de Jong
Plot time evolution of ice volume fraction for one set.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from tkinter import filedialog
from tkinter import *
import os
from fit_data import jmak_func

plt.style.use(['science'])
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Palatino Linotype"],  # specify font here
    "font.size": 30})          # specify font size here

def extract_Q(path):
    """Extract ice volume fraction information and fit results from csv file."""
    exp_df = pd.read_csv(path, index_col='index').dropna()
    
    return exp_df

def plot_Q(df, output_plot_dir):
    """Plot the Q over time with different lines for different IBP concentration."""
    fig = plt.figure(figsize=(100, 100))
    gs = fig.add_gridspec(1, 1)
    ax = gs.subplots()

    # for data in df:
    # label = ''.join((f"{df['IBP_conc']}", r"\,\mathrm{\mu}M"))
    ax.scatter(df['times'], df['Q'], s=100, color='black', marker='o')
    Q_est = jmak_func(df.times, df.Q_opt, df.Q_t0_opt, df.Q_tau_opt, df.Q_m_opt)
    ax.plot(df.times, Q_est, 'r', linewidth=3)

    # Settings for the axes.
    ax.set_yticks(np.arange(0, 1.1, .2))
    ax.set_yticklabels(np.around(np.arange(0, 1.1, .2),1))
    ax.set_ylabel("Q")
    ax.set_ylabel("Q")
    ax.legend()

    fig.savefig(os.path.join(output_plot_dir, os.pardir, 'Qseries.pdf'), bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    root = Tk() # File dialog
    INPUT_FILE_NAME =  filedialog.askopenfilename(title = "Select file")
    root.destroy()
    # INPUT_FOLDER_NAME = r'E:\Ice\analysis'
    OUTPUT_FILE_NAME = INPUT_FILE_NAME

    df = extract_Q(INPUT_FILE_NAME)
    plot_Q(df, OUTPUT_FILE_NAME)