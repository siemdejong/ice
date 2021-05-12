import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from collections import namedtuple
from tkinter import filedialog
from tkinter import *
import os

# def make_df():
#     """Make the dataframe."""
#     for csv_file_path in glob(os.path.join(INPUT_FOLDER_NAME, '*[!test]', '*.csv')):
#         df = pd.read_csv(csv_file_path, index_col='index').dropna()
#         experiment = os.path.splitext(os.path.basename(csv_file_path))[0].split('_')
#         IBP_conc = experiment[0][:-2]
#         IBP = experiment[1]
#         sucrose_conc = experiment[2][:-1]

def extract_Q(df):
    """Extract ice volume fraction information and fit results from csv file."""
    for csv_file_path in glob(os.path.join(INPUT_FOLDER_NAME, '*[!test]', '*.csv')):
        exp_df = pd.read_csv(csv_file_path, index_col='index').dropna()
        exp_name = os.path.splitext(os.path.basename(csv_file_path))[0].split('_')
        data = {
            'IBP_conc': int(exp_name[0][:-2]),
            'IBP': exp_name[1],
            'sucrose_conc': int(exp_name[2][:-1]),
            'Q_opt': float(exp_df['Q_opt'].sample()),
            'Q_err': float(exp_df['Q_err'].sample())
        }
        df = df.append(data, ignore_index=True)
    
    return df

def plot_Q(df):
    """Plot the Q over time with different lines for different IBP concentration."""
    fig = plt.figure()
    gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
    axs = gs.subplots(sharey=True)

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
        ax.plot(data['sucrose_conc'], data['Q_opt'], label="0uM")
    for data, ax in zip([df_WT_1, df_T18N_1], axs):
        ax.plot(data['sucrose_conc'], data['Q_opt'], label="1uM")
    for data, ax in zip([df_WT_4, df_T18N_4], axs):
        ax.plot(data['sucrose_conc'], data['Q_opt'], label="4uM")
    for data, ax in zip([df_WT_10, df_T18N_10], axs):
        ax.plot(data['sucrose_conc'], data['Q_opt'], label="10uM")

    # Settings for the axes.
    for title, ax in zip(['WT', 'T18N'], axs):
        ax.set_title(title)
        ax.legend()
        ax.set_yticks(np.arange(0, 1.1, .1))
        ax.set_xticks(np.arange(10, 50, 10))

    plt.show()


if __name__ == '__main__':
    # root = Tk() # File dialog
    # INPUT_FOLDER_NAME =  filedialog.askdirectory(title = "Select directory")
    # root.destroy()
    INPUT_FOLDER_NAME = r'E:\Ice\analysis'

    df = pd.DataFrame(columns=['sucrose_conc', 'IBP', 'IBP_conc', 'Q_opt', 'Q_err'])
    df = extract_Q(df)
    plot_Q(df)