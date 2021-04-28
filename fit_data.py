from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import json
from tkinter import filedialog
from tkinter import *
import os
import pandas as pd

def linear_func(x, a, b):
    return a * x + b

def horizontal_func(x, b):
    return 0 * x + b

def fitting(df):
    """Fit through the data."""
    # Ice volume fraction Q.
    Q_opt, Q_cov = curve_fit(horizontal_func, df.times, df.Q)
    Q_err = np.sqrt(np.diag(Q_cov))
    df.Q_opt = Q_opt[0]
    df.Q_err = Q_err[0]
    print(f"Q = {round(Q_opt[0], 3)} +/- {round(Q_err[0], 3)}.")

    # Mean area A.
    A_opt, A_cov = curve_fit(linear_func, df.times, df.Q)
    A_err = np.sqrt(np.diag(A_cov))
    df.At_opt = A_opt[0]
    df.A0_opt = A_opt[1]
    df.At_err = A_err[0]
    df.A0_err = A_err[1]

    print(f"Mean area = a*x + b, a={round(A_opt[0], 2)} +/- {round(A_opt[1], 2)}, b={round(A_err[0], 2)} +/- {round(A_err[1], 2)}.")

    return df

def plot(df):
    """Plot the data together with the fit."""
    fig, axs = plt.subplots(1, 2)

    # Ice volume fraction Q.
    axs[0].set_title("Q")
    Q_est = horizontal_func(df.times, df.Q_opt)
    axs[0].scatter(df.times, df.Q, s=0.8, c='k', label="data")
    axs[0].plot(df.times, Q_est, 'r', label="fit")
    axs[0].set_ylim([0, 1])

    # Mean area A
    axs[1].set_title("A")
    A_est = linear_func(df.times, df.At_opt, df.A0_opt)
    axs[1].scatter(df.times, df.A, s=0.8, c='k', label="data")
    axs[1].plot(df.times, A_est, 'r', label="fit")
    # y_err = np.sqrt((xdata * p_err[0])**2 + (p_err[0] * np.std(xdata))**2 + (p_err[1])**2) # Calculate error for linear fits.

    # plt.fill_between(xdata, y_est - y_err, y_est + y_err, alpha=0.2)

    # plt.yticks(np.linspace(0, 1, 11))
    # plt.legend()
    plt.show()

# root = Tk() # File dialog
# path = filedialog.askopenfilename(title = "Select data file")
# root.destroy()
path = r'E:\Ice\analysis\1uM_T18N_20%_0\data.csv'

df = pd.read_csv(path)
df.times = df.times * 13 / 21.52 # Correct for faster playback speed.

df = fitting(df)
plot(df)