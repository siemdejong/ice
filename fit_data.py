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

def jmak_func(t, a, t0, tau, m):
    return a * (1 - np.exp(-((t - 0) / tau)**m))

def exp_decrease_func(t, N0, t0, tau, N_end):
    return N0 * np.exp(-(t - t0) / tau) + N_end

def fitting(df, path):
    """Fit through the data."""
    # Ice volume fraction Q.
    # Q_opt, Q_cov = curve_fit(horizontal_func, df.times, df.Q)
    Q_opt, Q_cov = curve_fit(jmak_func, df.times, df.Q, [0.5, 0, 100, 2], bounds=([0, -np.inf, -np.inf, 2], [1, df.times.iat[-1], df.times.iat[-1], 3]), maxfev=1000000)
    Q_err = np.sqrt(np.diag(Q_cov))
    df['Q_opt'] = Q_opt[0]
    df['Q_err'] = Q_err[0]
    df['Q_t0_opt'] = Q_opt[1]
    df['Q_t0_err'] = Q_err[1]
    df['Q_tau_opt'] = Q_opt[2]
    df['Q_tau_err'] = Q_err[2]
    df['Q_m_opt'] = Q_opt[2]
    df['Q_m_err'] = Q_err[2]
    print(f"Q = {round(Q_opt[0], 3)} +/- {round(Q_err[0], 3)}.")
    print(f"t0 = {round(Q_opt[1], 3)} +/- {round(Q_err[1], 3)}.")
    print(f"tau = {round(Q_opt[2], 3)} +/- {round(Q_err[2], 3)}.")
    print(f"m = {round(Q_opt[3], 3)} +/- {round(Q_err[3], 3)}.")
    print("-------------")

    # Mean area A.
    A_opt, A_cov = curve_fit(linear_func, df.times, df.A)
    A_err = np.sqrt(np.diag(A_cov))
    df['At_opt'], df['A0_opt'] = A_opt
    df['At_err'], df['A0_err'] = A_err
    print(f"Mean area = a*x + b, a={round(A_opt[0], 2)} +/- {round(A_err[0], 2)}, b={round(A_opt[1], 2)} +/- {round(A_err[1], 2)}.")

    # Mean radius of curvature <r>^3.
    r3_opt, r3_cov = curve_fit(linear_func, df.times, df.r3)
    r3_err = np.sqrt(np.diag(r3_cov))
    df['r3t_opt'], df['r30_opt'] = r3_opt
    df['r3t_err'], df['r30_err'] = r3_err
    print(f"<r>^3 = a*x + b, a={round(r3_opt[0], 2)} +/- {round(r3_err[0], 2)}, b={round(r3_opt[1], 2)} +/- {round(r3_err[1], 2)}.")

    # Number of crystals N.
    # N_opt, N_cov = curve_fit(linear_func, df.times, df.N)
    N_opt, N_cov = curve_fit(exp_decrease_func, df.times, df.N, [60, 0, 100, 10], bounds=([0, -np.inf, -np.inf, 0], [100, np.inf, np.inf, 100]), maxfev = 100000)
    N_err = np.sqrt(np.diag(N_cov))
    df['N0_opt'], df['N_t0_opt'], df['N_tau_opt'], df['N_end_opt'] = N_opt
    df['N0_err'], df['N_t0_err'], df['N_tau_err'], df['N_end_err'] = N_err
    # print(f"N = a*x + b, a={round(N_opt[0], 2)} +/- {round(N_err[0], 2)}, b={round(N_opt[1], 2)} +/- {round(N_err[1], 2)}.")
    print("N:")
    print(f"N0 = {round(N_opt[0], 3)} +/- {round(N_err[0], 3)}.")
    print(f"t0 = {round(N_opt[1], 3)} +/- {round(N_err[1], 3)}.")
    print(f"tau = {round(N_opt[2], 3)} +/- {round(N_err[2], 3)}.")
    print(f"N_end = {round(N_opt[3], 3)} +/- {round(N_err[3], 3)}.")


    df.to_csv(path, index_label='index')

    return df

def plot(df, path, show=False):
    """Plot the data together with the fit."""
    fig, axs = plt.subplots(2, 2)

    fig.suptitle(f'{os.path.splitext(os.path.basename(path))[0]}')

    # Ice volume fraction Q.
    axs[0][0].set_title("Q")
    # Q_est = horizontal_func(df.times, df.Q_opt)
    Q_est = jmak_func(df.times, df.Q_opt, df.Q_t0_opt, df.Q_tau_opt, df.Q_m_opt)
    axs[0][0].scatter(df.times, df.Q, s=0.8, c='k', label="data")    
    axs[0][0].plot(df.times, Q_est, 'r', label="fit")
    axs[0][0].set_ylim([0, 1])
    # axs[0][0].set_xscale('log')

    # Mean area A
    axs[0][1].set_title("A")
    A_est = linear_func(df.times, df.At_opt, df.A0_opt)
    axs[0][1].scatter(df.times, df.A, s=0.8, c='k', label="data")
    axs[0][1].plot(df.times, A_est, 'r', label="fit")

    # Mean radius of curvature <r>^3.
    axs[1][0].set_title("<r>^3")
    r3_est = linear_func(df.times, df.r3t_opt, df.r30_opt)
    axs[1][0].scatter(df.times, df.r3, s=0.8, c='k', label="data")
    axs[1][0].plot(df.times, r3_est, 'r', label="fit")

    # Number of crystals N.
    axs[1][1].set_title("N")
    # N_est = linear_func(df.times, df.Nt_opt, df.N0_opt)
    N_est = exp_decrease_func(df.times, df.N0_opt, df.N_t0_opt, df.N_tau_opt, df.N_end_opt)
    axs[1][1].scatter(df.times, df.N, s=0.8, c='k', label="data")
    axs[1][1].plot(df.times, N_est, 'r', label="fit")
    # y_err = np.sqrt((xdata * p_err[0])**2 + (p_err[0] * np.std(xdata))**2 + (p_err[1])**2) # Calculate error for linear fits.

    # plt.fill_between(xdata, y_est - y_err, y_est + y_err, alpha=0.2)

    # plt.yticks(np.linspace(0, 1, 11))
    # plt.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(path), 'fits.png'))

    if show:
        plt.show()

if __name__ == '__main__':
    root = Tk() # File dialog
    path = filedialog.askopenfilename(title = "Select data file")
    root.destroy()
    # path = r'E:\Ice\analysis\0uM_X_10%_0\0uM_X_10%_0.csv'

    df = pd.read_csv(path, index_col='index').dropna() # Drop rows which have at least one NaN.
    try: # We only have to correct for the FPS difference once. If df.time_corrected is accessible, it is corrected.
        df.time_corrected
    except:
        df.times = df.times * 13 / 21.52 # Correct for faster playback speed.
        df['time_corrected'] = True

    df = fitting(df, path)
    # print(df.Q_opt.sample(), df.tau_opt.sample(), df.m_opt.sample())
    plot(df, path, True)
