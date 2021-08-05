import csv

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_from_progress(dir, alg, env_str, info=None):
    """Plots based on a given 'progress.txt' the evaluation return, Q_values and losses.

    Args:
        dir (string):     directory of 'progress.txt', most likely something like experiments/some_number
        env_str (string): name of environment 
        alg (string):     used algorithm
        info (string):    further information to display in the header
    """
    # open progress file and load it into pandas
    with open(f"{dir}/progress.txt") as f:
        reader = csv.reader(f, delimiter="\t")
        d = list(reader)

    df = pd.DataFrame(d)
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df = df.astype(float)

    # define moving and rolling average functions
    # Note: this could be defined outside of plot_from_progress(), but having just one function appears handy
    def moving_average(a, n):
        ret = np.cumsum(a)
        ret[n:] = ret[n:] - ret[:-n]
        res = ret[n - 1:] / n
        fill = np.array([None] * (n-1))
        return  np.concatenate((fill, res), axis=0)

    def rolling_average(a):
        ret = np.cumsum(a)
        div = np.cumsum(np.ones_like(a))
        return ret / div

    # create plot
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    
    # define title
    if info is not None:
        fig.suptitle(f"{alg} ({info}) | {env_str}")
    else:
        fig.suptitle(f"{alg} | {env_str}")

    # fill first axis
    ax[0,0].plot(df["Timestep"], df["Avg_Eval_ret"], label = "Avg. test return")
    if df.shape[0] > 20:
        ax[0,0].plot(df["Timestep"], moving_average(df["Avg_Eval_ret"].values, n = 20), label = "MA-20")
    ax[0,0].plot(df["Timestep"], rolling_average(df["Avg_Eval_ret"]), label = "All-time MA")
    ax[0,0].legend()
    ax[0,0].set_xlabel("Timestep")
    ax[0,0].set_ylabel("Test return")

    # fill second axis
    if any([alg.startswith(name) for name in ["TD3", "LSTM-TD3", "SAC", "LSTM-SAC"]]):
        ax[0,1].plot(df["Timestep"], df["Avg_Q1_val"])
        ax[0,1].set_ylabel("Avg_Q1_val")
    else:
        ax[0,1].plot(df["Timestep"], df["Avg_Q_val"])
        ax[0,1].set_ylabel("Avg_Q_val")
    
    ax[0,1].set_xlabel("Timestep")
    
    # fill third axis
    if any([alg.startswith(name) for name in ["TD3", "LSTM-TD3", "SAC", "LSTM-SAC"]]):
        ax[1,0].plot(df["Timestep"], df["Critic_loss"])
        ax[1,0].set_xlabel("Timestep")
        ax[1,0].set_ylabel("Critic loss")
    else:
        ax[1,0].plot(df["Timestep"], df["Loss"])
        ax[1,0].set_xlabel("Timestep")
        ax[1,0].set_ylabel("Loss")

    # fill fourth axis
    if any([alg.startswith(name) for name in ["TD3", "LSTM-TD3", "SAC", "LSTM-SAC"]]):
        ax[1,1].plot(df["Timestep"], df["Actor_loss"])
        ax[1,1].set_xlabel("Timestep")
        ax[1,1].set_ylabel("Actor loss")

    # safe figure and close
    plt.savefig(f"{dir}/{alg}_{env_str}.png")
    plt.close()
