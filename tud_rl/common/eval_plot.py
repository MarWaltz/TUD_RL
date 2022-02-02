import csv

import numpy as np
import pandas as pd
#matplotlib.use("agg")
import matplotlib.pyplot as plt


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

    runtime = df["Runtime_in_h"].iloc[-1].round(3)

    # Note: This helper fnc could be defined outside of plot_from_progress(), 
    #       but having just one function appears handy.
    def exponential_smoothing(x, alpha=0.05):
        s = np.zeros_like(x)

        for idx, x_val in enumerate(x):
            if idx == 0:
                s[idx] = x[idx]
            else:
                s[idx] = alpha * x_val + (1-alpha) * s[idx-1]

        return s

    # create plot
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    
    # define title
    if info is not None:
        fig.suptitle(f"{alg} ({info}) | {env_str} | Runtime (h): {runtime}")
    else:
        fig.suptitle(f"{alg} | {env_str} | Runtime (h): {runtime}")

    # fill first axis
    ax[0,0].plot(df["Timestep"], df["Avg_Eval_ret"], label = "Avg. test return")
    ax[0,0].plot(df["Timestep"], exponential_smoothing(df["Avg_Eval_ret"].values), label = "Exp. smooth. return")
    ax[0,0].legend()
    ax[0,0].set_xlabel("Timestep")
    ax[0,0].set_ylabel("Test return")

    # fill second axis
    if any([alg.startswith(name.lower()) for name in ["TD3", "LSTM_TD3", "SAC", "LSTM_SAC"]]):
        ax[0,1].plot(df["Timestep"], df["Avg_Q1_val"])
        ax[0,1].set_ylabel("Avg_Q1_val")
    else:
        ax[0,1].plot(df["Timestep"], df["Avg_Q_val"])
        ax[0,1].set_ylabel("Avg_Q_val")
    
    ax[0,1].set_xlabel("Timestep")
    
    # fill third axis
    if any([alg.startswith(name.lower()) for name in ["DDPG", "TD3", "LSTM_TD3", "SAC", "LSTM_SAC", "TQC"]]):
        ax[1,0].plot(df["Timestep"], df["Critic_loss"])
        ax[1,0].set_xlabel("Timestep")
        ax[1,0].set_ylabel("Critic loss")
    else:
        ax[1,0].plot(df["Timestep"], df["Loss"])
        ax[1,0].set_xlabel("Timestep")
        ax[1,0].set_ylabel("Loss")

    # fill fourth axis
    if all(ele in df.columns for ele in ["Avg_bias", "Std_bias", "Max_bias", "Min_bias"]):
        ax[1,1].plot(df["Timestep"], df["Avg_bias"], label="Avg. bias")
        ax[1,1].plot(df["Timestep"], df["Std_bias"], label="Std. bias")
        ax[1,1].plot(df["Timestep"], df["Max_bias"], label="Max. bias")
        ax[1,1].plot(df["Timestep"], df["Min_bias"], label="Min. bias")
        ax[1,1].legend()
        ax[1,1].set_xlabel("Timestep")

    elif any([alg.startswith(name.lower()) for name in ["DDPG", "TD3", "LSTM_TD3", "SAC", "LSTM_SAC", "TQC"]]):
        ax[1,1].plot(df["Timestep"], df["Actor_loss"])
        ax[1,1].set_xlabel("Timestep")
        ax[1,1].set_ylabel("Actor loss")

    # safe figure and close
    plt.savefig(f"{dir}/{alg}_{env_str}.pdf")
    plt.close()
