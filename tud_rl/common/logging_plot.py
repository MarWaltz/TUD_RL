import csv

#import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt
import pandas as pd

from tud_rl.common.helper_fnc import exponential_smoothing


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

    # create plot
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    
    # define title
    if info is not None:
        fig.suptitle(f"{alg} ({info}) | {env_str} | Runtime (h): {runtime}")
    else:
        fig.suptitle(f"{alg} | {env_str} | Runtime (h): {runtime}")

    # first axis
    for col in df.columns[[col.startswith("Avg_Eval_ret") for col in df.columns]]:
        ax[0,0].plot(df["Timestep"], df[col], label=col)
        ax[0,0].plot(df["Timestep"], exponential_smoothing(df[col].values), label = "Smoothed " + col)
    ax[0,0].legend()
    ax[0,0].set_xlabel("Timestep")
    ax[0,0].set_ylabel("Test return")

    # second axis
    for col in df.columns[["Q_val" in col for col in df.columns]]:
        ax[0,1].plot(df["Timestep"], df[col], label=col)
    ax[0,1].legend()
    ax[0,1].set_xlabel("Timestep")
    ax[0,1].set_ylabel("Q-value")
    
    # third axis
    if "Loss" in df.columns:
        ax[1,0].plot(df["Timestep"], df["Loss"])
        ax[1,0].set_xlabel("Timestep")
        ax[1,0].set_ylabel("Loss")

    if any([col.startswith("Critic_loss") for col in df.columns]) and any([col.startswith("Actor_loss") for col in df.columns]):
        for col in df.columns[[col.startswith("Critic_loss") for col in df.columns]]:
            ax[1,0].plot(df["Timestep"], df[col], label=col)

        for col in df.columns[[col.startswith("Actor_loss") for col in df.columns]]:
            ax[1,0].plot(df["Timestep"], df[col], label=col)

        ax[1,0].legend()
        ax[1,0].set_xlabel("Timestep")
        ax[1,0].set_ylabel("Loss")

    # fourth axis
    ax[1,1].set_xlabel("Timestep")
    
    if "Avg_bias" in df.columns:
        ax[1,1].plot(df["Timestep"], df["Avg_bias"], label="Avg. bias")
        ax[1,1].legend()
    
    # safe figure and close
    plt.savefig(f"{dir}/{alg}_{env_str}.pdf")
    plt.close()
