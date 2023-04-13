import os
import pickle

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import tud_rl.agents.continuous as agents
from tud_rl.agents.base import _Agent
from tud_rl.common.configparser import ConfigFile
from tud_rl.configs.continuous_actions import __path__ as cont_path
from tud_rl.envs._envs.VesselFnc import (COLREG_COLORS, ED, NM_to_meter,
                                         bng_rel, get_ship_domain)
from tud_rl.envs._envs.VesselPlots import get_triangle


def Imazu_eval(dir):
    # -------------- 1. Run Imazu problems while saving plots -----------------
    prior_wd = os.getcwd()
    os.chdir(dir)

    eval_CONFIG_FILE = "hhos_plan_validate_open.yaml"
    eval_ACTOR_WEIGHTS  = "LSTMRecTD3_actor_weights.pth"
    eval_CRITIC_WEIGHTS = "LSTMRecTD3_critic_weights.pth"

    eval_config_path = f"{cont_path[0]}/{eval_CONFIG_FILE}"
    eval_c = ConfigFile(eval_config_path)
    eval_c.overwrite(critic_weights=eval_CRITIC_WEIGHTS)
    eval_c.overwrite(actor_weights=eval_ACTOR_WEIGHTS)
    eval_c.overwrite(full_RL=False)
    eval_c.overwrite(APF_TS=False)
    eval_c.overwrite(star_formation=False)
    eval_c.overwrite(clock_formation=False)

    eval_agent_name = "LSTMRecTD3"
    if eval_agent_name[-1].islower():
        eval_agent_name_red = eval_agent_name[:-2] + "Agent"
    else:
        eval_agent_name_red = eval_agent_name + "Agent"

    for i in range(22):
        eval_c.overwrite(scenario=i+1)

        # create env
        eval_env: gym.Env = gym.make(eval_c.Env.name, **eval_c.Env.env_kwargs)
        eval_c.state_shape = eval_env.observation_space.shape[0]

        # mode and action details
        eval_c.mode = "test"
        eval_c.num_actions = eval_env.action_space.shape[0]

        # init agent
        agent_ = getattr(agents, eval_agent_name_red)  # Get agent class by name
        eval_agent: _Agent = agent_(eval_c, eval_agent_name)  # Instantiate agent

        # LSTM: init history
        if eval_agent.needs_history:
            s_hist = np.zeros((eval_agent.history_length, eval_agent.state_shape))
            a_hist = np.zeros((eval_agent.history_length, eval_agent.num_actions))
            hist_len = 0

        # get initial state
        s = eval_env.reset()

        d = False
        while not d:

            if eval_agent.needs_history:
                a = eval_agent.select_action(s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
            else:
                a = eval_agent.select_action(s)

            # perform step
            s2, _, d, _ = eval_env.step(a)

            # LSTM: update history
            if eval_agent.needs_history:
                if hist_len == eval_agent.history_length:
                    s_hist = np.roll(s_hist, shift=-1, axis=0)
                    s_hist[eval_agent.history_length - 1, :] = s

                    a_hist = np.roll(a_hist, shift=-1, axis=0)
                    a_hist[eval_agent.history_length - 1, :] = a
                else:
                    s_hist[hist_len] = s
                    a_hist[hist_len] = a
                    hist_len += 1
            s = s2
    
    # ------------------------ 2. Viz -------------------------------
    font = {'size' : 8}
    matplotlib.rc('font', **font)

    # ship domain
    Lpp = 64.0
    B = 11.6
    SD_A = 2 * Lpp + 0.5 * Lpp
    SD_B = 2 * B + 0.5 * B   
    SD_C = 2 * B + 0.5 * Lpp
    SD_D = 4 * B + 0.5 * B

    # viz
    TIME_GAP = 5 * 60 # [s]
    NROWS = 6
    NCOLS = 4
    fig, axs = plt.subplots(nrows=NROWS, ncols=NCOLS, sharex=True, sharey=True, figsize=(8, 12))
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    for i in range(NROWS):
        for j in range(NCOLS):

            # data loading
            ij = i*NCOLS + j
            try:
                with open(f"HHOS_Validate_Plan_Imazu_{ij+1}.pkl", "rb") as f:
                    x = pickle.load(f)
            except:
                continue
            ax = axs[i][j]

            # normalization
            N_zero = x["OS_N"][0]
            E_zero = x["OS_E"][0]

            for col in x.columns:
                if col.endswith("N"):
                    x[col] = x[col] - N_zero
                elif col.endswith("E"):
                    x[col] = x[col] - E_zero

            # labels
            if ij in [18, 19, 20, 21]:
                ax.set_xlabel("East [NM]", fontsize=8)
            
            if j == 0:
                ax.set_ylabel("North [NM]", fontsize=8)

            # OS triangle
            rec = get_triangle(E=x["OS_E"][0], N=x["OS_N"][0], l=3*64.0, heading=x["OS_head"][0],
                            facecolor="white", edgecolor="black", linewidth=1.5, zorder=10)
            ax.add_patch(rec)

            # OS trajectory with time dots
            ax.plot(x["OS_E"], x["OS_N"], color="black")
            ts = np.where(x["sim_t"] % TIME_GAP == 0)[0]
            ax.scatter(x["OS_E"][ts], x["OS_N"][ts], color="black", s=5)

            # TS
            for n in range(5):
                if f"TS{n}_N" in x.columns:

                    # triangle
                    rec = get_triangle(E=x[f"TS{n}_E"][0], N=x[f"TS{n}_N"][0], l=3*64.0, heading=x[f"TS{n}_head"][0],
                                    facecolor="white", edgecolor=COLREG_COLORS[n], linewidth=1.5, zorder=10)
                    ax.add_patch(rec)

                    # trajectory with time dots
                    ax.plot(x[f"TS{n}_E"], x[f"TS{n}_N"], color=COLREG_COLORS[n])
                    ax.scatter(x[f"TS{n}_E"][ts], x[f"TS{n}_N"][ts], color=COLREG_COLORS[n], s=5)

                    #----- check collisions -----
                    for t in range(len(x["OS_N"])):

                        # relative bearing
                        bng_rel_TS = bng_rel(N0=x["OS_N"][t], E0=x["OS_E"][t], N1=x[f"TS{n}_N"][t], E1=x[f"TS{n}_E"][t], head0=x["OS_head"][t])

                        # ship domain
                        D = get_ship_domain(A=SD_A, B=SD_B, C=SD_C, D=SD_D, OS=None, TS=None, ang=bng_rel_TS)

                        # dist
                        ED_TS = ED(N0=x["OS_N"][t], E0=x["OS_E"][t], N1=x[f"TS{n}_N"][t], E1=x[f"TS{n}_E"][t], sqrt=True)

                        # catch collisions
                        if ED_TS - D <= 0:
                            ax.scatter(x["OS_E"][t], x["OS_N"][t], color="red", s=20)

    # ticks and labels
    xlim = axs[0][0].get_xlim()
    num_ticks = int(np.floor(np.diff(xlim) / NM_to_meter(1)))
    if num_ticks % 2 == 0:
        tick_pos = np.mean(xlim)-(num_ticks/2-1)*NM_to_meter(1)-NM_to_meter(0.5) + np.arange(num_ticks) * NM_to_meter(1)
    else:
        tick_pos = np.mean(xlim)-(num_ticks-1)/2*NM_to_meter(1)-NM_to_meter(0.5) + np.arange(num_ticks) * NM_to_meter(1)

    axs[-1][0].set_xticks(tick_pos)
    axs[-1][0].set_xticklabels([nm for nm in range(num_ticks) if nm % 1 == 1])

    axs[-1][-1].set_visible(False)
    axs[-1][-2].set_visible(False)

    plt.savefig(f"Imazu_HHOS.pdf", bbox_inches="tight")
    plt.close("all")

    # change back wd
    os.chdir(prior_wd)
