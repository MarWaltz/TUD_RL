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
                                         bng_rel, dtr, get_ship_domain,
                                         head_inter, rtd)


def get_river_enc_range(ang:float):
    """Computes based on a relative bearing from TS perspective in [0,2pi) the assumed encounter range on the river."""
    ang = rtd(ang)
    river_enc_range_min = NM_to_meter(0.25)    # lower distance when we consider encounter situations on the river
    river_enc_range_max = NM_to_meter(0.50) 
    a = river_enc_range_min
    b = river_enc_range_max

    if 0 <= ang < 90.0:
        return a + ang * (b-a)/90.0
    
    elif 90.0 <= ang < 180.0:
        return (2*b-a) + ang * (a-b)/90.0

    elif 180.0 <= ang < 270.0:
        return (3*a-2*b) + ang * (b-a)/90.0

    else:
        return (4*b-3*a) + ang * (a-b)/90.0

def violates_river_traffic_rules(N0:float, E0:float, head0:float, v0:float, N1:float, E1:float, head1:float,\
    v1:float) -> bool:
    """Checks whether a situation violates the rules on the Elbe from Lighthouse Tinsdal to Cuxhaven.
    Args:
        N0(float):     north of OS
        E0(float):     east of OS
        head0(float):  heading of OS
        v0(float):     speed of OS
        N1(float):     north of TS
        E1(float):     east of TS
        head1(float):  heading of TS
        v1(float):     speed of TS"""
    # preparation
    ED_OS_TS = ED(N0=N0, E0=E0, N1=N1, E1=E1, sqrt=True)
    rev_dir = (abs(head_inter(head_OS=head0, head_TS=head1, to_2pi=False)) >= dtr(90.0))

    bng_rel_TS_pers = bng_rel(N0=N1, E0=E1, N1=N0, E1=E0, head0=head1)
    river_enc_range = get_river_enc_range(bng_rel_TS_pers)

    # check whether TS is too far away
    if ED_OS_TS > river_enc_range:
        return False
    else:
        # OS should pass opposing ships on their portside
        #if rev_dir:
        #    if dtr(0.0) <= bng_rel_TS_pers <= dtr(90.0):
        #        return True
        #else:
            # OS should let speedys pass on OS's portside
            #if v1 > v0:
            #    if dtr(180.0) <= bng_rel_TS_pers <= dtr(270.0):
            #        return True

        # normal target ships should be overtaken on their portside
        if (not rev_dir) and (v0 > v1):
            if dtr(90.0) <= bng_rel_TS_pers <= dtr(180.0):
                return True
    return False

def scenario_eval(dir):
    # -------------- 1. Run scenarios on river while saving plots -----------------
    prior_wd = os.getcwd()
    os.chdir(dir)

    eval_CONFIG_FILE = "hhos_plan_validate_river.yaml"
    eval_ACTOR_WEIGHTS  = "LSTMRecTD3_actor_weights.pth"
    eval_CRITIC_WEIGHTS = "LSTMRecTD3_critic_weights.pth"

    eval_config_path = f"{cont_path[0]}/{eval_CONFIG_FILE}"
    eval_c = ConfigFile(eval_config_path)
    eval_c.overwrite(critic_weights=eval_CRITIC_WEIGHTS)
    eval_c.overwrite(actor_weights=eval_ACTOR_WEIGHTS)
    eval_c.overwrite(river_curve="straight")

    eval_agent_name = "LSTMRecTD3"
    if eval_agent_name[-1].islower():
        eval_agent_name_red = eval_agent_name[:-2] + "Agent"
    else:
        eval_agent_name_red = eval_agent_name + "Agent"

    for i in range(4):
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
    # setup
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
    NROWS = 4
    NCOLS = 4
    _, axs = plt.subplots(nrows=NROWS, ncols=NCOLS, sharex=True, sharey=False, figsize=(8, 10))
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    for j in range(NCOLS):
        
        # data loading
        with open(f"Plan_river_straight_{j+1}.pkl", "rb") as f:
            x = pickle.load(f)

        # go to minutes
        sim_t = x["sim_t"]/60

        # title
        axs[0][j].set_title(f"Scenario {j+1}")

        for i in range(NROWS):
            # select axes
            ax = axs[i][j]

            # CTE
            if i == 0:
                ax.plot(sim_t, x["glo_ye"])
                ax.set_xlabel("Time in min")
                ax.set_ylabel("CTE in m")
                ax.set_ylim(-41.0, 41.0)
                if j != 0:
                    ax.yaxis.set_visible(False)

            # OS heading
            elif i == 1:
                ax.plot(sim_t, x["a"])
                ax.set_xlabel("Time in min")
                ax.set_ylim(-1.05, 1.05)
                ax.set_ylabel("Action")
                if j != 0:
                    ax.yaxis.set_visible(False)

            # TS
            else:
                for n in range(5):
                    if f"TS{n}_N" in x.columns:
                        
                        # ---------- check collision -------------
                        if i == 2:
                            dists = []
                            for t in range(len(sim_t)):

                                    # relative bearing
                                    bng_rel_TS = bng_rel(N0=x["OS_N"][t], E0=x["OS_E"][t], N1=x[f"TS{n}_N"][t], E1=x[f"TS{n}_E"][t], head0=x["OS_head"][t])

                                    # ship domain
                                    D = get_ship_domain(A=SD_A, B=SD_B, C=SD_C, D=SD_D, OS=None, TS=None, ang=bng_rel_TS)

                                    # dist
                                    d = ED(N0=x["OS_N"][t], E0=x["OS_E"][t], N1=x[f"TS{n}_N"][t], E1=x[f"TS{n}_E"][t], sqrt=True)-D
                                    dists.append(d)

                                    # catch collisions
                                    if d <= 0:
                                        ax.scatter(sim_t[t], d, color="red", s=20)
                            ax.plot(sim_t, dists, color=COLREG_COLORS[n], label=f"TS{n+1}")
                            ax.legend()
                            ax.set_ylabel("Distance in m")
                            ax.set_ylim(-50.0, 300.0)
                            if j != 0:
                                ax.yaxis.set_visible(False)

                        # ----------- rule violations -----------
                        elif i == 3:
                            rule_vios = []
                            for t in range(len(sim_t)):
                                bin = violates_river_traffic_rules(N0=x["OS_N"][t], E0=x["OS_E"][t], head0=x["OS_head"][t], v0=x["OS_V"][t], 
                                                                   N1=x[f"TS{n}_N"][t], E1=x[f"TS{n}_E"][t], head1=x[f"TS{n}_head"][t], v1=x[f"TS{n}_V"][t])
                                rule_vios.append(float(bin))
                            ax.plot(sim_t, rule_vios, color=COLREG_COLORS[n], label=f"TS{n+1}")
                            ax.legend()
                            ax.set_ylabel("Rule violation")
                            if j != 0:
                                ax.yaxis.set_visible(False)
    plt.savefig("RivScen_HHOS.pdf", bbox_inches="tight")
    plt.close("all")

    # change back wd
    os.chdir(prior_wd)
