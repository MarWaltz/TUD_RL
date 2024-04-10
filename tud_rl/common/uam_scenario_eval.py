import os
import pickle

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from bluesky.tools.geo import qdrpos

import tud_rl.agents.continuous as agents
from tud_rl.agents.base import _Agent
from tud_rl.common.configparser import ConfigFile
from tud_rl.configs.continuous_actions import __path__ as cont_path
from tud_rl.envs._envs.HHOS_Fnc import to_utm
from tud_rl.envs._envs.VesselFnc import COLREG_COLORS, meter_to_NM


def vec_ED(e1, e2, n1, n2):
    return np.sqrt((e1-e2)**2 + (n1-n2)**2)


def uam_eval(dir):
    # -------------- 1. Run scenario while saving plots -----------------
    prior_wd = os.getcwd()
    os.chdir(dir)

    eval_CONFIG_FILE = "uam_modular_validate.yaml"
    eval_ACTOR_WEIGHTS  = "LSTMRecTD3_actor_weights.pth"
    eval_CRITIC_WEIGHTS = "LSTMRecTD3_critic_weights.pth"

    eval_config_path = f"{cont_path[0]}/{eval_CONFIG_FILE}"
    eval_c = ConfigFile(eval_config_path)
    eval_c.overwrite(critic_weights=eval_CRITIC_WEIGHTS)
    eval_c.overwrite(actor_weights=eval_ACTOR_WEIGHTS)
    eval_c.overwrite(situation=1)
    eval_c.overwrite(seed=100)

    eval_agent_name = "LSTMRecTD3"
    if eval_agent_name[-1].islower():
        eval_agent_name_red = eval_agent_name[:-2] + "Agent"
    else:
        eval_agent_name_red = eval_agent_name + "Agent"

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
        s2, _, d, _ = eval_env.step(eval_agent)

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
    DOT_EVERY = 20 # [s] since dt = 1s
    D_LAT = 60
    D_LON = 9
    D_radius = 200  # [m]
    D_spawn_radius    = 1000 # [m]
    D_respawn_radius  = 1200 # [m]

    D_incident = 100 # [m]
    D_accident = 10  # [m]
    CLOCK_DEGS = np.linspace(0.0, 360.0, num=100, endpoint=True)

    # loading data
    with open("UAM_ValScene_1_12.pkl", "rb") as f:
        data = pickle.load(f)

    # rename columns for convenient access
    new_cols = [col[0:37] for col in data.columns[1:]]
    new_cols = dict.fromkeys(new_cols)
    for i, key in enumerate(new_cols):
        new_cols[key] = str(i)

    for i, col in enumerate(data.columns):
        for key in new_cols:
            if key in col:
                data = data.rename(columns={col : col.replace(key, new_cols[key])})

    N = len(new_cols) # number of vehicles
    data_n    = data.iloc[:, [col.endswith("n") for col in data.columns]]
    data_e    = data.iloc[:, [col.endswith("e") for col in data.columns]]
    data_hdg  = data.iloc[:, [col.endswith("hdg") for col in data.columns]]
    #data_goal = data.iloc[:, [col.endswith("goal") for col in data.columns]]

    for what in ["traj", "dist"]:
        # first case
        if what == "traj":

            # create figure
            fig = plt.figure(figsize=(8, 16))
            plt.subplots_adjust(hspace=0.0, wspace=0.0)

            ROWS = 5
            COLS = 2
            gs = fig.add_gridspec(ROWS, COLS)
            ax_list = []

            for row_idx in range(ROWS):
                for col_idx in range(COLS):
                    ax_list.append(fig.add_subplot(gs[row_idx, col_idx]))

            # plot
            for i, ax in enumerate(ax_list):

                try:
                    # appearance
                    ax.text(0.05, 0.875, i+1, size=12, transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # spawning area
                    lats, lons = map(list, zip(*[qdrpos(latd1=D_LAT, lond1=D_LON, qdr=deg, dist=meter_to_NM(D_spawn_radius)) for deg in CLOCK_DEGS]))
                    ns, es, _ = to_utm(lat=np.array(lats), lon=np.array(lons))
                    ax.plot(es, ns, color="purple", alpha=0.75)

                    # restricted area
                    lats, lons = map(list, zip(*[qdrpos(latd1=D_LAT, lond1=D_LON, qdr=deg, dist=meter_to_NM(D_radius))\
                        for deg in CLOCK_DEGS]))
                    ns, es, _ = to_utm(lat=np.array(lats), lon=np.array(lons))
                    ax.plot(es, ns, color="purple", alpha=0.75)

                    # respawn area
                    #lats, lons = map(list, zip(*[qdrpos(latd1=D_LAT, lond1=D_LON, qdr=deg, dist=meter_to_NM(D_respawn_radius))\
                    #    for deg in CLOCK_DEGS]))
                    #ns, es, _ = to_utm(lat=np.array(lats), lon=np.array(lons))
                    #ax.plot(es, ns, color="black")

                    # plot vehicles
                    rows = range(100*i, 100*(i+1))
                    rows_red = [r for r in list(rows) if r % DOT_EVERY == 0]

                    for n in range(N):
                        ax.plot(data_e.iloc[rows, n], data_n.iloc[rows, n], color=COLREG_COLORS[n])

                        for r in rows_red:
                            ax.scatter(data_e.iloc[r, n], data_n.iloc[r, n], color=COLREG_COLORS[n],
                                    marker=(3, 0, -data_hdg.iloc[r, n]))
                except:
                    pass

            #plt.show()
            plt.savefig("UAM_Val_traj.pdf", bbox_inches="tight")
        
        # second case
        else:

            # create figure
            fig = plt.figure(figsize=(8, 16))
            #plt.subplots_adjust(hspace=0.0, wspace=0.0)

            ROWS = int(N/2)
            COLS = 2
            gs = fig.add_gridspec(ROWS, COLS)
            ax_list = []

            for row_idx in range(ROWS):
                for col_idx in range(COLS):
                    ax_list.append(fig.add_subplot(gs[row_idx, col_idx]))

            # plot distances
            for i, ax in enumerate(ax_list):

                try:
                    ax.text(0.85, 0.875, f"FT {i+1}", size=8, transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_ylim(-100, 2500)

                    e1 = data_e.iloc[:, i]
                    n1 = data_n.iloc[:, i]

                    for other in range(N):
                        if i != other:
                            d = vec_ED(e1=e1, n1=n1, e2=data_e.iloc[:, other], n2=data_n.iloc[:, other])
                            ax.plot(d)
                            #if any(d <= D_incident):
                            #    raise Exception("Collision!")
                    ax.axhline(y=D_incident, color="red")
                except:
                    pass

            #plt.show()
            plt.savefig("UAM_Val_dist.pdf", bbox_inches="tight")
 
    plt.close("all")

    # change back wd
    os.chdir(prior_wd)
