import random
from copy import copy
from string import ascii_letters
from typing import List

import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt
from mycolorpy import colorlist as mcp

from tud_rl.agents.base import _Agent

COLORS = [plt.rcParams["axes.prop_cycle"].by_key()["color"][i] for i in range(8)] + 5 * mcp.gen_color(cmap="tab20b", n=20) 

class Vehicle:
    def __init__(self, x, v) -> None:
        self.x = x
        self.v = v
    
    def move(self, a) -> None:
        assert a in [0, 1, 2, 3], "Only four possible actions in dynamics update."
        if a in [1, 3]:
            new_x = self.x + self.v
            if np.sign(new_x) == np.sign(self.x):
                self.x = new_x

    def reset_x(self) -> None:
        if self.x > 0:
            self.x =  float(np.random.uniform(9, 11, 1))
        else:
            self.x = -float(np.random.uniform(9, 11, 1))


class Destination:
    def __init__(self) -> None:
        self.radius = 2

        # timing
        self.dt            = 1   # simulation time step
        self._t_close      = 10  # time the destination is closed after an aircraft has entered 
        self._t_nxt_open   = 0   # current time until the destination opens again
        self._t_open_since = 0   # current time since the vertiport is open
        self._was_open = True
        self.open()

    def reset(self):
        self.open()

    def step(self, planes: List[Vehicle]):
        """Updates status of the destination.
        Returns:
            np.ndarray([number_of_planes,]): who entered a closed destination
            np.ndarray([number_of_planes,]): who entered an open destination"""
        # count time until next opening
        if self._is_open is False:
            self._t_nxt_open -= self.dt
            if self._t_nxt_open <= 0:
                self.open()
        else:
            self._t_open_since += self.dt

        # store opening status
        self._was_open = copy(self._is_open)

        # check who entered a closed or open destination
        entered_close = np.zeros(len(planes), dtype=bool)
        entered_open  = np.zeros(len(planes), dtype=bool)

        for i, p in enumerate(planes):
            if abs(p.x) <= self.radius:            
                if self._is_open:
                    entered_open[i] = True
                else:
                    entered_close[i] = True

        #  close if someone entered
        if any(entered_open):
            self.close()

        return entered_close, entered_open

    def open(self):
        self._t_open_since = 0
        self._t_nxt_open = 0
        self._is_open = True
        self.color = "green"
    
    def close(self):
        self._t_open_since = 0
        self._is_open = False
        self._t_nxt_open = copy(self._t_close)
        self.color = "red"

    @property
    def t_nxt_open(self):
        return self._t_nxt_open

    @property
    def t_close(self):
        return self._t_close

    @property
    def t_open_since(self):
        return self._t_open_since

    @property
    def is_open(self):
        return self._is_open

    @property
    def was_open(self):
        return self._was_open


class SimpleComm(gym.Env):
    def __init__(self,
                 N_agents_max :int,
                 RIAL1        :bool,  # RIAL1 is a simplified version in which we just expand the action space to include comm-signals
                 RIAL2        :bool,  # RIAL2 is the RIAL approach according to Foerster et al. (2016)
                 RIAL2_n_comms:int,
                 DIAL         :bool,
                 DIAL_n_comms :int,
                 game_setup   :bool,
                 partial_obs  :bool,
                 competitive  :bool,
                 abs_x        :bool):
        super(SimpleComm, self).__init__()

        # checks
        assert N_agents_max == 2, "Use two agents, please."
        RIAL = RIAL1 or RIAL2
        assert not(RIAL1 and RIAL2), "Choose RIAL1 or RIAL2, not both."
        assert not (RIAL and DIAL), "Choose RIAL or DIAL, not both."

        self.RIAL1 = RIAL1
        self.RIAL2 = RIAL2
        self.RIAL  = RIAL
        self.DIAL  = DIAL

        self.RIAL2_n_comms = RIAL2_n_comms
        self.DIAL_n_comms  = DIAL_n_comms

        assert game_setup in ["alternate", "coin"], "'game_setup' should be either 'alternate' or 'coin'."
        self.game_setup = game_setup

        self.partial_obs = partial_obs
        self.competitive = competitive
        self.abs_x = abs_x

        # config
        self.obs_size = 8 if not self.partial_obs else 7 # own (and other's) x, v and coin/live time, two vertiport time infos
        if self.RIAL:
            self.obs_size += 1  # always plus one since there is only one communication signal, which has two possible realizations
        if self.DIAL:
            self.obs_size += DIAL_n_comms

        self.observation_space = spaces.Box(low  = np.full(self.obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(self.obs_size,  np.inf, dtype=np.float32))
        if self.RIAL1:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Discrete(2)

        self._max_episode_steps = 100

        # destination
        self.dest = Destination()

        # viz
        self.plot_reward = True
        self.plot_state  = True

        info = "living time" if game_setup == "alternate" else "coin"
        self.obs_names = ["|own x|" if abs_x else "own x", "|own v|" if abs_x else "own v", f"own {info}", "t_nxt_open", "t_open_since",
                          "|other x|" if abs_x else "other x",  "|other v|" if abs_x else "other v"]
        if not self.partial_obs:
            self.obs_names += [f"other {info}"]

        if self.RIAL:
            self.comm_names = ["comm"]
        if self.DIAL:
            self.comm_names = []
            for i in range(self.DIAL_n_comms):
                self.comm_names.append("comm " + ascii_letters[i]) 
        if self.RIAL or self.DIAL:     
            self.obs_names += self.comm_names

    def reset(self):
        """Resets environment to initial state."""
        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # create some vehicles
        self.N_vehicles = 2
        self.vehicles:List[Vehicle] = []
        
        # random who is player0 and player1
        if bool(random.getrandbits(1)):
            self.vehicles.append(self._get_vehicle(True))
            self.vehicles.append(self._get_vehicle(False))
        else:
            self.vehicles.append(self._get_vehicle(False))
            self.vehicles.append(self._get_vehicle(True))

        # init live times or coins
        if self.game_setup == "alternate":
            self.ts_alive = np.zeros(self.N_vehicles)
        else:
            self.coins = np.random.choice([-1, 1], size=2)

        # reset dest
        self.dest.reset()

        # init communication signals from last time step
        if self.RIAL:
            self.comm_tm1 = [0] * self.N_vehicles
        elif self.DIAL:
            self.comm_tm1 = [[0] * self.DIAL_n_comms] * self.N_vehicles

        # init state
        self._set_state()
        self.state_init = self.state
        return self.state

    def _get_vehicle(self, neg:bool):
        if neg:
            x = -float(np.random.uniform(9, 11, 1))
            v =  float(np.random.uniform(0.5, 1.5, 1))
        else:
            x = +float(np.random.uniform(9, 11, 1))
            v = -float(np.random.uniform(0.5, 1.5, 1))
        return Vehicle(x=x, v=v)

    def _set_state(self):
        if self.DIAL:
            self.state, self.comm0_id = self._get_state_multi(True)
        else:
            self.state = self._get_state(0)

    def _get_state_multi(self, with_comm0_id:bool=False) -> None:
        """Computes the state in the multi-agent scenario."""
        s = np.zeros((self.N_vehicles, self.obs_size), dtype=np.float32)

        if with_comm0_id:
            comm0_id = np.zeros(self.N_vehicles, dtype=np.float32)

        for i, _ in enumerate(self.vehicles):
            if with_comm0_id:
                s[i], comm0_id[i] = self._get_state(i, True)
            else:
                s[i] = self._get_state(i, False)

        if with_comm0_id:
            return s, comm0_id
        else:
            return s

    def _get_state(self, i:int, with_comm0_id:bool=False) -> np.ndarray:
        """Computes the state from the perspective of the i-th agent of the internal plane array."""
        assert i in [0, 1], "Two vehicles, please."
        other_i = int(not i)

        # own position, other position, time to next open, time since open 
        x_one = self.vehicles[i].x/10
        v_one = self.vehicles[i].v
        t_one = self.ts_alive[i]/10 if self.game_setup == "alternate" else self.coins[i]
        
        x_two = self.vehicles[other_i].x/10
        v_two = self.vehicles[other_i].v

        if not self.partial_obs:
            t_two = self.ts_alive[other_i]/10 if self.game_setup == "alternate" else self.coins[other_i]

        if self.abs_x:
            x_one = abs(x_one)
            v_one = abs(v_one)
            #if not self.partial_obs:
            x_two = abs(x_two)
            v_two = abs(v_two)
            
        s_i = np.array([x_one, v_one, t_one, 1.0-self.dest.t_nxt_open/self.dest.t_close,
                        1.0-self.dest.t_open_since/self.dest.t_close, x_two, v_two])
        if not self.partial_obs:
            s_i = np.append(s_i, np.array([t_two]))

        # add comm-signal of the other guy
        if self.DIAL:
            s_i = np.append(s_i, self.comm_tm1[other_i])
        elif self.RIAL:
            s_i = np.append(s_i, [self.comm_tm1[other_i]])

        # get position of first communication signal of Agent 0
        if with_comm0_id:
            if i == 0:
                comm0_id = -1
            else:
                comm0_id = 8 if not self.partial_obs else 7

        if with_comm0_id:
            return s_i, comm0_id
        else:
            return s_i

    def _a_to_comm(self, a):
        """Relates a discrete (communication) action to a non-zero communication signal."""
        if self.RIAL1:
            assert a in [0, 1, 2, 3], "Unknown action."
            return -1.0 if a in [0, 1] else 1.0
        elif self.RIAL2:
            assert a in range(self.RIAL2_n_comms), "Unknown action."
            return -1.0 if a == 0 else float(a)
        else:
            raise RuntimeError()

    def step(self, a):
        """Arg a: _agent"""
        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += 1

        # get agent
        cnt_agent:_Agent = a

        # collect states from vehicles
        states_multi = self._get_state_multi(False)

        # prepare storage of communication signals
        if self.RIAL or self.DIAL:
            comm_new = []
        if self.DIAL:
            all_acts = []

        for i, p in enumerate(self.vehicles):

            # fly planes depending on whether they are RL-, VFG-, or RND-controlled
            if self.DIAL or self.RIAL2:
                act, comm = cnt_agent.select_action(s = states_multi[i])#

                if self.RIAL2 and i == 0:
                    a0_env  = act
                    a0_comm = comm
            else:
               act = cnt_agent.select_action(s = states_multi[i])
            #act = float(np.random.randint(low=0, high=2, size=1))

            # update dynamics
            p.move(act)

            # store comm signal and possibly action
            if self.RIAL1:
                comm_new.append(self._a_to_comm(act))

            elif self.RIAL2:
                comm_new.append(self._a_to_comm(comm))

            elif self.DIAL:
                comm_new.append(list(comm.flatten()))
                all_acts.append(act)

        # update live times/coins
        if self.game_setup == "alternate":
            self.ts_alive += 1
            self.ts_before_respawn = copy(self.ts_alive)
        else:
            self.coins_before_respawn = copy(self.coins)
            self.xs_before_respawn = [v.x for v in self.vehicles]

        # store communication signal
        if self.RIAL or self.DIAL:
            self.comm_tm1 = comm_new

        # check destination entries
        entered_close, entered_open = self.dest.step(self.vehicles)

        # respawning
        self._handle_respawn(entered_open)

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward(entered_close, entered_open)
        d = self._done()

        if self.DIAL:
            return self.state, self.r, d, {"a": np.array(all_acts), "comm0_id": self.comm0_id}
        elif self.RIAL2:
            return self.state, float(self.r[0]), d, {"a": a0_env, "a_comm": a0_comm}  
        else:
            return self.state, float(self.r[0]), d, {}

    def _handle_respawn(self, respawn_flags):
        for i, veh in enumerate(self.vehicles):
            if respawn_flags[i]:

                # respawn the vehicle
                if veh.v > 0:
                    self.vehicles[i] = self._get_vehicle(neg=True)
                else:
                    self.vehicles[i] = self._get_vehicle(neg=False)

                # reset live time / coins
                if self.game_setup == "alternate":
                    self.ts_alive[i] = 0
                else:
                    self.coins[i] = int(np.random.choice([-1, 1], size=1))

    def _calculate_reward(self, entered_close:np.ndarray, entered_open:np.ndarray):
        """Args:
            entered_close: np.ndarray([number_of_planes,]): who entered a closed destination
            entered_open:  np.ndarray([number_of_planes,]): who entered an open destination"""
        r_goal = np.zeros((self.N_vehicles, 1), dtype=np.float32)
        
        # closed goal entering
        if any(entered_close):
            r_goal -= 5.0

        # open goal entering
        if any(entered_open):

            # check whether only one vehicle entered
            if sum(entered_open) == 1:

                if self.game_setup == "alternate":

                    # check whether the vehicle which entered had the longest living time
                    entering_i = np.where(entered_open)[0][0]
                    if self.ts_before_respawn[entering_i] == np.max(self.ts_before_respawn):
                        r_goal += 5.0

                    # otherwise punish all
                    else:
                        r_goal -= 5.0

                else:
                    # good when the left vehicle (meaning x < 0) entered when the coins are identical
                    entering_i = np.where(entered_open)[0][0]
                    left_entered = self.xs_before_respawn[entering_i] < 0

                    if abs(sum(self.coins_before_respawn)) == 2:
                        if left_entered:
                            r_goal += 5.0
                        else:
                            r_goal -= 5.0
                    else:
                        if left_entered:
                            r_goal -= 5.0
                        else:
                            r_goal += 5.0

            # bad if someone entered simultaneously
            else:
                r_goal -= 5.0

        # incentive structure
        if self.dest.is_open:
            r_goal -= self.dest.t_open_since/self.dest.t_close
        else:
            r_goal += 0.25
        self.r = r_goal

    def _done(self):
        # artificial done signal
        if self.step_cnt >= self._max_episode_steps:
            return True
        return False

    def __str__(self):
        return f"Step: {self.step_cnt}, Sim-Time [s]: {int(self.sim_t)}" + "\n" +\
            f"Time-to-open [s]: {int(self.dest.t_nxt_open)}, Time-since-open[s]: {int(self.dest.t_open_since)}"

    def render(self, mode=None):
        """Renders the current environment."""

        # plot every nth timestep
        if self.step_cnt % 1 == 0: 
            
            # init figure
            if len(plt.get_fignums()) == 0:
                if self.plot_reward and self.plot_state:
                    self.f = plt.figure(figsize=(12, 8))
                    self.gs  = self.f.add_gridspec(3, 2)
                    self.ax1 = self.f.add_subplot(self.gs[0, 0]) # ship
                    self.ax2 = self.f.add_subplot(self.gs[0, 1]) # reward
                    self.ax3 = self.f.add_subplot(self.gs[1, 0]) # state0
                    self.ax4 = self.f.add_subplot(self.gs[2, 0]) # comm0
                    self.ax5 = self.f.add_subplot(self.gs[1, 1]) # state1
                    self.ax6 = self.f.add_subplot(self.gs[2, 1]) # comm1

                elif self.plot_reward:
                    raise NotImplementedError()
                    self.f = plt.figure(figsize=(14, 8))
                    self.gs  = self.f.add_gridspec(1, 2)
                    self.ax1 = self.f.add_subplot(self.gs[0, 0]) # ship
                    self.ax2 = self.f.add_subplot(self.gs[0, 1]) # reward

                elif self.plot_state:
                    raise NotImplementedError()
                    self.f = plt.figure(figsize=(14, 8))
                    self.gs  = self.f.add_gridspec(2, 2)
                    self.ax1 = self.f.add_subplot(self.gs[:, 0]) # ship
                    self.ax3 = self.f.add_subplot(self.gs[0, 1]) # state
                    self.ax4 = self.f.add_subplot(self.gs[1, 1]) # comm

                else:
                    self.f, self.ax1 = plt.subplots(1, 1, figsize=(10, 10))
                plt.ion()
                plt.show()           

            # storage
            if self.plot_reward:
                if self.step_cnt == 0:
                    self.ax2.r0 = np.zeros(self._max_episode_steps)
                    self.ax2.r1 = np.zeros(self._max_episode_steps)
                else:
                    self.ax2.r0[self.step_cnt] = float(self.r[0])
                    self.ax2.r1[self.step_cnt] = float(self.r[1])

            if self.plot_state:
                if self.step_cnt == 0:
                    self.ax3.s = np.zeros((self.obs_size, self._max_episode_steps))
                    self.ax5.s = np.zeros((self.obs_size, self._max_episode_steps))

                    if self.RIAL:
                        self.ax4.c = np.zeros(self._max_episode_steps)
                        self.ax6.c = np.zeros(self._max_episode_steps)
                    elif self.DIAL:
                        self.ax4.c = np.zeros((self.DIAL_n_comms, self._max_episode_steps))
                        self.ax6.c = np.zeros((self.DIAL_n_comms, self._max_episode_steps))
                else:
                    s_multi = self._get_state_multi()
                    self.ax3.s[:, self.step_cnt] = s_multi[0]
                    self.ax5.s[:, self.step_cnt] = s_multi[1]

                    if self.RIAL:
                        self.ax4.c[self.step_cnt] = self.comm_tm1[0]
                        self.ax6.c[self.step_cnt] = self.comm_tm1[1]
                    elif self.DIAL:
                        self.ax4.c[:, self.step_cnt] = self.comm_tm1[0]
                        self.ax6.c[:, self.step_cnt] = self.comm_tm1[1]

            # periodically clear and init
            if self.step_cnt % 50 == 0:

                # clearance
                self.ax1.clear()
                if self.plot_reward:
                    self.ax2.clear()
                if self.plot_state:
                    [ax.clear() for ax in [self.ax3, self.ax4, self.ax5, self.ax6]]

                # appearance
                self.ax1.set_title("Simple Communication Env")
                self.ax1.set_xlabel("x", labelpad=-2)
                self.ax1.set_xlim(-12, 12)
                self.ax1.set_ylim(-0.5, 0.5)

                if self.plot_reward:
                    self.ax2.set_xlabel("Timestep", labelpad=-2)
                    self.ax2.set_ylabel("Reward")
                    self.ax2.set_xlim(0, 50*(np.ceil(self.step_cnt/50)+1))
                    self.ax2.set_ylim(-6, 6)

                if self.plot_state:
                    self.ax3.set_xlabel("Timestep", labelpad=-2)
                    self.ax3.set_ylabel("State of ID0")
                    self.ax3.set_xlim(0, 50*(np.ceil(self.step_cnt/50)+1))
                    self.ax3.set_ylim(-5, 5)

                    self.ax5.set_xlabel("Timestep", labelpad=-2)
                    self.ax5.set_ylabel("State of ID1")
                    self.ax5.set_xlim(0, 50*(np.ceil(self.step_cnt/50)+1))
                    self.ax5.set_ylim(-5, 5)

                    if self.RIAL or self.DIAL:
                        self.ax4.set_xlabel("Timestep", labelpad=-2)
                        self.ax4.set_ylabel("Sent comm of ID0 at t-1")
                        self.ax4.set_xlim(0, 50*(np.ceil(self.step_cnt/50)+1))
                        self.ax4.set_ylim(-2, 4)

                        self.ax6.set_xlabel("Timestep", labelpad=-2)
                        self.ax6.set_ylabel("Sent comm of ID1 at t-1")
                        self.ax6.set_xlim(0, 50*(np.ceil(self.step_cnt/50)+1))
                        self.ax6.set_ylim(-2, 4)

                # ---------- animated artists: initial drawing ---------
                # step info
                self.ax1.info_txt = self.ax1.text(x=-7, y=0.3, s="", fontdict={"size" : 10}, animated=True)

                # destination
                self.ax1.dest_ln = self.ax1.plot([-2, 2], [0, 0], color=self.dest.color, animated=True)[0]

                # vehicles
                self.ax1.scs  = []
                self.ax1.txts = []
                for i, _ in enumerate(self.vehicles):
                    # vehicle
                    self.ax1.scs.append(self.ax1.scatter([], [], color=COLORS[i], animated=True))
                    
                    # information
                    self.ax1.txts.append(self.ax1.text(x=0.0, y=0.0, s="", color=COLORS[i], fontdict={"size" : 10}, animated=True))

                if self.plot_reward:
                    self.ax2.lns_agg  = []
                    self.ax2.lns_agg.append(self.ax2.plot([], [], color=COLORS[0], label=f"ID0", animated=True)[0])
                    self.ax2.lns_agg.append(self.ax2.plot([], [], color=COLORS[1], label=f"ID1", animated=True)[0])
                    self.ax2.legend()

                if self.plot_state:
                    self.ax3.lns = []
                    for i in range(self.obs_size):
                        self.ax3.lns.append(self.ax3.plot([], [], label=self.obs_names[i], color=COLORS[i], animated=True)[0])
                    self.ax3.legend()

                    self.ax5.lns = []
                    for i in range(self.obs_size):
                        self.ax5.lns.append(self.ax5.plot([], [], label=self.obs_names[i], color=COLORS[i], animated=True)[0])
                    self.ax5.legend()

                    if self.RIAL:
                        self.ax4.lns = [self.ax4.plot([], [], color=COLORS[0], animated=True)[0]]
                        self.ax6.lns = [self.ax6.plot([], [], color=COLORS[1], animated=True)[0]]
                    elif self.DIAL:
                        self.ax4.lns = []
                        for i in range(self.DIAL_n_comms):
                            self.ax4.lns.append(self.ax4.plot([], [], color=COLORS[i], animated=True, label=self.comm_names[i])[0])
                        self.ax4.legend()

                        self.ax6.lns = []
                        for i in range(self.DIAL_n_comms):
                            self.ax6.lns.append(self.ax6.plot([], [], color=COLORS[i], animated=True, label=self.comm_names[i])[0])
                        self.ax6.legend()

                # ----------------- store background -------------------
                self.f.canvas.draw()
                self.ax1.bg = self.f.canvas.copy_from_bbox(self.ax1.bbox)
                if self.plot_reward:
                    self.ax2.bg = self.f.canvas.copy_from_bbox(self.ax2.bbox)
                if self.plot_state:
                    self.ax3.bg = self.f.canvas.copy_from_bbox(self.ax3.bbox)
                    self.ax4.bg = self.f.canvas.copy_from_bbox(self.ax4.bbox)
                    self.ax5.bg = self.f.canvas.copy_from_bbox(self.ax5.bbox)
                    self.ax6.bg = self.f.canvas.copy_from_bbox(self.ax6.bbox)
            else:
                # ------------- restore the background ---------------
                self.f.canvas.restore_region(self.ax1.bg)
                if self.plot_reward:
                    self.f.canvas.restore_region(self.ax2.bg)
                if self.plot_state:
                    self.f.canvas.restore_region(self.ax3.bg)
                    self.f.canvas.restore_region(self.ax4.bg)
                    self.f.canvas.restore_region(self.ax5.bg)
                    self.f.canvas.restore_region(self.ax6.bg)

                # ----------- animated artists: update ---------------
                # step info
                self.ax1.info_txt.set_text(self.__str__())
                self.ax1.draw_artist(self.ax1.info_txt)

                # destination
                self.ax1.dest_ln.set_color(self.dest.color)
                self.ax1.draw_artist(self.ax1.dest_ln)

                # show vehicle
                for i, p in enumerate(self.vehicles):    
                    self.ax1.scs[i].set_offsets(np.array([p.x, 0]))
                    self.ax1.draw_artist(self.ax1.scs[i])

                    # information
                    s = f"id: {i}" + "\n" + (f"t alive: {int(self.ts_alive[i])}" if self.game_setup == "alternate" else f"coin: {int(self.coins[i])}")
                    self.ax1.txts[i].set_text(s)
                    self.ax1.txts[i].set_position((p.x-2, 0.1))
                    self.ax1.draw_artist(self.ax1.txts[i])

                # reward
                if self.plot_reward:
                    self.ax2.lns_agg[0].set_data(np.arange(self.step_cnt+1), self.ax2.r0[:self.step_cnt+1])
                    self.ax2.lns_agg[1].set_data(np.arange(self.step_cnt+1), self.ax2.r1[:self.step_cnt+1])

                    self.ax2.draw_artist(self.ax2.lns_agg[0])                    
                    self.ax2.draw_artist(self.ax2.lns_agg[1])

                # state
                if self.plot_state:
                    # state 0
                    for i in range(self.obs_size):
                        self.ax3.lns[i].set_data(np.arange(self.step_cnt+1), self.ax3.s[i][:self.step_cnt+1])
                        self.ax3.draw_artist(self.ax3.lns[i])

                    # state 1
                    for i in range(self.obs_size):
                        self.ax5.lns[i].set_data(np.arange(self.step_cnt+1), self.ax5.s[i][:self.step_cnt+1])
                        self.ax5.draw_artist(self.ax5.lns[i])

                    # comm 0 and 1
                    if self.RIAL:
                        self.ax4.lns[0].set_data(np.arange(self.step_cnt+1), self.ax4.c[:self.step_cnt+1])
                        self.ax4.draw_artist(self.ax4.lns[0])

                        self.ax6.lns[0].set_data(np.arange(self.step_cnt+1), self.ax6.c[:self.step_cnt+1])
                        self.ax6.draw_artist(self.ax6.lns[0])

                    elif self.DIAL:
                        for i in range(self.DIAL_n_comms):
                            self.ax4.lns[i].set_data(np.arange(self.step_cnt+1), self.ax4.c[i, :self.step_cnt+1])
                            self.ax4.draw_artist(self.ax4.lns[i])

                            self.ax6.lns[i].set_data(np.arange(self.step_cnt+1), self.ax6.c[i, :self.step_cnt+1])
                            self.ax6.draw_artist(self.ax6.lns[i])

                # show it on screen
                self.f.canvas.blit(self.ax1.bbox)
                if self.plot_reward:
                    self.f.canvas.blit(self.ax2.bbox)
                if self.plot_state:
                    self.f.canvas.blit(self.ax3.bbox)
                    self.f.canvas.blit(self.ax4.bbox)
                    self.f.canvas.blit(self.ax5.bbox)
                    self.f.canvas.blit(self.ax6.bbox)
               
            plt.pause(0.25)
