from copy import copy
from string import ascii_letters
from typing import List

import gym
from bluesky.tools.geo import latlondist, qdrdist, qdrpos
from gym import spaces
from matplotlib import pyplot as plt
from mycolorpy import colorlist as mcp

from tud_rl.agents.base import _Agent
from tud_rl.envs._envs.HHOS_Fnc import to_utm
from tud_rl.envs._envs.Plane import *
from tud_rl.envs._envs.VesselFnc import (NM_to_meter, angle_to_2pi,
                                         angle_to_pi, dtr, meter_to_NM)

COLORS = [plt.rcParams["axes.prop_cycle"].by_key()["color"][i] for i in range(8)] + 5 * mcp.gen_color(cmap="tab20b", n=20) 


class Destination:
    def __init__(self, dt) -> None:
        # size
        self.radius = 100          # [m]
        self.spawn_radius   = 1100 # [m]
        self.respawn_radius = 1300 # [m]
        
        # position
        self.lat = 10  # [deg]
        self.lon = 10  # [deg]
        self.N, self.E, _ = to_utm(self.lat, self.lon) # [m], [m]

        # timing
        self.dt = dt          # [s], simulation time step
        self._t_close = 60    # [s], time the destination is closed after an aircraft has entered 
        self._t_nxt_open = 0  # [s], current time until the destination opens again
        self._was_open = True
        self.open()

    def reset(self):
        self.open()

    def step(self, planes: List[Plane]):
        """Updates status of the destination.
        Returns:
            np.ndarray([number_of_planes,]): who entered a closed destination
            np.ndarray([number_of_planes,]): who entered an open destination"""
        # count time until next opening
        if self._is_open is False:
            self._t_nxt_open -= self.dt
            if self._t_nxt_open <= 0:
                self.open()

        # store opening status
        self._was_open = copy(self._is_open)

        # check who entered a closed or open destination
        entered_close = np.zeros(len(planes), dtype=bool)
        entered_open  = np.zeros(len(planes), dtype=bool)

        for i, p in enumerate(planes):
            if p.D_dest <= self.radius: 

                # close if someone enters             
                if self._is_open:
                    entered_open[i] = True
                    self.close()
                else:
                    entered_close[i] = True
        return entered_close, entered_open

    def open(self):
        self._t_nxt_open = 0
        self._is_open = True
        self.color = "green"
    
    def close(self):
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
    def is_open(self):
        return self._is_open

    @property
    def was_open(self):
        return self._was_open


class UAM(gym.Env):
    """Urban air mobility simulation env based on the BlueSky simulator of Ellerbroek and Hoekstra.
    Note: If multi_agent is True, each plane is considered an agent. Otherwise, the first plane operates as a single agent."""
    def __init__(self, N_agents=None, multi_agent=None):
        super(UAM, self).__init__()

        # setup
        self.N_agents = N_agents
        assert N_agents > 1, "Need at least two aircrafts."

        self.multi_agent  = multi_agent
        self.acalt = 300 # [m]
        self.actas = 15  # [m/s]
        self.actype = "MAVIC"

        if not multi_agent:
            self.history_length = 2

        # domain params
        self.LoS_dist = 100 # [m]
        self.clock_degs = np.linspace(0.0, 360.0, num=100, endpoint=True)

        # destination
        self.dt = 1.0
        self.dest = Destination(self.dt)

        # performance model
        self.perf = OpenAP(self.actype, self.actas, self.acalt)

        # config
        self.obs_per_TS = 4
        self.obs_size   = 3 + self.obs_per_TS*(self.N_agents-1)

        self.observation_space = spaces.Box(low  = np.full(self.obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(self.obs_size,  np.inf, dtype=np.float32))
        self.act_size = 1
        self.action_space = spaces.Box(low  = np.full(self.act_size, -1.0, dtype=np.float32), 
                                       high = np.full(self.act_size, +1.0, dtype=np.float32))
        self._max_episode_steps = 250

        # viz
        self.plot_reward = True
        self.plot_state  = False

        #self.r = np.zeros((self.N_planes_max, 1))
        #self.state = np.zeros((self.N_planes_max, self.obs_size))

        atts = ["D_TS", "bng_TS", "V_R", "C_T"]
        other_names = []
        for i in range(self.N_agents-1):
            others = [ele + ascii_letters[i] for ele in atts]
            other_names += others
        self.obs_names = ["bng_goal", "D_goal", "t_close"] + other_names

    def reset(self):
        """Resets environment to initial state."""
        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # create some aircrafts
        self.planes:List[Plane] = []

        if self.multi_agent:
            self.N_planes = self.N_agents
        else:
            self.N_planes = np.random.choice(np.arange(2, self.N_agents+1))

        for n in range(self.N_planes):
            self.planes.append(self._spawn_plane(n))

        # reset dest
        self.dest.reset()

        # init state
        self._set_state()
        self.state_init = self.state
        return self.state
  
    def _spawn_plane(self, n:int=None):
        """Spawns the n-th plane. Currently, the argument is not in use but might be relevant for some validation scenarios."""
        # sample heading and speed
        hdg = float(np.random.uniform(0.0, 360.0, size=1))
        tas = float(np.random.uniform(self.actas-3.0, self.actas+3.0, size=1))
        
        # determine origin
        lat, lon = qdrpos(latd1=self.dest.lat, lond1=self.dest.lon, qdr=hdg, dist=meter_to_NM(self.dest.spawn_radius))

        # consider behavior type
        if self.multi_agent:
            role = "RL"
        else:
            if n == 0:
                role = "RL"
            else:
                role = np.random.choice(["RL", "VFG"], size=1)[0]
        p = Plane(role=role, dt=self.dt, actype=self.actype, lat=lat, lon=lon, alt=self.acalt, hdg=(hdg+180)%360, tas=tas)   

        # compute initial distance to destination
        p.D_dest       = latlondist(latd1=self.dest.lat, lond1=self.dest.lon, latd2=lat, lond2=lon)
        p.D_dest_old   = copy(p.D_dest)
        p.just_spawned = True
        return p

    def _set_state(self):
        # usual state of shape [N_planes, obs_size]
        if self.multi_agent:
            self.state = self._get_state_multi()
        
        # since we use LSTMRecTD3, we need multi-agent history as well
        else:
            self.state = self._get_state(0)

            if self.step_cnt == 0:
                self.s_multi_hist = np.zeros((self.history_length, self.N_planes, self.obs_size))
                self.a_multi_hist = np.zeros((self.history_length, self.N_planes, self.act_size))
                self.hist_len = 0
                self.s_multi_old = np.zeros((self.N_planes, self.obs_size))
            else:
                # update history, where most recent state component is the old state from last step
                if self.hist_len == self.history_length:
                    self.s_multi_hist = np.roll(self.s_multi_hist, shift=-1, axis=0)
                    self.s_multi_hist[self.history_length - 1] = self.s_multi_old
                else:
                    self.s_multi_hist[self.hist_len] = self.s_multi_old
                    self.hist_len += 1

            # overwrite old state
            self.s_multi_old = self._get_state_multi()

    def _get_state_multi(self) -> None:
        """Computes the state in the multi-agent scenario."""
        s = np.zeros((self.N_planes, self.obs_size), dtype=np.float32)
        for i, _ in enumerate(self.planes):
            s[i] = self._get_state(i)
        return s

    def _get_state(self, i:int) -> np.ndarray:
        """Computes the state from the perspective of the i-th agent of the internal plane array.
        
        This is a np.array of size [3 + 4*(N_planes-1),] containing own relative bearing of goal, distance to goal, 
        common four information about target ships (relative speed, relative bearing, distance, heading intersection angle),
        and time until destination opens again."""

        # select plane of interest
        p = self.planes[i]

        # distance, bearing to goal and time to opening
        abs_bng_goal, d_goal = qdrdist(latd1=p.lat, lond1=p.lon, latd2=self.dest.lat, lond2=self.dest.lon) # outputs ABSOLUTE bearing
        bng_goal = angle_to_pi(angle_to_2pi(dtr(abs_bng_goal)) - dtr(p.hdg))
        s_i = np.array([bng_goal/np.pi,\
                        NM_to_meter(d_goal)/self.dest.spawn_radius,\
                        1.0-self.dest.t_nxt_open/self.dest.t_close])

        # information about other planes
        if self.N_planes > 1:
            TS_info = []
            for other in self.planes:
                if p is not other:
                    # relative speed
                    v_r = other.tas - p.tas

                    # bearing and distance
                    abs_bng, d = qdrdist(latd1=p.lat, lond1=p.lon, latd2=other.lat, lond2=other.lon)
                    bng = angle_to_pi(angle_to_2pi(dtr(abs_bng)) - dtr(p.hdg))/np.pi
                    d = NM_to_meter(d)/self.dest.spawn_radius

                    # heading intersection
                    C_T = angle_to_pi(np.radians(other.hdg - p.hdg))/np.pi

                    TS_info.append([d, bng, v_r, C_T])

            # sort array according to distance
            TS_info = np.hstack(sorted(TS_info, key=lambda x: x[0], reverse=True)).astype(np.float32)

            # ghost ship padding not needed since we always demand at least two planes
            # however, we need to pad NA's as usual in single-agent LSTMRecTD3
            if not self.multi_agent:
                desired_length = self.obs_per_TS * (self.N_agents-1)
                TS_info = np.pad(TS_info, (0, desired_length - len(TS_info)), 'constant', constant_values=np.nan).astype(np.float32)

            s_i = np.concatenate((s_i, TS_info))
        return s_i

    def step(self, a):
        """a is np.array([N_planes, action_dim]), where action_dim is 2 or 1, in multi-agent scenarios.
        In single-agent, it is a list:
            [np.array([action_dim,]), _agent]."""
        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += self.dt
 
        # fly all planes in multi-agent situation
        if self.multi_agent:
            [p.upd_dynamics(a=a[i], perf=self.perf, dest=None) for i, p in enumerate(self.planes)]

        # in single-agent situation, action corresponds to first plane, while the others are either RL or VFG.
        else:
            # greedy
            cnt_agent = a[1]
            prior_mode = copy(cnt_agent.mode)
            cnt_agent.mode = "test"

            # collect states from planes
            states_multi = self._get_state_multi()

            for i, p in enumerate(self.planes):

                # fly first agent-plane based on given action
                if i == 0:
                    p.upd_dynamics(a=a[0], perf=self.perf, dest=None)

                # fly remaining planes, depending on whether they are RL- or VFG-controlled
                else:
                    if p.role == "RL":
                        p.upd_dynamics(a=cnt_agent.select_action(s        = states_multi[i], 
                                                                 s_hist   = self.s_multi_hist[:, i, :], 
                                                                 a_hist   = self.a_multi_hist[:, i, :], 
                                                                 hist_len = self.hist_len),
                                       perf=self.perf, dest=None)
                    elif p.role == "VFG":
                        p.upd_dynamics(a=None, perf=self.perf, dest=None)
            
            # back to prior mode
            cnt_agent.mode = copy(prior_mode)

        # update distances to destination
        for p in self.planes:
            p.D_dest_old = copy(p.D_dest)
            p.D_dest = latlondist(latd1=p.lat, lond1=p.lon, latd2=self.dest.lat, lond2=self.dest.lon)

        # check destination entries
        entered_close, entered_open = self.dest.step(self.planes)

        # respawning
        self._handle_respawn(entered_open)

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward(entered_close, entered_open)
        d = self._done()
        return self.state, self.r, d, {}
 
    def _handle_respawn(self, respawn_flags):
        """Respawns planes when they entered the open destination area or are at the outer simulation radius."""
        for i, p in enumerate(self.planes):
            if (p.D_dest >= self.dest.respawn_radius) or respawn_flags[i]:
                self.planes[i] = self._spawn_plane(i)
            else:
                p.just_spawned = False

    def _calculate_reward(self, entered_close:np.ndarray, entered_open:np.ndarray):
        """Args:
            entered_close: np.ndarray([number_of_planes,]): who entered a closed destination
            entered_open:  np.ndarray([number_of_planes,]): who entered an open destination"""
        # empty array
        r = np.zeros((self.N_planes, 1), dtype=np.float32)

        # ---------------- individual reward -------------------
        for i, pi in enumerate(self.planes):

            # consider only first reward in single-agent situation
            if (not self.multi_agent) and (i != 0):
                continue

            # collision
            for j, pj in enumerate(self.planes):
                if i != j:
                    D = latlondist(latd1=pi.lat, lond1=pi.lon, latd2=pj.lat, lond2=pj.lon)-self.LoS_dist
                    if D <= 0:
                        r[i] -= 5.0
                    else:
                        r[i] -= np.exp(-D/self.LoS_dist)
            
            # off-map (+5 agains numerical issues)
            if pi.D_dest > (self.dest.spawn_radius + 5.0): 
                r[i] -= 5.0
            
            # closed goal entering
            if entered_close[i]:
                r[i] -= 5.0

            # open goal entering
            if entered_open[i]:
                r[i] += 5.0

            # open goal approaching
            if self.dest.was_open:
                if (not entered_close[i]) and (not entered_open[i]):
                    r[i] += (pi.D_dest_old - pi.D_dest)/15.0

        # ---------------- collective reward -------------------
        if False:
            if self.plain_reward:
                if self.dest.is_open:
                    r += -1.0
                else:
                    r += 1.0
            else:            
                # collective reward: open goal entering
                if any(entered_open):
                    r += 10.0

                # collective reward: progress to open goal
                if self.dest.was_open:
                    deltas = [0.0]
                    for i, p in enumerate(self.planes):
                        if (not entered_close[i]) and (not entered_open[i]):
                            deltas.append(p.D_dest_old - p.D_dest)
                    r += max(deltas)/100.0

        # consider only first reward in single-agent situation
        if self.multi_agent:
            self.r = r
        else:
            self.r = float(r[0])

    def _done(self):
        # artificial done signal
        if self.step_cnt >= self._max_episode_steps:
            return True
        return False

    def __str__(self):
        return f"Step: {self.step_cnt}, Sim-Time [s]: {int(self.sim_t)}, Time-to-open [s]: {int(self.dest.t_nxt_open)}"

    def render(self, mode=None):
        """Renders the current environment."""

        # plot every nth timestep
        if self.step_cnt % 1 == 0: 
            
            # init figure
            if len(plt.get_fignums()) == 0:
                if self.plot_reward and self.plot_state:
                    self.f = plt.figure(figsize=(14, 8))
                    self.gs  = self.f.add_gridspec(2, 2)
                    self.ax1 = self.f.add_subplot(self.gs[:, 0]) # ship
                    self.ax2 = self.f.add_subplot(self.gs[0, 1]) # reward
                    self.ax3 = self.f.add_subplot(self.gs[1, 1]) # state

                elif self.plot_reward:
                    self.f = plt.figure(figsize=(14, 8))
                    self.gs  = self.f.add_gridspec(1, 2)
                    self.ax1 = self.f.add_subplot(self.gs[0, 0]) # ship
                    self.ax2 = self.f.add_subplot(self.gs[0, 1]) # reward

                elif self.plot_state:
                    self.f = plt.figure(figsize=(14, 8))
                    self.gs  = self.f.add_gridspec(1, 2)
                    self.ax1 = self.f.add_subplot(self.gs[0, 0]) # ship
                    self.ax3 = self.f.add_subplot(self.gs[0, 1]) # state

                else:
                    self.f, self.ax1 = plt.subplots(1, 1, figsize=(10, 10))
                plt.ion()
                plt.show()           

            # storage
            if self.plot_reward:
                if self.step_cnt == 0:
                    if self.multi_agent:
                        self.ax2.r = np.zeros((self.N_planes, self._max_episode_steps))
                    else:
                        self.ax2.r = np.zeros(self._max_episode_steps)
                else:
                    if self.multi_agent:
                        self.ax2.r[:, self.step_cnt] = self.r.flatten()
                    else:
                        self.ax2.r[self.step_cnt] = self.r

            if self.plot_state:
                if self.step_cnt == 0:
                    self.ax3.s = np.zeros((self.obs_size, self._max_episode_steps))
                else:
                    if self.multi_agent:
                        self.ax3.s[:, self.step_cnt] = self.state[0]
                    else:
                        self.ax3.s[:, self.step_cnt] = self.state

            # periodically clear and init
            if self.step_cnt % 50 == 0:

                # clearance
                self.ax1.clear()
                if self.plot_reward:
                    self.ax2.clear()
                if self.plot_state:
                    self.ax3.clear()

                # appearance
                self.ax1.set_title("Urban Air Mobility")
                self.ax1.set_xlabel("Lon [°]")
                self.ax1.set_ylabel("Lat [°]")
                self.ax1.set_xlim(9.985, 10.015)
                self.ax1.set_ylim(9.985, 10.015)

                if self.plot_reward:
                    self.ax2.set_xlabel("Timestep in episode")
                    self.ax2.set_ylabel("Reward")
                    self.ax2.set_xlim(0, 50*(np.ceil(self.step_cnt/50)+1))
                    self.ax2.set_ylim(-10, 10)

                if self.plot_state:
                    self.ax3.set_xlabel("Timestep in episode")
                    self.ax3.set_ylabel("State of Agent 0")
                    self.ax3.set_xlim(0, 50*(np.ceil(self.step_cnt/50)+1))
                    self.ax3.set_ylim(-2, 2)

                # ---------------- non-animated artists ----------------
                # spawning area
                lats, lons = map(list, zip(*[qdrpos(latd1=self.dest.lat, lond1=self.dest.lon, qdr=deg, dist=meter_to_NM(self.dest.spawn_radius))\
                    for deg in self.clock_degs]))
                self.ax1.plot(lons, lats, color="grey")

                # respawn area
                lats, lons = map(list, zip(*[qdrpos(latd1=self.dest.lat, lond1=self.dest.lon, qdr=deg, dist=meter_to_NM(self.dest.respawn_radius))\
                    for deg in self.clock_degs]))
                self.ax1.plot(lons, lats, color="black")

                # ---------- animated artists: initial drawing ---------
                # step info
                self.ax1.info_txt = self.ax1.text(x=9.9925, y=10.0125, s="", fontdict={"size" : 9}, animated=True)

                # destination
                lats, lons = map(list, zip(*[qdrpos(latd1=self.dest.lat, lond1=self.dest.lon, qdr=deg, dist=meter_to_NM(self.dest.radius))\
                    for deg in self.clock_degs]))
                self.ax1.dest_ln = self.ax1.plot(lons, lats, color=self.dest.color, animated=True)[0]

                # aircraft information
                self.ax1.scs  = []
                self.ax1.lns  = []
                self.ax1.txts = []

                for i, p in enumerate(self.planes):

                    # show aircraft
                    self.ax1.scs.append(self.ax1.scatter([], [], marker=(3, 0, -p.hdg), color=COLORS[i], animated=True))

                    # LoS area
                    self.ax1.lns.append(self.ax1.plot([], [], color=COLORS[i], animated=True)[0])

                    # information
                    self.ax1.txts.append(self.ax1.text(x=0.0, y=0.0, s="", color=COLORS[i], fontdict={"size" : 8}, animated=True))

                if self.plot_reward:
                    self.ax2.lns = []
                    if self.multi_agent:
                        for i in range(self.N_planes):
                            self.ax2.lns.append(self.ax2.plot([], [], color=COLORS[i], label=f"r {i}", animated=True)[0])
                    else:
                        self.ax2.lns.append(self.ax2.plot([], [], color=COLORS[0], label="reward", animated=True)[0])
                    self.ax2.legend()

                if self.plot_state:
                    self.ax3.lns = []
                    for i in range(self.obs_size):
                        self.ax3.lns.append(self.ax3.plot([], [], label=self.obs_names[i], color=COLORS[i], animated=True)[0])
                    self.ax3.legend()

                # ----------------- store background -------------------
                self.f.canvas.draw()
                self.ax1.bg = self.f.canvas.copy_from_bbox(self.ax1.bbox)
                if self.plot_reward:
                    self.ax2.bg = self.f.canvas.copy_from_bbox(self.ax2.bbox)
                if self.plot_state:
                    self.ax3.bg = self.f.canvas.copy_from_bbox(self.ax3.bbox)
            else:

                # ------------- restore the background ---------------
                self.f.canvas.restore_region(self.ax1.bg)
                if self.plot_reward:
                    self.f.canvas.restore_region(self.ax2.bg)
                if self.plot_state:
                    self.f.canvas.restore_region(self.ax3.bg)

                # ----------- animated artists: update ---------------
                # step info
                self.ax1.info_txt.set_text(self.__str__())
                self.ax1.draw_artist(self.ax1.info_txt)

                # destination
                self.ax1.dest_ln.set_color(self.dest.color)
                self.ax1.draw_artist(self.ax1.dest_ln)

                for i, p in enumerate(self.planes):

                    # show aircraft
                    self.ax1.scs[i].set_offsets(np.array([p.lon, p.lat]))
                    self.ax1.draw_artist(self.ax1.scs[i])

                    # LoS area
                    lats, lons = map(list, zip(*[qdrpos(latd1=p.lat, lond1=p.lon, qdr=deg, dist=meter_to_NM(self.LoS_dist))\
                        for deg in self.clock_degs]))
                    self.ax1.lns[i].set_data(lons, lats) 
                    self.ax1.draw_artist(self.ax1.lns[i])

                    # information
                    s = f"id: {i}" + "\n" + f"hdg: {p.hdg:.1f}" + "\n" + f"alt: {p.alt:.1f}" + "\n" + f"tas: {p.tas:.1f}"
                    if hasattr(p, "role"):
                        s += "\n" + f"role: {p.role}"
                    self.ax1.txts[i].set_text(s)
                    self.ax1.txts[i].set_position((p.lon, p.lat))
                    self.ax1.draw_artist(self.ax1.txts[i])

                # reward
                if self.plot_reward:
                    if self.multi_agent:
                        for i in range(self.N_planes):
                            self.ax2.lns[i].set_data(np.arange(self.step_cnt+1), self.ax2.r[i][:self.step_cnt+1])
                            self.ax2.draw_artist(self.ax2.lns[i])
                    else:
                        self.ax2.lns[0].set_data(np.arange(self.step_cnt+1), self.ax2.r[:self.step_cnt+1])
                        self.ax2.draw_artist(self.ax2.lns[0])

                # state
                if self.plot_state:
                    for i in range(self.obs_size):
                        self.ax3.lns[i].set_data(np.arange(self.step_cnt+1), self.ax3.s[i][:self.step_cnt+1])
                        self.ax3.draw_artist(self.ax3.lns[i])

                # show it on screen
                self.f.canvas.blit(self.ax1.bbox)
                if self.plot_reward:
                    self.f.canvas.blit(self.ax2.bbox)
                if self.plot_state:
                    self.f.canvas.blit(self.ax3.bbox)
               
            plt.pause(0.005)
