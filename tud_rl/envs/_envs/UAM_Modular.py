import random
from copy import copy
from string import ascii_letters
from typing import List, Union

import gym
from bluesky.tools.geo import latlondist, qdrdist, qdrpos
from gym import spaces
from matplotlib import pyplot as plt
from mycolorpy import colorlist as mcp

from tud_rl.agents.base import _Agent
from tud_rl.envs._envs.HHOS_Fnc import VFG, get_init_two_wp, to_utm
from tud_rl.envs._envs.Plane import *
from tud_rl.envs._envs.VesselFnc import (NM_to_meter, angle_to_2pi,
                                         angle_to_pi, cpa, dtr, meter_to_NM)

COLORS = [plt.rcParams["axes.prop_cycle"].by_key()["color"][i] for i in range(8)] + 5 * mcp.gen_color(cmap="tab20b", n=20) 


class Destination:
    def __init__(self, dt) -> None:
        # size
        self.radius = 100          # [m]
        self.spawn_radius   = 1100 # [m]
        self.respawn_radius = 1300 # [m]
        
        # position
        self.lat = 60  # [deg]
        self.lon = 9  # [deg]
        self.N, self.E, _ = to_utm(self.lat, self.lon) # [m], [m]

        # timing
        self.dt = dt             # [s], simulation time step
        self._t_close = 60       # [s], time the destination is closed after an aircraft has entered 
        self._t_nxt_open = 0     # [s], current time until the destination opens again
        self._t_open_since = 0   # [s], current time since the vertiport is open
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
        else:
            self._t_open_since += self.dt

        # store opening status
        self._was_open = copy(self._is_open)

        # check who entered a closed or open destination
        entered_close = np.zeros(len(planes), dtype=bool)
        entered_open  = np.zeros(len(planes), dtype=bool)

        for i, p in enumerate(planes):
            if p.D_dest <= self.radius:            
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


class Path:
    """Defines a path for flight taxis."""
    def __init__(self, lat=None, lon=None, north=None, east=None) -> None:
        # checks
        assert not ((lat is None and lon is None) and (north is None and east is None)), "Need some data to construct the path."

        # UTM and lat-lon
        if (lat is None) or (lon is None):
            self._store(north, "north")
            self._store(east, "east")
            self.lat, self.lon = to_latlon(north=self.north, east=self.east, number=32)
        else:
            self._store(lat, "lat")
            self._store(lon, "lon")

        # UTM coordinates
        if (north is None) or (east is None):
            self.north, self.east, _ = to_utm(lat=self.lat, lon=self.lon)
        else:
            self._store(north, "north")
            self._store(east, "east")

    def _store(self, data:Union[list,np.ndarray], name:str):
        """Stores data by transforming to np.ndarray."""
        if not isinstance(data, np.ndarray):
            setattr(self, name, np.array(data))
        else:
            setattr(self, name, data)

    def _reverse(self):
        """Reverse the path."""
        self.north = np.flip(self.north)
        self.east  = np.flip(self.east)
        self.lat   = np.flip(self.lat)
        self.lon   = np.flip(self.lon)

class UAM_Modular(gym.Env):
    """Urban air mobility simulation env based on the BlueSky simulator of Ellerbroek and Hoekstra.
    Note: If multi_policy is True, each plane is considered an agent. Otherwise, the first plane operates as a single agent."""
    def __init__(self, 
                 N_agents_max :int, 
                 w_coll:float, 
                 w_ye:float,
                 w_ce:float):
        super(UAM_Modular, self).__init__()

        # setup
        self.N_agents_max = N_agents_max
        assert N_agents_max > 1, "Need at least two aircrafts."

        self.acalt = 300 # [m]
        self.actas = 15  # [m/s]
        self.actype = "MAVIC"

        self.VFG_K = 0.01

        self.w_coll = w_coll
        self.w_ye   = w_ye
        self.w_ce   = w_ce
        self.w = self.w_coll + self.w_ye + self.w_ce

        # domain params
        self.incident_dist = 100 # [m]
        self.accident_dist = 10  # [m]
        self.clock_degs = np.linspace(0.0, 360.0, num=100, endpoint=True)

        # destination
        self.dt = 1.0
        self.dest = Destination(self.dt)

        # performance model
        self.perf = OpenAP(self.actype, self.actas, self.acalt)

        # config
        self.history_length = 2
        self.OS_obs     = 2     # loc_ye, loc_course_error
        self.obs_per_TS = 6     # distance, relative bearing, speed difference, heading intersection, DCPA, TCPA
        self.obs_size   = self.OS_obs + self.obs_per_TS*(self.N_agents_max-1)

        self.observation_space = spaces.Box(low  = np.full(self.obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(self.obs_size,  np.inf, dtype=np.float32))
        self.act_size = 1
        self.action_space = spaces.Box(low  = np.full(self.act_size, -1.0, dtype=np.float32), 
                                        high = np.full(self.act_size, +1.0, dtype=np.float32))
        self._max_episode_steps = 200

        # viz
        self.plot_reward = True
        self.plot_state  = False

        assert not (self.plot_state and self.N_agents_max > 2), "State plotting is only reasonable for two flight taxis."
        atts = ["D_TS", "bng_TS", "V_R", "C_T", "DCPA", "TPCA"]

        other_names = []
        for i in range(self.N_agents_max-1):
            others = [ele + ascii_letters[i] for ele in atts]
            other_names += others

        self.obs_names = ["ye", "ce"] + other_names

    def reset(self):
        """Resets environment to initial state."""
        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # set circle radius of RL-controlled flight taxi
        self.main_circle = np.random.uniform(low=self.dest.radius+200, high=self.dest.spawn_radius-200, size=1)

        # create some aircrafts
        self.planes:List[Plane] = []

        self.N_planes = np.random.choice(list(range(2, self.N_agents_max+1)))
        for n in range(self.N_planes):
            if n == 0:
                role = "RL" 
            else:
                role = np.random.choice(["VFG", "RND"], p=[0.75, 0.25])

            self.planes.append(self._spawn_plane(role))

        # interface to high-level module including goal decision and path planning
        self.planes = self._high_level_control(planes=self.planes)

        # set waypoints and compute cross-track and course error
        for p in self.planes:
            p = self._init_wps(p)
            p = self._set_path_metrics(p)

        # reset dest
        self.dest.reset()

        # init state
        self._set_state()
        self.state_init = self.state
        return self.state

    def _spawn_plane(self, role:str):
        assert role in ["RL", "VFG", "RND"], "Unknown role."

        # sample speed and bearing
        tas = float(np.random.uniform(self.actas-5.0, self.actas+5.0, size=1))
        qdr = float(np.random.uniform(0.0, 360.0, size=1))

        # determine origin
        lat, lon = qdrpos(latd1=self.dest.lat, lond1=self.dest.lon, qdr=qdr, dist=meter_to_NM(self.dest.spawn_radius))

        # consider behavior type
        p = Plane(role=role, dt=self.dt, actype=self.actype, lat=lat, lon=lon, alt=self.acalt, hdg=(qdr+180)%360, tas=tas)

        # set UTM coordinates
        p.n, p.e, _ = to_utm(lat=lat, lon=lon)

        # compute initial distance to destination
        p.D_dest     = latlondist(latd1=self.dest.lat, lond1=self.dest.lon, latd2=lat, lond2=lon)
        p.D_dest_old = copy(p.D_dest)

        # set circle distance
        if role == "RL":
            p.circle = self.main_circle
        else:
            p.circle = self.main_circle + np.random.uniform(low=-200, high=200, size=1)

        # set circle direction
        p.clockwise = bool(random.getrandbits(1))
        return p

    def _high_level_control(self, planes:List[Plane]) -> List[Plane]:
        """Decides who out of the current flight taxis should fly toward the goal and plans the paths accordingly.
        This can be the interface to the supervised learning module."""
        # decision
        for p in planes:                
            if p.role == "RL":
                p.fly_to_goal = np.random.choice([True, False], p=[0.25, 0.75]) # bool(random.getrandbits(1))
            else:
                p.fly_to_goal = False
        
        # path planning
        for p in planes:
            p = self._plan_path(p)
        return planes

    def _plan_path(self, p:Plane):
        """Planes a path for a given plane."""
        # linear path to goal
        if p.fly_to_goal:
            ang, _ = qdrdist(latd1=self.dest.lat, lond1=self.dest.lon, latd2=p.lat, lond2=p.lon)
            dist = np.flip(np.linspace(start=0.0, stop=meter_to_NM(self.dest.spawn_radius), num=20))
            reverse = False

        # keep distance at circle
        else:
            ang, _ = qdrdist(latd1=self.dest.lat, lond1=self.dest.lon, latd2=p.lat, lond2=p.lon)
            ang = ang % 360
            ang = (-np.arange(361) + ang) % 360
            dist = meter_to_NM(p.circle)
            reverse = not p.clockwise

        # set path
        lat_path, lon_path = qdrpos(latd1=self.dest.lat, lond1=self.dest.lon, qdr=ang, dist=dist)
        p.Path = Path(lat=lat_path, lon=lon_path)

        # potentially reverse it
        if reverse:
            p.Path._reverse()
        return p

    def _init_wps(self, p:Plane) -> Plane:
        """Initializes the waypoints on the path, respectively, based on the position of the p."""
        p.wp1_idx, p.wp1_N, p.wp1_E, p.wp2_idx, p.wp2_N, p.wp2_E = get_init_two_wp(n_array=p.Path.north, 
                                                                                   e_array=p.Path.east, 
                                                                                   a_n=p.n, a_e=p.e)
        try:
            p.wp3_idx = p.wp2_idx + 1
            p.wp3_N = p.Path.north[p.wp3_idx] 
            p.wp3_E = p.Path.east[p.wp3_idx]
        except:
            p.wp3_idx = p.wp2_idx
            p.wp3_N = p.Path.north[p.wp3_idx] 
            p.wp3_E = p.Path.east[p.wp3_idx]
        return p

    def _set_path_metrics(self, p:Plane) -> Plane:
        p.ye, p.dc, _, _ = VFG(N1 = p.wp1_N, 
                               E1 = p.wp1_E, 
                               N2 = p.wp2_N, 
                               E2 = p.wp2_E,
                               NA = p.n, 
                               EA = p.e, 
                               K  = self.VFG_K, 
                               N3 = p.wp3_N,
                               E3 = p.wp3_E)
        p.ce = angle_to_pi(p.dc - np.radians(p.hdg))
        return p

    def _set_state(self):
        # since we use a spatial-temporal recursive approach, we need multi-agent history as well
        self.state = self._get_state(0)

        if self.step_cnt == 0:
            self.s_multi_hist = np.zeros((self.history_length, self.N_planes, self.obs_size))                   
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
        """Computes the state from the perspective of the i-th agent of the internal plane array."""

        # select plane of interest
        p = self.planes[i]

        # path cross-track and course error
        s_i = np.array([p.ye/50.0, p.ce/np.pi])

        # information about other planes
        if self.N_planes > 1:
            TS_info = []
            for j, other in enumerate(self.planes):
                if i != j:
                    # relative speed
                    v_r = (other.tas - p.tas)/10.0

                    # bearing and distance
                    abs_bng, d = qdrdist(latd1=p.lat, lond1=p.lon, latd2=other.lat, lond2=other.lon)
                    bng = angle_to_pi(angle_to_2pi(dtr(abs_bng)) - dtr(p.hdg))/np.pi
                    d = NM_to_meter(d)/self.dest.spawn_radius

                    # heading intersection
                    C_T = angle_to_pi(np.radians(other.hdg - p.hdg))/np.pi

                    # CPA metrics
                    NOS, EOS, _ = to_utm(lat=p.lat, lon=p.lon)
                    NTS, ETS, _ = to_utm(lat=other.lat, lon=other.lon)
                    DCPA, TCPA = cpa(NOS=NOS, EOS=EOS, NTS=NTS, ETS=ETS, chiOS=np.radians(p.hdg), chiTS=np.radians(other.hdg),
                                     VOS=p.tas, VTS=other.tas)
                    DCPA = DCPA / 100.0
                    TCPA = TCPA / 60.0

                    # aggregate
                    TS_info.append([d, bng, v_r, C_T, DCPA, TCPA])

            # sort array according to distance
            TS_info = np.hstack(sorted(TS_info, key=lambda x: x[0], reverse=True)).astype(np.float32)

            # ghost ship padding not needed since we always demand at least two planes
            # however, we need to pad NA's as usual in single-agent LSTMRecTD3
            desired_length = self.obs_per_TS * (self.N_agents_max-1)
            TS_info = np.pad(TS_info, (0, desired_length - len(TS_info)), 'constant', constant_values=np.nan).astype(np.float32)

            s_i = np.concatenate((s_i, TS_info))
        return s_i

    def step(self, a):
        """Arg a:
        In multi-policy scenarios with continuous actions and no communication:
            np.array([N_planes, action_dim])

        In single-policy:
            _agent"""
        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += self.dt
 
        # in single-policy situation, action corresponds to first plane, while the others are either RL or VFG
        cnt_agent:_Agent = a
        
        # collect states from planes
        states_multi = self._get_state_multi()

        for i, p in enumerate(self.planes):

            # fly planes depending on whether they are RL-, VFG-, or RND-controlled
            if p.role == "RL":

                # spatial-temporal recurrent
                act = cnt_agent.select_action(s        = states_multi[i], 
                                              s_hist   = self.s_multi_hist[:, i, :], 
                                              a_hist   = None, 
                                              hist_len = self.hist_len)

                # move plane
                p.upd_dynamics(a=act, discrete_acts=False, perf=self.perf, dest=None)

            else:
                p.upd_dynamics(perf=self.perf, dest=None)

        # update UTM coordinates
        for p in self.planes:
            p.n, p.e, _ = to_utm(lat=p.lat, lon=p.lon)

        # update distances to destination
        for p in self.planes:
            p.D_dest_old = copy(p.D_dest)
            p.D_dest = latlondist(latd1=p.lat, lond1=p.lon, latd2=self.dest.lat, lond2=self.dest.lon)

        # check destination entries
        _, entered_open = self.dest.step(self.planes)

        # respawning
        self._handle_respawn(entered_open)

        # set waypoints, and compute cross-track and course error
        for p in self.planes:
            p = self._init_wps(p)
            p = self._set_path_metrics(p)

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward()
        d = self._done()
        return self.state, float(self.r[0]), d, {}

    def _handle_respawn(self, entered_open:np.ndarray):
        """Respawns planes when they entered the open destination area or are at the outer simulation radius."""
        had_respawn = False
        for i, p in enumerate(self.planes):
            if (p.D_dest >= self.dest.respawn_radius) or entered_open[i]:
                had_respawn = True
                self.planes[i] = self._spawn_plane(role=self.planes[i].role)
        
        if had_respawn:
            self._high_level_control(planes=self.planes)

    def _calculate_reward(self):
        r_coll = np.zeros((self.N_planes, 1), dtype=np.float32)
        r_ye   = np.zeros((self.N_planes, 1), dtype=np.float32)
        r_ce   = np.zeros((self.N_planes, 1), dtype=np.float32)

        # ------ collision reward ------
        D_matrix = np.ones((len(self.planes), len(self.planes))) * np.inf
        for i, pi in enumerate(self.planes):
            for j, pj in enumerate(self.planes):
                if i != j:
                    D_matrix[i][j] = latlondist(latd1=pi.lat, lond1=pi.lon, latd2=pj.lat, lond2=pj.lon)

        for i, pi in enumerate(self.planes):
            D = float(np.min(D_matrix[i]))

            if D <= self.accident_dist:
                r_coll[i] -= 10.0

            elif D <= self.incident_dist:
                r_coll[i] -= 5.0

            else:
                r_coll[i] -= 1*np.exp(-D/(2*self.incident_dist))

            # off-map (+5 agains numerical issues)
            if pi.D_dest > (self.dest.spawn_radius + 5.0): 
                r_coll[i] -= 5.0
        
        # ------ path reward ------
        for i, pi in enumerate(self.planes):

            # cross-track error
            k_ye = 0.05
            r_ye[i] = np.exp(-k_ye * abs(pi.ye))

            # course error
            k_ce = 5.0
            if abs(rtd(pi.ce)) >= 90.0:
                r_ce[i] = -2.0
            else:
                r_ce[i] = np.exp(-k_ce * abs(pi.ce))

        # aggregate reward components
        r = (self.w_coll*r_coll + self.w_ye*r_ye + self.w_ce*r_ce)/self.w

        # store
        self.r = r
        self.r_coll = r_coll
        self.r_ye   = r_ye
        self.r_ce   = r_ce

    def _done(self):
        # artificial done signal
        if self.step_cnt >= self._max_episode_steps:
            return True
        return False

    def __str__(self):
        return f"Step: {self.step_cnt}, Sim-Time [s]: {int(self.sim_t)}, # Flight Taxis: {self.N_planes}, " +\
            f"Time-to-open [s]: {int(self.dest.t_nxt_open)}" + "\n" + f"Time-since-open[s]: {int(self.dest.t_open_since)}"

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
                    self.ax2.r      = np.zeros(self._max_episode_steps)
                    self.ax2.r_coll = np.zeros(self._max_episode_steps)
                    self.ax2.r_ye   = np.zeros(self._max_episode_steps)
                    self.ax2.r_ce   = np.zeros(self._max_episode_steps)
                else:
                    self.ax2.r[self.step_cnt] = self.r if isinstance(self.r, float) else float(self.r[0])
                    self.ax2.r_coll[self.step_cnt] = self.r_coll if isinstance(self.r_coll, float) else float(self.r_coll[0])
                    self.ax2.r_ye[self.step_cnt] = self.r_ye if isinstance(self.r_ye, float) else float(self.r_ye[0])
                    self.ax2.r_ce[self.step_cnt] = self.r_ce if isinstance(self.r_ce, float) else float(self.r_ce[0])

            if self.plot_state:
                if self.step_cnt == 0:
                    self.ax3.s = np.zeros((self.obs_size, self._max_episode_steps))
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
                self.ax1.set_xlabel("East [m]")
                self.ax1.set_ylabel("North [m]")
                #self.ax1.set_xlim(8.975, 9.025)
                #self.ax1.set_ylim(59.985, 60.015)
                
                #self.ax1.set_xlim(9.985, 10.015)
                #self.ax1.set_ylim(9.985, 10.015)

                if self.plot_reward:
                    self.ax2.set_xlabel("Timestep in episode")
                    self.ax2.set_ylabel("Reward of ID0")
                    self.ax2.set_xlim(0, 50*(np.ceil(self.step_cnt/50)+1))
                    self.ax2.set_ylim(-7, 7)

                if self.plot_state:
                    self.ax3.set_xlabel("Timestep in episode")
                    self.ax3.set_ylabel("State of Agent 0")
                    self.ax3.set_xlim(0, 50*(np.ceil(self.step_cnt/50)+1))
                    self.ax3.set_ylim(-2, 5)

                # ---------------- non-animated artists ----------------
                # spawning area
                lats, lons = map(list, zip(*[qdrpos(latd1=self.dest.lat, lond1=self.dest.lon, qdr=deg, dist=meter_to_NM(self.dest.spawn_radius))\
                    for deg in self.clock_degs]))
                ns, es, _ = to_utm(lat=np.array(lats), lon=np.array(lons))
                self.ax1.plot(es, ns, color="grey")

                # respawn area
                lats, lons = map(list, zip(*[qdrpos(latd1=self.dest.lat, lond1=self.dest.lon, qdr=deg, dist=meter_to_NM(self.dest.respawn_radius))\
                    for deg in self.clock_degs]))
                ns, es, _ = to_utm(lat=np.array(lats), lon=np.array(lons))
                self.ax1.plot(es, ns, color="black")

                # ---------- animated artists: initial drawing ---------
                # step info
                self.ax1.info_txt = self.ax1.text(x=498_700, y=6.6527e6, s="", fontdict={"size" : 9}, animated=True)

                # destination
                lats, lons = map(list, zip(*[qdrpos(latd1=self.dest.lat, lond1=self.dest.lon, qdr=deg, dist=meter_to_NM(self.dest.radius))\
                    for deg in self.clock_degs]))
                ns, es, _ = to_utm(lat=np.array(lats), lon=np.array(lons))
                self.ax1.dest_ln = self.ax1.plot(es, ns, color=self.dest.color, animated=True)[0]

                # aircraft information
                self.ax1.scs   = []
                self.ax1.lns   = []
                self.ax1.paths = []
                self.ax1.pts1  = []
                self.ax1.pts2  = []
                self.ax1.pts3  = []
                self.ax1.txts  = []

                for i, p in enumerate(self.planes):

                    # show aircraft
                    self.ax1.scs.append(self.ax1.scatter([], [], marker=(3, 0, -p.hdg), color=COLORS[i], animated=True))

                    # incident area
                    self.ax1.lns.append(self.ax1.plot([], [], color=COLORS[i], animated=True, zorder=10)[0])

                    # planned paths
                    self.ax1.paths.append(self.ax1.plot([], [], color=COLORS[i], animated=True, zorder=-50)[0])

                    # wps
                    self.ax1.pts1.append(self.ax1.scatter([], [], color=COLORS[i], s=7, animated=True))
                    self.ax1.pts2.append(self.ax1.scatter([], [], color=COLORS[i], s=7, animated=True))
                    self.ax1.pts3.append(self.ax1.scatter([], [], color=COLORS[i], s=7, animated=True))

                    # information
                    self.ax1.txts.append(self.ax1.text(x=0.0, y=0.0, s="", color=COLORS[i], fontdict={"size" : 8}, animated=True))

                if self.plot_reward:
                    self.ax2.lns_agg  = []
                    self.ax2.lns_coll = []
                    self.ax2.lns_ye   = []
                    self.ax2.lns_ce   = []

                    self.ax2.lns_agg.append(self.ax2.plot([], [], color=COLORS[0], label=f"Agg", animated=True)[0])
                    self.ax2.lns_coll.append(self.ax2.plot([], [], color=COLORS[1], label=f"Collision", animated=True)[0])
                    self.ax2.lns_ye.append(self.ax2.plot([], [], color=COLORS[2], label=f"CTE", animated=True)[0])
                    self.ax2.lns_ce.append(self.ax2.plot([], [], color=COLORS[3], label=f"Course", animated=True)[0])

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
                    self.ax1.scs[i].set_offsets(np.array([p.e, p.n]))
                    self.ax1.draw_artist(self.ax1.scs[i])

                    # incident area
                    lats, lons = map(list, zip(*[qdrpos(latd1=p.lat, lond1=p.lon, qdr=deg, dist=meter_to_NM(self.incident_dist/2))\
                        for deg in self.clock_degs]))
                    ns, es, _ = to_utm(lat=np.array(lats), lon=np.array(lons))
                    self.ax1.lns[i].set_data(es, ns) 
                    self.ax1.draw_artist(self.ax1.lns[i])

                    # planned paths
                    self.ax1.paths[i].set_data(p.Path.east, p.Path.north)
                    self.ax1.draw_artist(self.ax1.paths[i])

                    # waypoints
                    self.ax1.pts1[i].set_offsets(np.array([p.Path.east[p.wp1_idx], p.Path.north[p.wp1_idx]]))
                    self.ax1.pts2[i].set_offsets(np.array([p.Path.east[p.wp2_idx], p.Path.north[p.wp2_idx]]))
                    self.ax1.pts3[i].set_offsets(np.array([p.Path.east[p.wp3_idx], p.Path.north[p.wp3_idx]]))
                    self.ax1.draw_artist(self.ax1.pts1[i])
                    self.ax1.draw_artist(self.ax1.pts2[i])
                    self.ax1.draw_artist(self.ax1.pts3[i])

                    # information
                    s = f"id: {i}" + "\n" + f"hdg: {p.hdg:.1f}" + "\n" + f"alt: {p.alt:.1f}" + "\n" + f"tas: {p.tas:.1f}"
                    s+= "\n" + f"ye: {p.ye:.1f}" + "\n" + f"ce: {rtd(p.ce):.1f}"
                    self.ax1.txts[i].set_text(s)
                    self.ax1.txts[i].set_position((p.e, p.n))
                    self.ax1.draw_artist(self.ax1.txts[i])

                # reward
                if self.plot_reward:
                    self.ax2.lns_agg[0].set_data(np.arange(self.step_cnt+1), self.ax2.r[:self.step_cnt+1])
                    self.ax2.lns_coll[0].set_data(np.arange(self.step_cnt+1), self.ax2.r_coll[:self.step_cnt+1])
                    self.ax2.lns_ye[0].set_data(np.arange(self.step_cnt+1), self.ax2.r_ye[:self.step_cnt+1])
                    self.ax2.lns_ce[0].set_data(np.arange(self.step_cnt+1), self.ax2.r_ce[:self.step_cnt+1])
                        
                    self.ax2.draw_artist(self.ax2.lns_agg[0])
                    self.ax2.draw_artist(self.ax2.lns_coll[0])
                    self.ax2.draw_artist(self.ax2.lns_ye[0])
                    self.ax2.draw_artist(self.ax2.lns_ce[0])

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
               
            plt.pause(0.05)
