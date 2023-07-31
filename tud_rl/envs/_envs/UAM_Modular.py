import random
import uuid
from copy import copy
from string import ascii_letters
from typing import List, Union

import gym
from bluesky.tools.geo import qdrpos
from gym import spaces
from matplotlib import pyplot as plt
from mycolorpy import colorlist as mcp

from tud_rl.agents.base import _Agent
from tud_rl.envs._envs.HHOS_Fnc import VFG, get_init_two_wp, to_utm
from tud_rl.envs._envs.Plane import *
from tud_rl.envs._envs.VesselFnc import (ED, NM_to_meter, angle_to_2pi,
                                         angle_to_pi, bng_abs, bng_rel, cpa,
                                         dtr, meter_to_NM, xy_from_polar)

COLORS = [plt.rcParams["axes.prop_cycle"].by_key()["color"][i] for i in range(8)] + 5 * mcp.gen_color(cmap="tab20b", n=20) 


class Destination:
    def __init__(self, dt, c) -> None:
        # size
        self.radius          = c * 200  # [m]
        self.restricted_area = c * 200  # [m]
        self.spawn_radius    = c * 1000 # [m]
        self.respawn_radius  = c * 1200 # [m]
        
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
            np.ndarray([number_of_planes,]): who entered an open destination
            bool: whether destination just openeded again"""
        just_opened = False

        # count time until next opening
        if self._is_open is False:
            self._t_nxt_open -= self.dt
            if self._t_nxt_open <= 0:
                self.open()
                just_opened = True
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

        #  close only if the correct AC entered
        for i, p in enumerate(planes):
            if entered_open[i] and p.fly_to_goal == 1.0:
                self.close()
        return entered_close, entered_open, just_opened

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
    """Urban air mobility simulation env based on the BlueSky simulator of Ellerbroek and Hoekstra."""
    def __init__(self, 
                 N_agents_max :int, 
                 N_cutters_max :int,
                 w_coll:float, 
                 w_goal:float,
                 w_comf:float,
                 r_goal_norm:float,
                 c:float,
                 N_agents_min :int = 1,
                 N_cutters_min : int = 0):
        super(UAM_Modular, self).__init__()

        # setup
        self.N_agents_max  = N_agents_max
        self.N_agents_min  = N_agents_min
        self.N_cutters_max = N_cutters_max
        self.N_cutters_min = N_cutters_min
        
        assert min([N_agents_min, N_agents_max]) >= 1, "Please at least on RL-controlled aircraft."
        #assert 0.5 <= c <= 1.0, "The scaling parameter c should be in [0.5, 1.0]."

        self.acalt = 300 # [m]
        self.actas = 13  # [m/s]
        self.delta_tas = 3 # [m/s]
        self.actype = "MAVIC"

        self.w_coll = w_coll
        self.w_goal = w_goal
        self.w_comf = w_comf
        self.r_goal_norm = r_goal_norm
        self.w = self.w_coll + self.w_goal + self.w_comf
        self.c = c

        # domain params
        self.incident_dist = 100 # [m]
        self.accident_dist = 10  # [m]
        self.clock_degs = np.linspace(0.0, 360.0, num=100, endpoint=True)

        # destination
        self.dt = 1.0
        self.dest = Destination(dt=self.dt, c=c)

        # performance model
        self.perf = OpenAP(self.actype, self.actas, self.acalt)

        # config
        self.history_length = 2
        self.OS_obs     = 3     # abs bng goal, rel bng goal, dist goal, fly to goal
        self.obs_per_TS = 6     # distance, relative bearing, speed difference, heading intersection, DCPA, TCPA
        self.obs_size   = self.OS_obs + self.obs_per_TS*(self.N_agents_max-1) + self.obs_per_TS*self.N_cutters_max

        self.observation_space = spaces.Box(low  = np.full(self.obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(self.obs_size,  np.inf, dtype=np.float32))
        self.act_size = 1
        self.action_space = spaces.Box(low  = np.full(self.act_size, -1.0, dtype=np.float32), 
                                        high = np.full(self.act_size, +1.0, dtype=np.float32))
        self._max_episode_steps = 250

        # viz
        self.plot_reward = True
        self.plot_state  = False

        assert not (self.plot_state and self.N_agents_max > 2), "State plotting is only reasonable for two flight taxis."
        if self.plot_state:
            atts = ["D_TS", "bng_TS", "V_R", "C_T", "DCPA", "TPCA", "Goal"]

            other_names = []
            for i in range(self.N_agents_max-1):
                others = [ele + ascii_letters[i] for ele in atts]
                other_names += others

            self.obs_names = ["bng_goal", "d_goal", "fly to goal"] + other_names

    def reset(self):
        """Resets environment to initial state."""
        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # create some aircrafts
        self.planes:List[Plane] = []
        self.N_RL      = np.random.choice(list(range(self.N_agents_min,  self.N_agents_max+1)))
        self.N_cutters = np.random.choice(list(range(self.N_cutters_min, self.N_cutters_max+1)))
        self.N_planes = self.N_RL + self.N_cutters

        for n in range(self.N_planes):
            if n < self.N_RL:
                role = "RL" 
            else:
                role = "CUT"

            self.planes.append(self._spawn_plane(role))
      
        # interface to high-level module including goal decision
        self._high_level_control()

        # reset dest
        self.dest.reset()

        # init state
        self._set_state()
        self.state_init = self.state
        return self.state

    def _spawn_plane(self, role:str):
        assert role in ["RL", "VFG", "RND", "CUT"], "Unknown role."

        if role == "CUT":
            p0 = self.planes[0]

            while True:
                # sample time
                dt = self.c * random.uniform(30.0, 45.0)

                # linear prediction of p0's path
                vE, vN = xy_from_polar(r=p0.tas, angle=dtr(p0.hdg))
                x0_E = p0.e + vE * dt
                x0_N = p0.n + vN * dt

                # sample angle and speed
                if bool(random.getrandbits(1)):
                    ang = random.uniform(0.0, 2*np.pi)  # all random
                else:
                    ang = angle_to_2pi(dtr(p0.hdg) + np.random.uniform(-np.pi/6, np.pi/6)) # head-on
                tas = random.uniform(self.actas - self.delta_tas, self.actas + self.delta_tas)
                d = dt * tas

                # set position and heading
                E_add, N_add = xy_from_polar(r=d, angle=ang)
                lat, lon = to_latlon(north=x0_N + N_add, east=x0_E + E_add, number=32)
                hdg = (rtd(ang) + 180) % 360

                if ED(N0=x0_N + N_add, E0=x0_E + E_add, N1=self.dest.N, E1=self.dest.E) <= self.dest.respawn_radius:
                    break
        else:
            # sample speed and bearing
            tas = random.uniform(self.actas - self.delta_tas, self.actas + self.delta_tas)
            qdr = random.uniform(0.0, 360.0)

            # determine origin
            E_add, N_add = xy_from_polar(r=self.dest.spawn_radius, angle=dtr(qdr))
            lat, lon = to_latlon(north=self.dest.N+N_add, east=self.dest.E+E_add, number=32)

            # add noise to heading
            hdg = (qdr + 180) % 360
            sgn = 1 if bool(random.getrandbits(1)) else -1
            hdg = (hdg + sgn * random.uniform(20.0, 45.0)) % 360

        # construct plane
        p = Plane(role=role, dt=self.dt, actype=self.actype, lat=lat, lon=lon, alt=self.acalt, hdg=hdg, tas=tas)

        # set UTM coordinates
        p.n, p.e, _ = to_utm(lat=lat, lon=lon)

        # compute initial distance to destination
        p.D_dest     = ED(N0=self.dest.N, E0=self.dest.E, N1=p.n, E1=p.e)
        p.D_dest_old = copy(p.D_dest)
        return p

    def _high_level_control(self):
        """Decides who out of the current flight taxis should fly toward the goal."""
        for i, _ in enumerate(self.planes):
            if i == 0 and self.step_cnt >= 200:
                self.planes[i].fly_to_goal = 1.0
            else:
                self.planes[i].fly_to_goal = -1.0

    def _set_state(self):
        if len(self.planes) == 0:
            self.state = None
            return

        # state observation of id0 will be used for learning
        self.state = self._get_state(0)

        for i, p in enumerate(self.planes):

            # compute current state
            if i == 0:
                p.s = self.state
            else:
                p.s = self._get_state(i)

            # update history
            if not hasattr(p, "s_hist"):
                p.s_hist = np.zeros((self.history_length, self.obs_size))
                p.hist_len = 0
            else:
                if p.hist_len == self.history_length:
                    p.s_hist = np.roll(p.s_hist, shift=-1, axis=0)
                    p.s_hist[self.history_length - 1] = p.s_old
                else:
                    p.s_hist[p.hist_len] = p.s_old
                    p.hist_len += 1
            
            # safe old state
            p.s_old = copy(p.s)

    def _get_state(self, i:int) -> np.ndarray:
        """Computes the state from the perspective of the i-th agent of the internal plane array."""

        # select plane of interest
        p = self.planes[i]

        # relative bearing to goal, distance, fly to goal
        rel_bng_goal = bng_rel(N0=p.n, E0=p.e, N1=self.dest.N, E1=self.dest.E, head0=dtr(p.hdg), to_2pi=False)/np.pi
        d_goal   = ED(N0=p.n, E0=p.e, N1=self.dest.N, E1=self.dest.E)/self.dest.spawn_radius
        task     = p.fly_to_goal
        s_i = np.array([rel_bng_goal, d_goal, task])

        # information about other planes
        TS_info = []
        for j, other in enumerate(self.planes):
            if i != j:
                # relative speed
                v_r = (other.tas - p.tas)/(2*self.delta_tas)

                # relative bearing
                bng = bng_rel(N0=p.n, E0=p.e, N1=other.n, E1=other.e, head0=dtr(p.hdg), to_2pi=False)/np.pi

                # distance
                d = ED(N0=p.n, E0=p.e, N1=other.n, E1=other.e)/self.dest.spawn_radius

                # heading intersection
                C_T = angle_to_pi(dtr(other.hdg - p.hdg))/np.pi

                # CPA metrics
                DCPA, TCPA = cpa(NOS=p.n, EOS=p.e, NTS=other.n, ETS=other.e, chiOS=np.radians(p.hdg), chiTS=dtr(other.hdg),
                                    VOS=p.tas, VTS=other.tas)
                DCPA = DCPA / 100.0
                TCPA = TCPA / 60.0

                # aggregate
                TS_info.append([d, bng, v_r, C_T, DCPA, TCPA])

        # no TS is in sight: pad a 'ghost ship' to avoid confusion for the agent
        if len(TS_info) == 0:
            TS_info.append([1.0, -1.0, -1.0, -1.0, 1.0, -1.0])

        # sort array according to distance
        TS_info = np.hstack(sorted(TS_info, key=lambda x: x[0], reverse=True)).astype(np.float32)

        # pad NA's as usual in single-agent LSTMRecTD3
        desired_length = self.obs_per_TS * (self.N_agents_max-1) + self.obs_per_TS * self.N_cutters_max
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

        # single-policy situation
        cnt_agent:_Agent = a
        
        for i, p in enumerate(self.planes):

            # fly planes depending on whether they are RL-, VFG-, or RND-controlled
            if p.role == "RL":

                # spatial-temporal recurrent
                act = cnt_agent.select_action(s        = p.s, 
                                              s_hist   = p.s_hist, 
                                              a_hist   = None, 
                                              hist_len = p.hist_len)
                # move plane
                p.upd_dynamics(a=act, discrete_acts=False, perf=self.perf, dest=None)

                # save action of id0 for comfort reward
                if i == 0:
                    a0 = act[0]

            else:
                p.upd_dynamics(perf=self.perf, dest=None)

        # update UTM coordinates
        for p in self.planes:
            p.n, p.e, _ = to_utm(lat=p.lat, lon=p.lon)

        # update distances to destination
        for p in self.planes:
            p.D_dest_old = copy(p.D_dest)
            p.D_dest = ED(N0=self.dest.N, E0=self.dest.E, N1=p.n, E1=p.e)

        # check destination entries
        _, entered_open, just_opened = self.dest.step(self.planes)

        # compute reward before potentially changing the priority order
        self._calculate_reward(a0)

        # respawning
        self._handle_respawn(entered_open)

        # possibly spawn additional planes in validation
        if "Validation" in type(self).__name__:
            if self.id_counter < self.N_agents_max:

                if self.sim_study:
                    if self.sim_t % 10 == 0:
                        p = self._spawn_plane(gate=self.id_counter % 4, noise=True)
                        p.id = self.unique_ids[self.id_counter]
                        self.planes.append(p)
                        self.id_counter += 1

                elif self.situation == 1:
                    if self.sim_t % 30 == 0:
                        for _ in range(4):
                            p = self._spawn_plane(gate=self.id_counter % 4, noise=True)
                            p.id = self.unique_ids[self.id_counter]
                            self.planes.append(p)
                            self.id_counter += 1
                self.N_planes = len(self.planes)

        # high-level control
        if "Validation" in type(self).__name__:
            if just_opened:
                self._high_level_control()
        else:
            self._high_level_control()

        # compute state and done        
        self._set_state()
        d = self._done()

        # logging
        if hasattr(self, "logger"):
            P_info = {}
            for id in self.unique_ids:
                try:
                    i = np.nonzero([p.id == id for p in self.planes])[0][0]
                    p = self.planes[i]
                    n = p.n
                    e = p.e 
                    hdg = p.hdg
                    tas = p.tas
                    goal = int(p.fly_to_goal)
                except:
                    n, e, hdg, tas, goal = None, None, None, None, None

                P_info[f"P{id}_n"] = n
                P_info[f"P{id}_e"] = e
                P_info[f"P{id}_hdg"] = hdg
                P_info[f"P{id}_tas"] = tas
                P_info[f"P{id}_goal"] = goal
            self.logger.store(sim_t=self.sim_t, **P_info)
        return self.state, float(self.r[0]), d, {}

    def _handle_respawn(self, entered_open:np.ndarray):
        """Respawns planes when they left the map."""
        for i, p in enumerate(self.planes):
            r = False

            # check conditions
            if p.D_dest >= self.dest.respawn_radius:
                r = True
            elif p.role == "CUT" and i != 0:
                p0 = self.planes[0]
                _, TCPA = cpa(NOS=p0.n, EOS=p0.e, NTS=p.n, ETS=p.e, chiOS=np.radians(p0.hdg), 
                              chiTS=dtr(p.hdg), VOS=p0.tas, VTS=p.tas)
                d = ED(N0=p0.n, E0=p0.e, N1=p.n, E1=p.e)
                if TCPA < 0 and d > (self.c * 400.0):
                    r = True

            # perform respawn
            if r:
                fly_to_goal = p.fly_to_goal
                self.planes[i] = self._spawn_plane(role=self.planes[i].role)
                self.planes[i].fly_to_goal = fly_to_goal

    def _calculate_reward(self, a0:float):
        r_coll = np.zeros((self.N_planes, 1), dtype=np.float32)
        r_goal = np.zeros((self.N_planes, 1), dtype=np.float32)
        r_comf = np.zeros((self.N_planes, 1), dtype=np.float32)

        # ------ collision reward ------
        D_matrix = np.ones((len(self.planes), len(self.planes))) * np.inf
        for i, pi in enumerate(self.planes):
            for j, pj in enumerate(self.planes):
                if i != j and i == 0:
                    D_matrix[i][j] = ED(N0=pi.n, E0=pi.e, N1=pj.n, E1=pj.e)

        for i, pi in enumerate(self.planes):
            if i != 0:
                continue

            D = float(np.min(D_matrix[i]))

            if D <= self.accident_dist:
                r_coll[i] -= 10.0

            elif D <= self.incident_dist:
                r_coll[i] -= 10.0

            else:
                r_coll[i] -= 5*np.exp(-(D-self.incident_dist)**2/(160.4549)**2) 
                # approximately yields reward of -5 at 100m and -0.01 at 500m
                # b = function(x, y){
                # return(sqrt(-(x-100)^2/log(y/-5)))
                #}

            # off-map
            if pi.D_dest > self.dest.spawn_radius: 
                r_coll[i] -= 5.0

        # ------ goal reward ------
        for i, p in enumerate(self.planes):

            if i != 0:
                continue
            
            # goal-approach reward for the one who should fly toward the goal
            if p.fly_to_goal == 1.0:
                r_goal[i] = (p.D_dest_old - p.D_dest)/self.r_goal_norm
            
            # punish others for getting into the restricted area
            elif p.D_dest <= self.dest.restricted_area:
                r_goal[i] = -5.0

        #--------------- comfort reward --------------------
        r_comf[0] = -(a0)**4

        # aggregate reward components
        if self.w == 0.0:
            r = np.zeros((self.N_planes, 1), dtype=np.float32)
        else:
            r = (self.w_coll*r_coll + self.w_goal*r_goal + self.w_comf*r_comf)/self.w

        # store
        self.r = r
        self.r_coll = r_coll
        self.r_goal = r_goal
        self.r_comf = r_comf

    def _done(self):
        # id-0 successfully reached the goal
        if self.planes[0].D_dest <= self.dest.radius and self.planes[0].fly_to_goal == 1:
            return True

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
                    self.ax2.r_goal = np.zeros(self._max_episode_steps)
                    self.ax2.r_comf = np.zeros(self._max_episode_steps)
                else:
                    self.ax2.r[self.step_cnt] = self.r if isinstance(self.r, float) else float(self.r[0])
                    self.ax2.r_coll[self.step_cnt] = self.r_coll if isinstance(self.r_coll, float) else float(self.r_coll[0])
                    self.ax2.r_goal[self.step_cnt] = self.r_goal if isinstance(self.r_goal, float) else float(self.r_goal[0])
                    self.ax2.r_comf[self.step_cnt] = self.r_comf if isinstance(self.r_comf, float) else float(self.r_comf[0])

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
                self.ax1.plot(es, ns, color="purple", alpha=0.75)

                # restricted area
                lats, lons = map(list, zip(*[qdrpos(latd1=self.dest.lat, lond1=self.dest.lon, qdr=deg, dist=meter_to_NM(self.dest.restricted_area))\
                    for deg in self.clock_degs]))
                ns, es, _ = to_utm(lat=np.array(lats), lon=np.array(lons))
                self.ax1.plot(es, ns, color="purple", alpha=0.75)

                # respawn area
                lats, lons = map(list, zip(*[qdrpos(latd1=self.dest.lat, lond1=self.dest.lon, qdr=deg, dist=meter_to_NM(self.dest.respawn_radius))\
                    for deg in self.clock_degs]))
                ns, es, _ = to_utm(lat=np.array(lats), lon=np.array(lons))
                self.ax1.plot(es, ns, color="black")

                # ---------- animated artists: initial drawing ---------
                # step info
                self.ax1.info_txt = self.ax1.text(x=498_800, y=6.6526e6, s="", fontdict={"size" : 9}, animated=True)

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

                for i in range(self.N_agents_max):
                    try:
                        hdg = self.planes[i].hdg
                    except:
                        hdg = 0

                    color = COLORS[i]

                    # show aircraft
                    self.ax1.scs.append(self.ax1.scatter([], [], marker=(3, 0, -hdg), color=color, animated=True))

                    # incident area
                    self.ax1.lns.append(self.ax1.plot([], [], color=color, animated=True, zorder=10)[0])

                    # information
                    self.ax1.txts.append(self.ax1.text(x=0.0, y=0.0, s="", color=color, fontdict={"size" : 8}, animated=True))

                if self.plot_reward:
                    self.ax2.lns_agg  = []
                    self.ax2.lns_coll = []
                    self.ax2.lns_goal = []
                    self.ax2.lns_comf = []

                    self.ax2.lns_agg.append(self.ax2.plot([], [], color=COLORS[0], label=f"Agg", animated=True)[0])
                    self.ax2.lns_coll.append(self.ax2.plot([], [], color=COLORS[1], label=f"Collision", animated=True)[0])
                    self.ax2.lns_goal.append(self.ax2.plot([], [], color=COLORS[2], label=f"Goal", animated=True)[0])
                    self.ax2.lns_comf.append(self.ax2.plot([], [], color=COLORS[3], label=f"Comfort", animated=True)[0])
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

                    # information
                    #s = f"id: {i}" + "\n" + f"hdg: {p.hdg:.1f}" + "\n" + f"alt: {p.alt:.1f}" + "\n" + f"tas: {p.tas:.1f}"
                    #if hasattr(p, "t_alive"):
                    #    s+= "\n" + f"t_alive: {int(p.t_alive)}" 
                    #if p.fly_to_goal == 1.0:
                    #    s += "\n" + "Go!!!"
                    
                    if p.fly_to_goal == 1.0:
                        s = "Go!"
                    else:
                        s = ""
                    self.ax1.txts[i].set_text(s)
                    self.ax1.txts[i].set_position((p.e, p.n))
                    self.ax1.draw_artist(self.ax1.txts[i])

                # reward
                if self.plot_reward:
                    self.ax2.lns_agg[0].set_data(np.arange(self.step_cnt+1), self.ax2.r[:self.step_cnt+1])
                    self.ax2.lns_coll[0].set_data(np.arange(self.step_cnt+1), self.ax2.r_coll[:self.step_cnt+1])
                    self.ax2.lns_goal[0].set_data(np.arange(self.step_cnt+1), self.ax2.r_goal[:self.step_cnt+1])
                    self.ax2.lns_comf[0].set_data(np.arange(self.step_cnt+1), self.ax2.r_comf[:self.step_cnt+1])
                        
                    self.ax2.draw_artist(self.ax2.lns_agg[0])
                    self.ax2.draw_artist(self.ax2.lns_coll[0])
                    self.ax2.draw_artist(self.ax2.lns_goal[0])
                    self.ax2.draw_artist(self.ax2.lns_comf[0])

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
               
            plt.pause(0.01)
