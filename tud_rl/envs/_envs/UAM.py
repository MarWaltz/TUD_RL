from copy import copy
from string import ascii_letters
from typing import List

import gym
from bluesky.tools.geo import latlondist, qdrdist, qdrpos
from gym import spaces
from matplotlib import pyplot as plt
from mycolorpy import colorlist as mcp

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
        self.open()

    def reset(self):
        self.open()

    def step(self, planes: List[Plane], collective_r=True):
        """Updates status of the destination and computes destination reward component.
        Returns:
            np.array([len(planes),1]) with rewards
            np.array(len(planes),) with respawn flags from entering the destination area"""
        # count time until next opening
        if self._is_open is False:
            self._t_nxt_open -= self.dt
            if self._t_nxt_open <= 0:
                self.open()

        # close if someone enters
        r = np.zeros((len(planes), 1), dtype=np.float32)
        respawn_flags = np.zeros(len(planes), dtype=bool)

        for i, p in enumerate(planes):
            if p.D_dest <= self.radius:
                
                # respawn only if airplane entered open destination
                if self._is_open:
                    if collective_r:
                        r += 5.0
                    else:
                        r[i] += 5.0
                    respawn_flags[i] = True
                    self.close()

                # a violation is considered a collision and is thus individual reward
                else:
                    r[i] -= 5.0
        return r, respawn_flags

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


class UAM(gym.Env):
    """Urban air mobility simulation env based on the BlueSky simulator of Ellerbroek and Hoekstra."""
    def __init__(self, N_agents):
        super(UAM, self).__init__()

        # flight params
        self.N_agents = N_agents
        self.acalt = 300 # [m]
        self.actas = 15  # [m/s]
        self.actype = "MAVIC"

        # domain params
        self.LoS_dist = 100 # [m]
        self.clock_degs = np.linspace(0.0, 360.0, num=100, endpoint=True)

        # destination
        self.dt = 1.0
        self.dest = Destination(self.dt)

        # performance model
        self.perf = OpenAP(self.actype, self.actas, self.acalt)

        # config
        self.N_agents   = N_agents
        self.obs_per_TS = 4
        self.obs_size   = 3 + self.obs_per_TS*(self.N_agents-1)

        self.observation_space = spaces.Box(low  = np.full(self.obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(self.obs_size,  np.inf, dtype=np.float32))
        act_size = 1
        self.action_space = spaces.Box(low  = np.full(act_size, -1.0, dtype=np.float32), 
                                       high = np.full(act_size, +1.0, dtype=np.float32))
        self._max_episode_steps = 250

        # viz
        self.plot_reward = False
        self.plot_state = False
        self.r = np.zeros((self.N_agents, 1))
        self.state = np.zeros((self.N_agents, self.obs_size))
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
        self.planes = []
        for n in range(self.N_agents):
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
        
        # set plane
        lat, lon = qdrpos(latd1=self.dest.lat, lond1=self.dest.lon, qdr=hdg, dist=meter_to_NM(self.dest.spawn_radius))
        p = Plane(dt=self.dt, actype=self.actype, lat=lat, lon=lon, alt=self.acalt, hdg=(hdg+180)%360, tas=tas)

        # compute initial distance to destination
        p.D_dest       = latlondist(latd1=self.dest.lat, lond1=self.dest.lon, latd2=lat, lond2=lon)
        p.D_dest_old   = copy(p.D_dest)
        p.just_spawned = True
        return p

    def _set_state(self):
        """State contains own relative bearing of goal, distance to goal, common four information about target ships
        (relative speed, relative bearing, distance, heading intersection angle), and time until destination opens again.
        Overall 3 + 4*(N_agents-1) features. Thus, state is np.array([N_agents, 3 + 4*(N_agents-1)])."""
        self.state = np.zeros((self.N_agents, self.obs_size), dtype=np.float32)

        for i, p in enumerate(self.planes):

            # distance, bearing to goal and time to opening
            abs_bng_goal, d_goal = qdrdist(latd1=p.lat, lond1=p.lon, latd2=self.dest.lat, lond2=self.dest.lon) # outputs ABSOLUTE bearing
            bng_goal = angle_to_pi(angle_to_2pi(dtr(abs_bng_goal)) - dtr(p.hdg))
            s_i = np.array([bng_goal/np.pi,\
                            NM_to_meter(d_goal)/self.dest.spawn_radius,\
                            1.0-self.dest.t_nxt_open/self.dest.t_close])

            # four information about other planes
            if self.N_agents > 1:
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

                # sort array
                TS_info = np.hstack(sorted(TS_info, key=lambda x: x[0])).astype(np.float32)
                s_i = np.concatenate((s_i, TS_info))

            # store it
            self.state[i] = s_i

    def step(self, a):
        """a is np.array([N_agents, action_dim]), where action_dim is 2 or 1."""
        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += self.dt
 
        # fly planes
        [p.upd_dynamics(a[i], perf=self.perf, dest=None) for i, p in enumerate(self.planes)]

        # update distances to destination
        for p in self.planes:
            p.D_dest_old = copy(p.D_dest)
            p.D_dest = latlondist(latd1=p.lat, lond1=p.lon, latd2=self.dest.lat, lond2=self.dest.lon)

        # check destination entries
        r, respawn_flags = self.dest.step(self.planes)

        # respawning
        self._handle_respawn(respawn_flags)

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward(r, a)
        d = self._done()
        return self.state, self.r, d, {}
 
    def _handle_respawn(self, respawn_flags):
        """Respawns planes when they entered the open destination area or are at the outer simulation radius."""
        for i, p in enumerate(self.planes):
            if (latlondist(latd1=self.dest.lat, lond1=self.dest.lon, latd2=p.lat, lond2=p.lon) >= self.dest.respawn_radius)\
                 or respawn_flags[i]:
                self.planes[i] = self._spawn_plane()
            else:
                p.just_spawned = False

    def _calculate_reward(self, r, a, collective_r=True):
        """Computes reward.
        Args:
            r: np.array([N_agents, 1]), destination reward component
            a: np.array([N_agents, action_dim]), selected actions
            collective_r: bool, whether to include collective reward component
        Returns:
            None, but sets self.r to np.array([N_agents, 1]) with the complete reward
        """
        # individual reward components: collision and off-map
        for i, pi in enumerate(self.planes):

            # collision reward
            for j, pj in enumerate(self.planes):
                if i != j:
                    D = latlondist(latd1=pi.lat, lond1=pi.lon, latd2=pj.lat, lond2=pj.lon)-self.LoS_dist
                    if D <= 0:
                        r[i] -= 5.0
                    else:
                        r[i] -= np.exp(-D/self.LoS_dist)
            
            # off-map reward
            if pi.D_dest > (self.dest.spawn_radius+5): # against numerical issues
                r[i] -= 5.0

        # collective reward component: goal approaching when it is open
        if collective_r:
            deltas = [p.D_dest_old - p.D_dest for p in self.planes] + [0.0]
            if self.dest.t_nxt_open == 0:
                r += max(deltas)/15.0
        else:
            for i, p in enumerate(self.planes):
                if (not p.just_spawned) and (self.dest.t_nxt_open == 0):
                    r[i] += (p.D_dest_old - p.D_dest) / (15.0)       
        self.r = r

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
            
            # set screen
            self.ax1.clear()
            #self.ax1.set_xlim(8, 12)
            #self.ax1.set_ylim(8, 12)
            self.ax1.set_xlabel("Lon [°]")
            self.ax1.set_ylabel("Lat [°]")
            self.ax1.text(0.25, 0.8875, self.__str__(), fontsize=9, transform=plt.gcf().transFigure)

            # destination
            lats, lons = map(list, zip(*[qdrpos(latd1=self.dest.lat, lond1=self.dest.lon, qdr=deg, dist=meter_to_NM(self.dest.radius))\
                 for deg in self.clock_degs]))
            self.ax1.plot(lons, lats, color=self.dest.color)
            
            # spawning area
            lats, lons = map(list, zip(*[qdrpos(latd1=self.dest.lat, lond1=self.dest.lon, qdr=deg, dist=meter_to_NM(self.dest.spawn_radius))\
                 for deg in self.clock_degs]))
            self.ax1.plot(lons, lats, color="grey")

            # respawn area
            lats, lons = map(list, zip(*[qdrpos(latd1=self.dest.lat, lond1=self.dest.lon, qdr=deg, dist=meter_to_NM(self.dest.respawn_radius))\
                 for deg in self.clock_degs]))
            self.ax1.plot(lons, lats, color="black")

            for i, p in enumerate(self.planes):
                # show aircraft
                self.ax1.scatter(p.lon, p.lat, marker=(3, 0, -p.hdg), color=COLORS[i])

                # LoS area
                lats, lons = map(list, zip(*[qdrpos(latd1=p.lat, lond1=p.lon, qdr=deg, dist=meter_to_NM(self.LoS_dist))\
                     for deg in self.clock_degs]))
                self.ax1.plot(lons, lats, color=COLORS[i])

                # information
                s = f"id: {i}" + "\n" + f"hdg: {p.hdg:.1f}" + "\n" + f"alt: {p.alt:.1f}" + "\n" + f"tas: {p.tas:.1f}"
                self.ax1.text(x=p.lon, y=p.lat, s=s, color=COLORS[i], fontdict={"size" : 8})

            if self.plot_reward:
                # reward
                if self.step_cnt == 0:
                    self.ax2.clear()
                    self.ax2.old_time = 0
                    self.ax2.old_r = np.zeros((self.N_agents, 1))
                    self.ax2.set_title("Urban Air Mobility")

                self.ax2.set_xlabel("Timestep in episode")
                self.ax2.set_ylabel("Reward")

                for i in range(self.r.shape[0]):
                    self.ax2.plot([self.ax2.old_time, self.step_cnt], [self.ax2.old_r[i], self.r[i]],\
                         color = COLORS[i], label=f"R {i}")

                if self.step_cnt == 0:
                    self.ax2.legend()

                self.ax2.old_time = self.step_cnt
                self.ax2.old_r = self.r

            # state
            if self.plot_state:
                if self.step_cnt == 0:
                    self.ax3.clear(loc='upper right')
                    self.ax3.old_time = 0
                    self.ax3.old_s = np.zeros((self.N_agents, self.obs_size))

                self.ax3.set_xlabel("Timestep in episode")
                self.ax3.set_ylabel("State of Agent 0")

                for i in range(self.obs_size):
                    self.ax3.plot([self.ax3.old_time, self.step_cnt], [self.ax3.old_s[0][i], self.state[0][i]],\
                         label=self.obs_names[i], color=COLORS[i])

                if self.step_cnt == 0:
                    self.ax3.legend(loc='upper right')

                self.ax3.old_time = self.step_cnt
                self.ax3.old_s = self.state
                
            plt.pause(0.001)
