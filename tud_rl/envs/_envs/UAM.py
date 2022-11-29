import gym
from bluesky.tools.geo import latlondist, qdrpos
from gym import spaces
from matplotlib import pyplot as plt
from mycolorpy import colorlist as mcp

from tud_rl.envs._envs.Plane import *
from tud_rl.envs._envs.VesselFnc import angle_to_2pi, meter_to_NM

COLORS = [plt.rcParams["axes.prop_cycle"].by_key()["color"][i] for i in range(8)] + 5 * mcp.gen_color(cmap="tab20b", n=20) 


class UAM(gym.Env):
    """Urban air mobility simulation env based on the BlueSky simulator of Ellerbroek and Hoekstra."""
    def __init__(self, N_agents):
        super(UAM, self).__init__()

        # flight params
        self.acalt = 300 # m
        self.actas = 50  # m/s
        self.actype = "DJI Mavic pro"

        # size of domain
        self.r_inner = 100  # m
        self.r_outer = 1000 # m
        self.lat_c = 10 # deg
        self.lon_c = 10 # deg
        self.LoS_dist = 100 # m
        self.clock_degs = np.linspace(0.0, 360.0, num=100, endpoint=True)

        # performance model
        self.perf = OpenAP(self.actype, self.actas, self.acalt)

        # config
        self.N_agents = N_agents
        obs_size = 1
        self.observation_space = spaces.Box(low  = np.full(obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(obs_size,  np.inf, dtype=np.float32))
        self.action_space = spaces.Discrete(3)

        self.dt = 0.5
        self._max_episode_steps = 500

    def reset(self):
        """Resets environment to initial state."""
        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # create some aircrafts
        self.planes = [self._spawn_plane() for _ in range(self.N_agents)]

        # init state
        self._set_state()
        self.state_init = self.state
        return self.state
  
    def _spawn_plane(self):
        hdg = float(np.random.uniform(0.0, 360.0, size=1))
        lat, lon = qdrpos(latd1=self.lat_c, lond1=self.lon_c, qdr=hdg, dist=meter_to_NM(self.r_inner+self.r_outer))
        p = Plane(dt=self.dt, actype=self.actype, lat=lat, lon=lon, alt=self.acalt, hdg=hdg+180, tas=self.actas)
        p.just_spawned = True
        return p

    def _set_state(self):
        self.state = None

    def step(self, a):
        """a is np.array([N_agents, action_dim]), where action_dim = 2."""
        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += self.dt
 
        # fly planes
        [p.upd_dynamics(a[i], perf=self.perf) for i, p in enumerate(self.planes)]

        # potentially respawned if a plane left the area
        for i, p in enumerate(self.planes):
            d = latlondist(latd1=self.lat_c, lond1=self.lon_c, latd2=p.lat, lond2=p.lon)

            if d > (self.r_inner + self.r_outer + 5):   # against potential rounding issues
                self.planes[i] = self._spawn_plane()
            else:
                self.planes[i].just_spawned = False

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward(a)
        d = self._done()
        return self.state, self.r, d, {}
 
    def _calculate_reward(self, a):
        self.r = 0.0

    def _done(self):
        return False

    def _get_LoSs(self):
        """Computes current Loss of Separations (LoSs)."""
        LoSs = 0
        for i, pi in enumerate(self.planes):
            for j, pj in enumerate(self.planes):
                if i < j:
                    if latlondist(latd1=pi.lat, lond1=pi.lon, latd2=pj.lat, lond2=pj.lon) <= self.LoS_dist:
                        LoSs += 1
        return LoSs

    def __str__(self):
        return f"Step: {self.step_cnt}, Sim-Time [s]: {int(self.sim_t)}, Cnt LoSs: {self._get_LoSs()}"

    def render(self, mode=None):
        """Renders the current environment."""

        # plot every nth timestep
        if self.step_cnt % 1 == 0: 
            
            # init figure
            if len(plt.get_fignums()) == 0:
                self.f, self.ax1 = plt.subplots(1, 1, figsize=(10, 10))
                plt.ion()
                plt.show()           
            
            # set screen
            self.ax1.clear()
            #self.ax1.set_xlim(8, 12)
            #self.ax1.set_ylim(8, 12)
            self.ax1.set_xlabel("Lon [°]")
            self.ax1.set_ylabel("Lat [°]")
            self.ax1.text(0.125, 0.8875, self.__str__(), fontsize=9, transform=plt.gcf().transFigure)

            # domain
            lats, lons = map(list, zip(*[qdrpos(latd1=self.lat_c, lond1=self.lon_c, qdr=deg, dist=meter_to_NM(self.r_inner)) for deg in self.clock_degs]))
            self.ax1.plot(lons, lats, color="blue")
            
            lats, lons = map(list, zip(*[qdrpos(latd1=self.lat_c, lond1=self.lon_c, qdr=deg, dist=meter_to_NM(self.r_outer+self.r_inner)) for deg in self.clock_degs]))
            self.ax1.plot(lons, lats, color="grey")

            for i, p in enumerate(self.planes):
                # show aircraft
                self.ax1.scatter(p.lon, p.lat, marker=(3, 0, -p.hdg), color=COLORS[i])

                # LoS area
                lats, lons = map(list, zip(*[qdrpos(latd1=p.lat, lond1=p.lon, qdr=deg, dist=meter_to_NM(self.LoS_dist)) for deg in self.clock_degs]))
                self.ax1.plot(lons, lats, color=COLORS[i])

                # information
                s = f"hdg: {p.hdg:.1f}" + "\n" + f"alt: {p.alt}" + "\n" + f"tas: {p.cas}"
                self.ax1.text(x=p.lon, y=p.lat, s=s, color=COLORS[i], fontdict={"size" : 8})

            plt.pause(0.001)
