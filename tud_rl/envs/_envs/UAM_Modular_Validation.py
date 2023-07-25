import uuid

from tud_rl.envs._envs.UAM_Modular import *


class UAMLogger:
    """Implements trajectory storing to enable validation plotting for the urban air mobility project."""
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, [value])

    def store(self, **kwargs):
        for key, value in kwargs.items():
            eval(f"self.{key}.append({value})")

    def dump(self, name):
        data = vars(self)
        L = max([len(e) for e in data.values()])
        for key, value in data.items():
            if len(value) != L:
                data[key] += [None] * (L - len(value))

        df = pd.DataFrame(vars(self))
        df.replace(to_replace=[None], value=0.0, inplace=True) # clear None
        df.to_pickle(f"{name}.pkl")


class UAM_Modular_Validation(UAM_Modular):
    """Validation scenarios for the Urban Air Mobility agent."""
    def __init__(self, situation:int, sim_study:bool, sim_study_N:int, safe_number:int):

        self.situation   = situation
        self.sim_study   = sim_study
        self.sim_study_N = sim_study_N
        self.safe_number = safe_number

        assert not (not sim_study and sim_study_N is not None), "Specify sim_study = True if you give number of agents for it."

        if sim_study:
            N_agents = sim_study_N

        elif situation == 1:
            N_agents = 10

        elif situation == 2:
            N_agents = 12

        elif situation == 3:
            N_agents = 9

        elif situation == 4:
            N_agents = 11

        super().__init__(N_agents_max=N_agents, N_cutters_max=0, w_coll=0.0, w_goal=0.0, w_comf=0.0, r_goal_norm=1.0, c=1.0)
        self._max_episode_steps = 1000

        # viz
        self.plot_reward = False
        self.plot_state  = False

    def reset(self):
        """Resets environment to initial state."""
        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # create some aircrafts
        self.planes:List[Plane] = []
        self.N_planes = self.N_agents_max

        # spawn planes
        for n in range(self.N_agents_max):
            while True:

                # select behavior
                if self.sim_study:
                    role = "RL"
                elif self.situation == 2 and n in [10, 11]:
                    role = "CUT"
                elif self.situation == 4 and n in [9, 10]:
                    role = "CUT"
                else:
                    role = "RL"

                # create plane
                p = self._spawn_plane(n=n, role=role)

                # make sure we don't spawn in a conflict already in random situation
                if np.min([np.inf] + [ED(N0=p.n, E0=p.e, N1=other.n, E1=other.e) for other in self.planes]) >= 1.0*self.incident_dist:
                    self.planes.append(p)
                    break

        # init life times
        ts_alive = random.sample(population=list(range(0, 61)), k=self.N_planes)
        for i, p in enumerate(self.planes):

            # only RL planes should fly to the goal
            if p.role == "RL":
                self.planes[i].t_alive = ts_alive[i]
            else:
                self.planes[i].t_alive = -1e10
        
        # interface to high-level module including goal decision
        self._high_level_control()

        # set waypoints and compute cross-track and course error
        for p in self.planes:
            if p.role == "VFG":
                p = self._plan_path(p)
                p = self._init_wps(p)
                p = self._set_path_metrics(p)

        # reset dest
        self.dest.reset()

        # init state
        self._set_state()
        self.state_init = self.state

        # set unique ID's
        for i, _ in enumerate(self.planes):
            self.planes[i].id = str(uuid.uuid4()).replace("-","_")

        # logging
        P_info = {}
        for i, p in enumerate(self.planes):
            P_info[f"P{p.id}_n"] = p.n
            P_info[f"P{p.id}_e"] = p.e
            P_info[f"P{p.id}_hdg"] = p.hdg
            P_info[f"P{p.id}_tas"] = p.tas
            P_info[f"P{p.id}_goal"] = int(p.fly_to_goal)
        self.logger = UAMLogger(sim_t=self.sim_t, **P_info)
        return self.state

    def _high_level_control(self):
        """Decides who out of the current flight taxis should fly toward the goal."""
        if len(self.planes) > 0:
            idx = np.argmax([p.t_alive for p in self.planes])
            for i, _ in enumerate(self.planes):
                if i == idx:
                    self.planes[i].fly_to_goal = 1.0
                else:
                    self.planes[i].fly_to_goal = -1.0

    def _spawn_plane(self, role:str, n:int=None, respawn:bool=False):
        # sim-study setup
        if self.sim_study:
            hdg = np.degrees(np.random.uniform(low=0, high=2*np.pi))
            qdr = (hdg + 180) % 360
            sgn = 1 if bool(random.getrandbits(1)) else -1
            hdg = (hdg + sgn * np.random.uniform(low=0.0, high=20.0)) % 360
            tas = self.actas
            dist = self.dest.spawn_radius

        # cutters
        elif role == "CUT" or respawn:
            hdg = np.degrees(np.random.uniform(low=0, high=2*np.pi))
            qdr = (hdg + 180) % 360
            sgn = 1 if bool(random.getrandbits(1)) else -1
            hdg = (hdg + sgn * np.random.uniform(low=20.0, high=45.0)) % 360
            tas = self.actas
            dist = self.dest.spawn_radius

        # equal
        elif self.situation in [1, 2]:
            if self.situation == 1:
                num = self.N_agents_max
            else:
                num = self.N_agents_max - 2

            hdg = np.degrees(np.linspace(start=0.0, stop=2*np.pi, num=num, endpoint=False)[n])
            qdr = (hdg + 180) % 360
            tas = self.actas
            dist = self.dest.spawn_radius

        # corridor
        elif self.situation in [3, 4]:
            hdg = [35., 45., 55., 170., 180., 190., 305., 315., 325.][n]
            qdr = (hdg + 180) % 360
            tas = self.actas
            dist = self.dest.spawn_radius

        # determine origin
        E_add, N_add = xy_from_polar(r=dist, angle=dtr(qdr))
        lat, lon = to_latlon(north=self.dest.N+N_add, east=self.dest.E+E_add, number=32)

        # consider behavior type
        p = Plane(role=role, dt=self.dt, actype=self.actype, lat=lat, lon=lon, alt=self.acalt, hdg=hdg, tas=tas)

        # set UTM coordinates
        p.n, p.e, _ = to_utm(lat=lat, lon=lon)

        # potentially create circle
        #if role == "VFG":
        #    p.circle = np.random.uniform(low=self.dest.restricted_area+200, high=self.dest.spawn_radius-200)

        # compute initial distance to destination
        p.D_dest     = ED(N0=self.dest.N, E0=self.dest.E, N1=p.n, E1=p.e)
        p.D_dest_old = copy(p.D_dest)
        return p

    def _plan_path(self, p:Plane):
        # linear path to goal
        if p.fly_to_goal:
            ang  = bng_abs(N0=self.dest.N, E0=self.dest.E, N1=p.n, E1=p.e)
            dist = np.linspace(start=self.dest.spawn_radius, stop=0.0, num=20)

        # circle-shaped path
        else:
            ang  = np.linspace(start=2*np.pi, stop=0.0, num=100) % (2*np.pi)
            dist = p.circle

        # create path
        E_add, N_add = xy_from_polar(r=dist, angle=ang)
        p.Path = Path(east=self.dest.E + E_add, north=self.dest.N + N_add)
        return p

    def _handle_respawn(self, entered_open:np.ndarray):
        """Respawns planes when they entered the open destination area."""
        for i, p in enumerate(self.planes):
            # clear plane completely
            if entered_open[i]:
                self.planes.pop(i)
            
            # priority unaffected after leaving the map
            elif p.D_dest >= self.dest.respawn_radius:

                if p.role == "RL":
                    self.planes.pop(i)
                else:
                    t_alive     = p.t_alive
                    fly_to_goal = p.fly_to_goal
                    p_id        = p.id
                    self.planes[i] = self._spawn_plane(role=self.planes[i].role, respawn=True)
                    
                    self.planes[i].t_alive     = t_alive
                    self.planes[i].fly_to_goal = fly_to_goal
                    self.planes[i].id          = p_id

    def _done(self):
        d = False

        # artificial done signal
        if self.step_cnt >= self._max_episode_steps:
            d = True
        
        # all planes gone
        elif len(self.planes) == 0:
            d = True
        
        # only non-RL-controlled planes left
        elif all([p.role != "RL" for p in self.planes]):
            d = True

        if d:
            if self.sim_study:
                self.logger.dump(name="UAM_SimStudy_" + str(self.N_agents_max) + "_" + str(self.safe_number))
            else:
                self.logger.dump(name="UAM_ValScene_" + str(self.situation) + "_" + str(self.N_agents_max))
        return d

    #def render(self, mode=None):
    #    pass
