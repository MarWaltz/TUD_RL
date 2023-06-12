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
    def __init__(self, situation:str, N_agents_max:int, VFG:bool, safe_number:int):

        assert situation in ["random", "equal", "corridor"], "Unknown validation situation."
        assert not (VFG and N_agents_max != 12), "Go for 12 agents if you want to incorporate VFG."

        self.situation = situation
        self.safe_number = safe_number
        self.VFG = VFG

        if VFG:
            self.VFG_K = 0.01

        super().__init__(N_agents_max=N_agents_max, w_coll=0.0, w_goal=0.0)
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

        # scenario-dependent spawning routine
        for n in range(self.N_planes):
            while True:
                # select behavior
                if self.VFG and n <= 5:
                    role = "VFG"
                else:
                    role = "RL"

                # create plane
                p = self._spawn_plane(n=n, role=role)

                # make sure we don't spawn in a conflict already in random situation
                if self.situation == "random":
                    if np.min([np.inf] + [ED(N0=p.n, E0=p.e, N1=other.n, E1=other.e) for other in self.planes]) >= 3.0*self.incident_dist:
                        self.planes.append(p)
                        break
                else:
                    self.planes.append(p)
                    break

        # init life times
        ts_alive = random.sample(population=list(range(0, 61)), k=self.N_planes)
        for i, p in enumerate(self.planes):

            # VFG vehicles shouldn't fly to the goal
            if p.role == "VFG":
                self.planes[i].t_alive = 0
            else:
                self.planes[i].t_alive = ts_alive[i]
        
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

        # logging
        P_info = {}
        for i, p in enumerate(self.planes):
            P_info[f"P{id(p)}_n"] = p.n
            P_info[f"P{id(p)}_e"] = p.e
            P_info[f"P{id(p)}_hdg"] = p.hdg
            P_info[f"P{id(p)}_tas"] = p.tas
            P_info[f"P{id(p)}_goal"] = int(p.fly_to_goal)
        self.logger = UAMLogger(sim_t=self.sim_t, **P_info)
        return self.state

    def _spawn_plane(self, n:int, role:str):
        assert role in ["RL", "VFG", "RND"], "Unknown role."

        # scenario-dependent
        if self.situation == "equal":
            hdg = np.degrees(np.linspace(start=0.0, stop=2*np.pi, num=self.N_agents_max, endpoint=False)[n])
            qdr = (hdg+180)%360
            tas = self.actas
            dist = meter_to_NM(self.dest.spawn_radius)

        elif self.situation == "corridor":
            assert self.N_agents_max == 9, "Consider 9 flight taxis in corridor setup."
            hdg = [35., 45., 55., 170., 180., 190., 305., 315., 325.][n]
            qdr = (hdg+180)%360
            tas = self.actas
            dist = meter_to_NM(self.dest.spawn_radius)

        elif self.situation == "random":
            hdg = np.degrees(np.linspace(start=0.0, stop=2*np.pi, num=self.N_agents_max, endpoint=False)[n])
            qdr = (hdg + 180) % 360
            hdg = (hdg + np.random.uniform(low=-45.0, high=45.0)) % 360
            tas = self.actas
            dist = meter_to_NM(self.dest.spawn_radius)

        # determine origin
        lat, lon = qdrpos(latd1=self.dest.lat, lond1=self.dest.lon, qdr=qdr, dist=dist)

        # consider behavior type
        p = Plane(role=role, dt=self.dt, actype=self.actype, lat=lat, lon=lon, alt=self.acalt, hdg=hdg, tas=tas)

        # set UTM coordinates
        p.n, p.e, _ = to_utm(lat=lat, lon=lon)

        # potentially create circle
        if role == "VFG":
            p.circle = np.random.uniform(low=self.dest.restricted_area+200, high=self.dest.spawn_radius-200)

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
                t_alive     = p.t_alive
                fly_to_goal = p.fly_to_goal
                self.planes[i] = self._spawn_plane(role=self.planes[i].role)
                
                self.planes[i].t_alive     = t_alive
                self.planes[i].fly_to_goal = fly_to_goal

    def _done(self):
        d = False

        # artificial done signal
        if self.step_cnt >= self._max_episode_steps:
            d = True
        
        # all planes gone
        elif len(self.planes) == 0:
            d = True
        
        # only VFG-controlled planes left
        elif all([p.role == "VFG" for p in self.planes]):
            d = True

        if d:
            self.logger.dump(name="UAM_" + str(self.situation) + "_" + str(self.N_agents_max) + "_" + str(self.VFG) + "_" + str(self.safe_number))
        return d

    #def render(self, mode=None):
    #    pass
