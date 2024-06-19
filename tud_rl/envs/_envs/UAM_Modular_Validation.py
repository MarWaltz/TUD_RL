import uuid

from tud_rl.envs._envs.UAM_Modular import *


class Gate:
    def __init__(self, destination:Destination, direction:str) -> None:
        assert direction in ["N", "E", "S", "W"], "Unknown direction."

        r = destination.spawn_radius
        N = destination.N
        E = destination.E

        if direction == "N":
            self.N = N + r
            self.E = E
        elif direction == "E":
            self.N = N
            self.E = E + r
        elif direction == "S":
            self.N = N - r
            self.E = E
        else:
            self.N = N
            self.E = E - r


class UAMLogger:
    """Implements trajectory storing to enable validation plotting for the urban air mobility project."""
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, [value])

    def store(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                eval(f"self.{key}.append({value})")
            else:
                setattr(self, key, [value])

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
    def __init__(self, situation:int, sim_study:bool, sim_study_N:int, safe_number:int, entry_check:bool, noise:float = 0.0):

        self.situation   = situation
        self.sim_study   = sim_study
        self.sim_study_N = sim_study_N
        self.safe_number = safe_number
        self.entry_check = entry_check
        self.noise = noise

        assert not (not sim_study and entry_check), "Entry check only in sim-study possible."

        assert not (not sim_study and sim_study_N is not None), "Specify sim_study = True if you give number of agents for it."
        assert situation == 1, "We have only situation 1 at the moment."

        if sim_study:
            N_agents_max = sim_study_N

        elif situation == 1:
            N_agents_max = 3 * 4 # 3 waves of 4 aircraft

        super().__init__(N_agents_max=N_agents_max, N_cutters_max=0, w_coll=0.0, w_goal=0.0, w_comf=0.0, r_goal_norm=1.0, c=1.0, noise=noise)
        self._max_episode_steps = 100_000 if sim_study else 2000

        # gates
        gate_N = Gate(destination=self.dest, direction="N")
        gate_E = Gate(destination=self.dest, direction="E")
        gate_S = Gate(destination=self.dest, direction="S")
        gate_W = Gate(destination=self.dest, direction="W")
        self.gates = [gate_N, gate_E, gate_S, gate_W]

        # viz
        self.plot_reward = False
        self.plot_state  = False

    def reset(self):
        """Resets environment to initial state."""
        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # create some aircraft
        self.planes:List[Plane] = []

        if self.sim_study:
            self.N_planes = 1
            noise = True

        elif self.situation == 1:
            self.N_planes = 4
            noise = False

        # create all unique id's beforehand for logging
        self.unique_ids = [str(uuid.uuid4()).replace("-","_") for _ in range(self.N_agents_max)]
        self.id_counter = 0

        # spawn planes
        for _ in range(self.N_planes):

            # create plane
            if self.sim_study:
                gate = np.random.choice(4)
            else:
                gate = len(self.planes) % 4
            p = self._spawn_plane(gate=gate, noise=noise)

            # assign unique id
            p.id = self.unique_ids[self.id_counter]
            self.planes.append(p)
            self.id_counter += 1

        # interface to high-level module including goal decision
        self._high_level_control()

        # reset dest
        self.dest.reset()

        # add positional noise
        self._add_noise()

        # init state
        self._set_state()
        self.state_init = self.state

        # logging
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
        self.logger = UAMLogger(sim_t=self.sim_t, **P_info)
        return self.state

    def _gate_is_free(self, gate:Gate):
        for p in self.planes:
            if ED(N0=p.n, E0=p.e, N1=gate.N, E1=gate.E) < 300:
                return False
        return True

    def _high_level_control(self):
        """Decides who out of the current flight taxis should fly toward the goal."""
        if len(self.planes) > 0:

            # check whether one guy has already a go-signal
            if all([p.fly_to_goal == -1.0 for p in self.planes]):

                idx = np.argmin([p.D_dest for p in self.planes])
                for i, _ in enumerate(self.planes):
                    if i == idx:
                        self.planes[i].fly_to_goal = 1.0
                    else:
                        self.planes[i].fly_to_goal = -1.0

    def _spawn_plane(self, gate:int=None, noise:bool=True, role:str="RL"):
        # set params
        qdr = [0.0, 90.0, 180.0, 270.0][gate]
        hdg = (qdr + 180) % 360
        tas = self.actas
        dist = self.dest.spawn_radius

        if noise:
            hdg = (hdg + np.random.uniform(low=-20.0, high=20.0)) % 360
            tas += np.random.uniform(low=-self.delta_tas, high=self.delta_tas)

        # determine origin
        E_add, N_add = xy_from_polar(r=dist, angle=dtr(qdr))
        lat, lon = to_latlon(north=self.dest.N+N_add, east=self.dest.E+E_add, number=32)

        # consider behavior type
        p = Plane(role=role, dt=self.dt, actype=self.actype, lat=lat, lon=lon, alt=self.acalt, hdg=hdg, tas=tas)

        # set UTM coordinates
        p.n, p.e, _ = to_utm(lat=lat, lon=lon)

        # compute initial distance to destination
        p.D_dest     = ED(N0=self.dest.N, E0=self.dest.E, N1=p.n, E1=p.e)
        p.D_dest_old = copy(p.D_dest)

        # do not fly to goal per default
        p.fly_to_goal = -1.0
        return p

    def _handle_respawn(self, entered_open:np.ndarray):
        """Respawns planes when they correctly entered the open destination area or left the whole map."""
        for i, p in enumerate(self.planes):
            if (entered_open[i] and p.fly_to_goal == 1.0) or p.D_dest >= self.dest.respawn_radius:
                self.planes.pop(i)
                self.N_planes = len(self.planes)

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
            # Release the video writer
            if hasattr(self, "video_writer"):
                self.video_writer.release()

            # Dump episode details
            if self.sim_study:
                self.logger.dump(name="UAM_SimStudy_" + str(self.N_agents_max) + "_" + str(self.safe_number))
            else:
                self.logger.dump(name="UAM_ValScene_" + str(self.situation) + "_" + str(self.N_agents_max))
        return d

    def render(self, mode=None):
        pass
        #super().render(mode=mode)
