from copy import copy, deepcopy
from datetime import timedelta
from pathlib import Path as PathL
from typing import List, Union

import dateutil.parser
import pytsa
import torch
from pytsa import TimePosition
from pytsa.structs import DataColumns

from tud_rl.common.nets import LSTMRecActor
from tud_rl.envs._envs.HHOS_Base_Env import *
from tud_rl.envs._envs.HHOS_Fnc import HHOSPlotter, knots_to_mps
from tud_rl.envs._envs.HHOS_RiverPlanning_Env import HHOS_RiverPlanning_Env


class HHOS_RiverPipeline_Env(HHOS_Base_Env):
    """Validation environment for the complete pipeline for planning & low-level control on rivers.
    Runs exlusively on real data."""
    def __init__(self, 
                 planner_weights:str,
                 planner_safety_net:bool, 
                 base_speed:float,
                 global_path_file:str,
                 depth_data_file:str,
                 current_data_file:str,
                 wind_data_file:str,
                 wave_data_file:str,
                 AIS_data_file:str=None,
                 N_TSs_max:int=0):
        super().__init__()

        self.planner_weights    = planner_weights
        self.planner_safety_net = planner_safety_net
        self.base_speed         = base_speed
        self.global_path_file   = global_path_file
        self.depth_data_file    = depth_data_file
        self.current_data_file  = current_data_file
        self.wind_data_file     = wind_data_file
        self.wave_data_file     = wave_data_file
        self.AIS_data_file      = AIS_data_file

        self.N_TSs     = N_TSs_max
        self.N_TSs_max = N_TSs_max

        # AIS data spec
        if AIS_data_file is not None:

            # specify start time of the travel
            if "calm" in AIS_data_file:
                self.t_travel = dateutil.parser.isoparse("2021-07-03T07:00:00.000Z")
            elif "storm" in AIS_data_file:
                self.t_travel = dateutil.parser.isoparse("2022-01-29T07:00:00.000Z")

            # define radius for target ship detection
            self.search_radius = 5.0 # NM

        # construct planner network
        self.lidar_n_beams = 10

        # init
        self.planner = LSTMRecActor(action_dim=1, num_obs_OS=4+self.lidar_n_beams, num_obs_TS=7)
        
        # weight loading
        self.planner.load_state_dict(torch.load(planner_weights))

        # construct planning env
        self.planning_env = HHOS_RiverPlanning_Env(N_TSs_max=N_TSs_max, N_TSs_random=False, 
                                                   w_ye=0.0, w_ce=0.0, w_coll=0.0, w_rule=0.0, w_comf=0.0)

        # update frequency
        self.loc_path_upd_freq = 6 # results in a new local path every 30s with delta t being 5s

        # path characteristics
        self.dist_des_rev_path = 300
        self.n_wps_loc = 10

        # vector field guidance
        self.VFG_K = 0.01  # follower value
        self.VFG_K_TS = 0.001

        # gym inherits
        OS_infos   = 5
        path_infos = 2
        env_infos  = 9
        obs_size   = OS_infos + path_infos + env_infos
        act_size   = 1

        self.observation_space = spaces.Box(low  = np.full(obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(obs_size,  np.inf, dtype=np.float32))
        self.action_space = spaces.Box(low  = np.full(act_size, -1.0, dtype=np.float32), 
                                       high = np.full(act_size,  1.0, dtype=np.float32))
        # vessel config
        self.rud_angle_max = dtr(20.0)
        self.rud_angle_inc = dtr(5.0)

        # how many longitude/latitude degrees to show for the visualization
        self.show_lon_lat = 0.05

        # episode length
        self._max_episode_steps = int(1e7)

    def _init_local_path(self):
        """Generates a local path based on the global one."""
        self.LocalPath = self.GlobalPath.construct_local_path(wp_idx = self.OS.glo_wp1_idx, n_wps_loc = self.n_wps_loc,
                                                              v_OS = self.OS._get_V())
        self.planning_method = "global"

    def reset(self):
        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # generate global path
        self._load_global_path()

        # add reversed global path for LiDAR and TS spawning
        self.RevGlobalPath = deepcopy(self.GlobalPath)
        self.RevGlobalPath.reverse(offset=self.dist_des_rev_path)

        # init OS
        OS_wp_idx = 20 if self.AIS_data_file is None else 200
        lat_init = self.GlobalPath.lat[OS_wp_idx]
        lon_init = self.GlobalPath.lon[OS_wp_idx]
        N_init = self.GlobalPath.north[OS_wp_idx]
        E_init = self.GlobalPath.east[OS_wp_idx]

        # always same speed
        spd = self.base_speed
        self.OS = KVLCC2(N_init    = N_init, 
                         E_init    = E_init, 
                         psi_init  = None,
                         u_init    = spd,
                         v_init    = 0.0,
                         r_init    = 0.0,
                         delta_t   = self.delta_t,
                         N_max     = np.infty,
                         E_max     = np.infty,
                         nps       = None,
                         full_ship = False,
                         ship_domain_size = 2)
        self.OS.rev_dir = False 

        # init waypoints and cte of OS for global path
        self.OS = self._init_wps(self.OS, "global")
        self._set_cte(path_level="global")

        # init local path
        self._init_local_path()

        # init waypoints and cte of OS for local path
        self.OS = self._init_wps(self.OS, "local")
        self._set_cte(path_level="local")

        # set heading
        self.OS.eta[2] = self.loc_pi_path

        # load environmental data
        self._load_depth_data()
        self._load_wind_data()
        self._load_current_data()
        self._load_wave_data()

        # environmental effects
        self._update_disturbances(lat_init, lon_init)

        # set nps to near-convergence
        self.OS.nps = self.OS._get_nps_from_u(u           = self.OS.nu[0], 
                                              psi         = self.OS.eta[2], 
                                              V_c         = self.V_c, 
                                              beta_c      = self.beta_c, 
                                              V_w         = self.V_w, 
                                              beta_w      = self.beta_w, 
                                              H           = self.H,
                                              beta_wave   = self.beta_wave, 
                                              eta_wave    = self.eta_wave, 
                                              T_0_wave    = self.T_0_wave, 
                                              lambda_wave = self.lambda_wave)
        # set course error
        self._set_ce(path_level="global")
        self._set_ce(path_level="local")

        # use the river planning env to generate target ships since we do not want to reimplement this functionality here
        if self.AIS_data_file is None:
            self.planning_env.OS = deepcopy(self.OS)
            self.planning_env.GlobalPath = deepcopy(self.GlobalPath)
            self.planning_env.RevGlobalPath = deepcopy(self.RevGlobalPath)
            self.planning_env._init_TSs()
            self.TSs = deepcopy(self.planning_env.TSs)
        else:
            self._init_AIS_data()
            self._set_TSs_from_AIS()

        # prepare planning env
        self.planning_env    = self.setup_planning_env(env=self.planning_env, initial=True)
        self.planning_env, _ = self.setup_planning_env(env=self.planning_env, initial=False)
        self.planning_env.step_cnt = 0
        self.planning_env.sim_t = 0.0

        # override the local path to start the control-system from the beginning
        self._update_local_path()

        # update wps and error
        self.OS = self._init_wps(self.OS, "local")
        self._set_cte(path_level="local")
        self._set_ce(path_level="local")

        # init state
        self._set_state()

        # ----- viz -----
        TS_info = {}
        for i in range(self.N_TSs_max):
            try:
                # positions and speed
                TS = self.TSs[i]
                TS_info[f"TS{str(i)}_N"] = TS.eta[0]
                TS_info[f"TS{str(i)}_E"] = TS.eta[1]
                TS_info[f"TS{str(i)}_head"] = TS.eta[2]
                TS_info[f"TS{str(i)}_V"] = TS._get_V()

                # distance
                D = get_ship_domain(A=self.OS.ship_domain_A, B=self.OS.ship_domain_B, C=self.OS.ship_domain_C, D=self.OS.ship_domain_D,
                                    OS=self.OS, TS=TS)               
                TS_info[f"TS{str(i)}_dist"] = ED(N0=self.OS.eta[0], E0=self.OS.eta[1], N1=TS.eta[0], E1=TS.eta[1]) - D
            except:
                TS_info[f"TS{str(i)}_N"] = None
                TS_info[f"TS{str(i)}_E"] = None
                TS_info[f"TS{str(i)}_head"] = None
                TS_info[f"TS{str(i)}_V"] = None
                TS_info[f"TS{str(i)}_dist"] = None

        self.plotter = HHOSPlotter(sim_t=self.sim_t, OS_N=self.OS.eta[0], OS_E=self.OS.eta[1], OS_head=self.OS.eta[2], OS_u=self.OS.nu[0],\
            OS_v=self.OS.nu[1], OS_r=self.OS.nu[2], loc_ye=self.loc_ye, glo_ye=self.glo_ye, loc_course_error=self.loc_course_error,\
                glo_course_error=self.glo_course_error, V_c=self.V_c, beta_c=self.beta_c, V_w=self.V_w, beta_w=self.beta_w,\
                    T_0_wave=self.T_0_wave, eta_wave=self.eta_wave, beta_wave=self.beta_wave, lambda_wave=self.lambda_wave,\
                        rud_angle=self.OS.rud_angle, nps=self.OS.nps, **TS_info)
        return self.state

    def _init_AIS_data(self):
        """Initializes the interface to the pytsa-package for working with AIS data."""
        frame = pytsa.LatLonBoundingBox(
            LATMIN = 52.2, # [°N]
            LATMAX = 56.9, # [°N]
            LONMIN = 6.3,  # [°E]
            LONMAX = 11,  # [°E]
        )
        sourcefile = PathL(self.AIS_data_file)

        def my_filter(df: pd.DataFrame) -> pd.DataFrame:
            return df[df[DataColumns.MESSAGE_ID].isin([1,2,3])]
        
        self.AIS_search_agent = pytsa.SearchAgent(datapath=sourcefile, frame=frame, search_radius=self.search_radius, 
                                                  n_cells=1, filter=my_filter)

        # initialize
        OS_lat, OS_lon = to_latlon(north=self.OS.eta[0], east=self.OS.eta[1], number=32)
        tpos = TimePosition(timestamp=self.t_travel, lat=OS_lat, lon=OS_lon)
        tpos.timestamp = tpos.timestamp.replace(tzinfo=None)
        self.AIS_search_agent.init(tpos)

    def _set_TSs_from_AIS(self):
        """Places the target ships close the OS by reading an AIS-data file."""
        # initialize current time stamp
        OS_lat, OS_lon = to_latlon(north=self.OS.eta[0], east=self.OS.eta[1], number=32)
        tpos = TimePosition(timestamp=self.t_travel, lat=OS_lat, lon=OS_lon)
        tpos.timestamp = tpos.timestamp.replace(tzinfo=None)
        
        # get target ships
        target_ships = self.AIS_search_agent.get_ships(tpos)

        self.TSs = []
        for ship in target_ships:
            lat, lon, head, spd = ship.observe()

            # transformations
            n, e, _ = to_utm(lat=lat, lon=lon)
            head = dtr(head)
            spd = knots_to_mps(spd)

            # init ship
            TS = TargetShip(N_init    = n,
                            E_init    = e,
                            psi_init  = head,
                            u_init    = spd,
                            v_init    = 0.0,
                            r_init    = 0.0,
                            delta_t   = self.delta_t,
                            N_max     = np.infty,
                            E_max     = np.infty,
                            nps       = None,
                            full_ship = False,
                            ship_domain_size = 2)
            TS.nps = TS._get_nps_from_u(TS.nu[0], psi=TS.eta[2])
            self.TSs.append(TS) 
        self.N_TSs = len(self.TSs)

    def setup_planning_env(self, env:HHOS_RiverPlanning_Env, initial:bool):
        """Prepares the planning env depending on the current status of the following env. Returns the env if initial, else
        returns env, state."""
        if initial:
            # number of target ships
            env.N_TSs = self.N_TSs

            # global path
            env.GlobalPath = deepcopy(self.GlobalPath)
            if hasattr(self, "RevGlobalPath"):
                env.RevGlobalPath = deepcopy(self.RevGlobalPath)

            # environmental disturbances (although not needed for dynamics, only for depth-checking)
            env.DepthData   = deepcopy(self.DepthData)
            env.log_Depth   = deepcopy(self.log_Depth)
            env.WindData    = deepcopy(self.WindData)
            env.CurrentData = deepcopy(self.CurrentData)
            env.WaveData    = deepcopy(self.WaveData)

            # visualization specs
            env.domain_xs = self.domain_xs
            env.domain_ys = self.domain_ys
            env.con_ticks = self.con_ticks
            env.con_ticklabels = self.con_ticklabels
            env.clev = self.clev

            # overwrite '_done()' function of planning env
            env._done = self._done
            return env
        else:
            # step count
            env.step_cnt = 0

            # OS and TSs
            env.OS  = deepcopy(self.OS)
            env.TSs = deepcopy(self.TSs)
            env.H   = copy(self.H)

            # guarantee OS moves linearly
            env.OS.nu[0]  = self.OS._get_V()
            env.OS.nu[1:] = 0.0
            env.OS.eta[2] = self.OS._get_course()
            env.OS.rud_angle = 0.0

            # global path error
            env.glo_ye = self.glo_ye
            env.glo_desired_course = self.glo_desired_course
            env.glo_course_error = self.glo_course_error
            env.glo_pi_path = self.glo_pi_path

            # set state in planning env and return it
            env._set_state()
            return env, env.state

    def step(self, a:np.ndarray):
        """Takes an action and performs one step in the environment.
        Returns new_state, r, done, {}."""

        # perform control action
        self._OS_control(a)

        # update agent dynamics
        self.OS._upd_dynamics(V_w=self.V_w, beta_w=self.beta_w, V_c=self.V_c, beta_c=self.beta_c, H=self.H, 
                              beta_wave=self.beta_wave, eta_wave=self.eta_wave, T_0_wave=self.T_0_wave, lambda_wave=self.lambda_wave)

        # environmental effects
        self._update_disturbances()

        # update target ships
        if self.AIS_data_file is None:

            for i, TS in enumerate(self.TSs):
                # update waypoints
                try:
                    self.TSs[i] = self._init_wps(TS, "global")
                    cnt = True
                except:
                    cnt = False

                # simple heading control
                if cnt:
                    other_vessels = [ele for ele in self.TSs if ele is not TS] + [self.OS]
                    TS.river_control(other_vessels, VFG_K=self.VFG_K_TS)

            # update TS dynamics (independent of environmental disturbances since they move linear and deterministic)
            [TS._upd_dynamics() for TS in self.TSs]

            # check respawn
            self.TSs = [self._handle_respawn(TS) for TS in self.TSs]

        else:
            # update travel time
            self.t_travel += timedelta(seconds=self.delta_t)

            # read AIS data
            self._set_TSs_from_AIS()

        # set the local path
        if self.step_cnt % self.loc_path_upd_freq == 0:
            self._update_local_path()

        # update OS waypoints of global and local path
        self.OS:KVLCC2 = self._init_wps(self.OS, "global")
        self.OS:KVLCC2 = self._init_wps(self.OS, "local")

        # compute new cross-track error and course error (for local and global path)
        self._set_cte(path_level="global")
        self._set_cte(path_level="local")
        self._set_ce(path_level="global")
        self._set_ce(path_level="local")

        # increase step cnt and overall simulation time
        self.sim_t += self.delta_t
        self.step_cnt += 1
        
        # compute state, reward, done        
        self._set_state()
        self._calculate_reward(a)
        d = self._done()

        # viz
        TS_info = {}
        for i in range(self.N_TSs_max):
            try:
                # positions and speed
                TS = self.TSs[i]
                TS_info[f"TS{str(i)}_N"] = TS.eta[0]
                TS_info[f"TS{str(i)}_E"] = TS.eta[1]
                TS_info[f"TS{str(i)}_head"] = TS.eta[2]
                TS_info[f"TS{str(i)}_V"] = TS._get_V()

                # distance
                D = get_ship_domain(A=self.OS.ship_domain_A, B=self.OS.ship_domain_B, C=self.OS.ship_domain_C, D=self.OS.ship_domain_D,
                                    OS=self.OS, TS=TS)               
                TS_info[f"TS{str(i)}_dist"] = ED(N0=self.OS.eta[0], E0=self.OS.eta[1], N1=TS.eta[0], E1=TS.eta[1]) - D
            except:
                TS_info[f"TS{str(i)}_N"] = None
                TS_info[f"TS{str(i)}_E"] = None
                TS_info[f"TS{str(i)}_head"] = None
                TS_info[f"TS{str(i)}_V"] = None
                TS_info[f"TS{str(i)}_dist"] = None

        self.plotter.store(sim_t=self.sim_t, OS_N=self.OS.eta[0], OS_E=self.OS.eta[1], OS_head=self.OS.eta[2], OS_u=self.OS.nu[0],\
                OS_v=self.OS.nu[1], OS_r=self.OS.nu[2], loc_ye=self.loc_ye, glo_ye=self.glo_ye, loc_course_error=self.loc_course_error,\
                    glo_course_error=self.glo_course_error, V_c=self.V_c, beta_c=self.beta_c, V_w=self.V_w, beta_w=self.beta_w,\
                    T_0_wave=self.T_0_wave, eta_wave=self.eta_wave, beta_wave=self.beta_wave, lambda_wave=self.lambda_wave,\
                         rud_angle=self.OS.rud_angle, nps=self.OS.nps, **TS_info)
        return self.state, self.r, d, {}

    def _OS_control(self, a:np.ndarray):
        """Performs the control action for the own ship."""
        # store for viz
        a = a.flatten()
        self.a = a

        # make sure array has correct size
        assert len(a) == 1, "There needs to be one action for the follower."

        # rudder control
        assert -1 <= float(a[0]) <= 1, "Unknown action."
        self.OS.rud_angle = np.clip(self.OS.rud_angle + float(a[0])*self.rud_angle_inc, -self.rud_angle_max, self.rud_angle_max)

    def _hasConflict(self, localPath:Path, TSnavData : Union[List[Path], None], path_from:str):
        """Checks whether a given local path yields a conflict.
        
        Args:
            localPath(Path): path of the OS
            TSnavData(list): contains Path-objects, one for each TS. If None, the planning_env is used to generate this data.
            path_from(str):  either 'global' or 'RL'. If 'global', checks for conflicts in the sense of collisions AND traffic rules. 
                             If 'RL', checks only for collisions.
        Returns:
            bool, conflict (True) or no conflict (False)"""
        assert path_from in ["global", "RL"], "Use either 'global' or 'RL' for conflict checking."

        # interpolate path of OS
        atts_to_interpolate = ["north", "east", "heads", "chis", "vs"]

        for att in atts_to_interpolate:
            angle = True if att in ["heads", "chis"] else False
            localPath.interpolate(attribute=att, n_wps_between=5, angle=angle)

        # update lat and lon from interpolated north and east
        localPath.lat, localPath.lon = to_latlon(north=localPath.north, east=localPath.east, number=32)
        
        # always check for collisions with land
        for i in range(len(localPath.lat)):
            if self._depth_at_latlon(lat_q=localPath.lat[i], lon_q=localPath.lon[i]) <= self.OS.critical_depth:
                return True    

        # create TSnavData if not existent
        if TSnavData is None:
            TSnavData = self._update_local_path_safe(method=None)
        
        # interpolate paths of TS to make sure to detect collisions
        for path in TSnavData:
            for att in atts_to_interpolate:
                angle = True if att in ["heads", "chis"] else False
                path.interpolate(attribute=att, n_wps_between=5, angle=angle)

        #----- check for target ship collisions in both methods -----
        n_TS  = len(TSnavData)
        n_wps = len(TSnavData[0].north)

        for t in range(n_wps):
            for i in range(n_TS):

                # quick access
                N0, E0, head0 = localPath.north[t], localPath.east[t], localPath.heads[t]
                N1, E1, head1 = TSnavData[i].north[t], TSnavData[i].east[t], TSnavData[i].heads[t]

                # compute ship domain
                ang = bng_rel(N0=N0, E0=E0, N1=N1, E1=E1, head0=head0)
                D = get_ship_domain(A=self.OS.ship_domain_A, B=self.OS.ship_domain_B, C=self.OS.ship_domain_C, D=self.OS.ship_domain_D,
                                    ang=ang, OS=None, TS=None)
            
                # check if collision
                ED_OS_TS = ED(N0=N0, E0=E0, N1=N1, E1=E1, sqrt=True)
                if ED_OS_TS <= D:
                    return True

        #----- check for traffic rules only when the path comes from the global path -----
        if path_from == "global":
            for t in range(n_wps):
                for i in range(n_TS):

                    # quick access
                    N0, E0, head0 = localPath.north[t], localPath.east[t], localPath.heads[t]
                    v0 = localPath.vs[t]
                    
                    N1, E1, head1 = TSnavData[i].north[t], TSnavData[i].east[t], TSnavData[i].heads[t]
                    v1 = TSnavData[i].vs[t]

                    # check whether the global path violates some rules
                    if self._violates_river_traffic_rules(N0=N0, E0=E0, head0=head0, v0=v0, N1=N1, E1=E1, 
                                                          head1=head1, v1=v1):
                        return True
        return False

    def _update_local_path(self):
        """Updates the local path either by simply using the global path, using the RL-based path planner, or the APF safety net."""
        # default: set part of global path as local path
        self._init_local_path()

        # replan via RL if someone is close
        if self.N_TSs > 0:
            if np.min([ED(N0=self.OS.eta[0], E0=self.OS.eta[1], N1=t.eta[0], E1=t.eta[1]) for t in self.TSs]) <= self.sight_river:
                TSnavData:List[Path] = self._update_local_path_safe(method="RL")
                self.planning_method = "RL"
            """
            conflict = self._hasConflict(localPath=copy(self.LocalPath), TSnavData=None, path_from="global")
            if conflict:
                TSnavData:List[Path] = self._update_local_path_safe(method="RL")
                self.planning_method = "RL"

                # if we use safety net, check only for collisions
                if self.planner_safety_net:
                    conflict = self._hasConflict(localPath=copy(self.LocalPath), TSnavData=TSnavData, path_from="RL")
                    if conflict:
                        self._update_local_path_safe(method="APF")
                        self.planning_method = "APF"
            """

    def _update_local_path_safe(self, method:Union[str, None]):
        """Updates the local path based on the RL or the APF planner. Can alternatively be used to solely generate TSNavData.
        
        Args:
            method(str): 'RL', 'APF', or None
        Returns:
            List[Path]: TSNavData"""
        assert method in [None, "RL", "APF"], "Use either 'RL' or 'APF' to safely update the local path. \
            Alternatively, use None to generate TSNavData."

        # prep planning env
        self.planning_env, s = self.setup_planning_env(self.planning_env, initial=False)

        # setup OS wps and speed in RL or APF case
        if method in ["RL", "APF"]:
            ns, es, heads = [self.planning_env.OS.eta[0]], [self.planning_env.OS.eta[1]], [self.planning_env.OS.eta[2]]
            chis, vs = [self.planning_env.OS._get_course()], [self.planning_env.OS._get_V()]

        # TSNavData in RL or None mode
        if method in [None, "RL"]:
            TS_ns      = [[TS.eta[0]] for TS in self.planning_env.TSs]
            TS_es      = [[TS.eta[1]] for TS in self.planning_env.TSs]
            TS_heads   = [[TS.eta[2]] for TS in self.planning_env.TSs]
            TS_chis    = [[TS._get_course()] for TS in self.planning_env.TSs]
            TS_vs      = [[TS._get_V()] for TS in self.planning_env.TSs]

        # setup history
        state_shape = 4 + 7 * max([self.N_TSs_max, 1]) + self.lidar_n_beams
        s_hist = np.zeros((2, state_shape))  # history length 2
        hist_len = 0

        # planning loop
        for _ in range(self.n_wps_loc-1):

            # planner's move
            if method == "RL":

                # recursive state design needs history
                s_tens        = torch.tensor(s, dtype=torch.float32).view(1, state_shape)
                s_hist_tens   = torch.tensor(s_hist, dtype=torch.float32).view(1, 2, state_shape) # batch size, history length, state shape
                hist_len_tens = torch.tensor(hist_len)
                a = self.planner(s=s_tens, s_hist=s_hist_tens, a_hist=None, hist_len=hist_len_tens)[0]

            # get APF move, always two components
            elif method == "APF":
                raise NotImplementedError()
                du, dh = self._get_APF_move(self.planning_env)

                # apply heading change
                self.planning_env.OS.eta[2] = angle_to_2pi(self.planning_env.OS.eta[2] + dh)

                # apply longitudinal speed change
                self.planning_env.OS.nu[0] = np.clip(self.planning_env.OS.nu[0] + du, \
                    self.planning_env.surge_min, self.planning_env.surge_max)

                # set nps accordingly
                self.planning_env.OS.nps = self.planning_env.OS._get_nps_from_u(self.planning_env.OS.nu[0])

                # need to define dummy zero-actions since we control the OS outside the planning env
                a = np.array([0.0])

            # TSnavData generation - use dummy zero-actions (although it does not matter anyways, we only extract TS info)
            elif method is None:
                a = np.array([0.0])

            # planning env's reaction
            s2, _, d, _ = self.planning_env.step(a, control_TS=False)

            if method == "RL":

                # update history
                if hist_len == 2:
                    s_hist = np.roll(s_hist, shift=-1, axis=0)
                    s_hist[1, :] = s
                else:
                    s_hist[hist_len] = s
                    hist_len += 1

                # s becomes s2
                s = s2

            # store wps and speed
            if method in ["RL", "APF"]:
                ns.append(self.planning_env.OS.eta[0])
                es.append(self.planning_env.OS.eta[1])
                heads.append(self.planning_env.OS.eta[2])
                chis.append(self.planning_env.OS._get_course())
                vs.append(self.planning_env.OS._get_V())

            # TSNavData in RL or None mode
            if method in ["RL", None]:
                for i, TS in enumerate(self.planning_env.TSs):
                    TS_ns[i].append(TS.eta[0])
                    TS_es[i].append(TS.eta[1])
                    TS_heads[i].append(TS.eta[2])
                    TS_chis[i].append(TS._get_course())
                    TS_vs[i].append(TS._get_V())
            if d:
                break

        # update the path in RL or APF mode
        if method in ["RL", "APF"]:
            self.LocalPath = Path(level="local", north=ns, east=es, heads=heads, vs=vs, chis=chis)

        # generate TSnavData object for conflict detection
        if method in ["RL", None]:
            TSnavData = []
            for i in range(len(TS_ns)):
                TSnavData.append(Path(level="local", north=TS_ns[i], east=TS_es[i], heads=TS_heads[i], vs=TS_vs[i], chis=TS_chis[i]))
            return TSnavData

    def _get_APF_move(self, env : HHOS_RiverPlanning_Env):
        """Takes an env as input and returns the du, dh suggested by the APF-method."""
        raise NotImplementedError()
        # define goal for attractive forces of APF
        try:
            g_idx = env.OS.glo_wp3_idx + 3
            g_n = env.GlobalPath.north[g_idx]
            g_e = env.GlobalPath.east[g_idx]
        except:
            g_idx = env.OS.glo_wp3_idx
            g_n = env.GlobalPath.north[g_idx]
            g_e = env.GlobalPath.east[g_idx]

        # consider river geometry
        N0, E0, head0 = env.OS.eta
        dists, _, river_n, river_e = env.sense_LiDAR(N0=N0, E0=E0, head0=head0)
        i = np.where(dists != env.lidar_range)

        # computation
        du, dh = apf(OS=env.OS, TSs=env.TSs, G={"x" : g_e, "y" : g_n}, 
                     river_n=river_n[i], river_e=river_e[i],
                     du_clip=env.surge_scale, dh_clip=env.d_head_scale)
        return du, dh

    def _set_state(self):
        #--------------------------- OS information ----------------------------
        cmp1 = self.OS.nu / np.array([3.0, 0.2, 0.002])                # u, v, r
        cmp2 = np.array([self.OS.nu_dot[2] / (8e-5), self.OS.rud_angle / self.OS.rud_angle_max])   # r_dot, rudder angle
        state_OS = np.concatenate([cmp1, cmp2])

        # ------------------------- local path information ---------------------------
        state_path = np.array([self.loc_ye/self.OS.Lpp, self.loc_course_error/math.pi])

        # -------------------- environmental disturbances ----------------------
        if any([pd.isnull(ele) or np.isinf(ele) or ele is None for ele in\
             [self.beta_wave, self.eta_wave, self.T_0_wave, self.lambda_wave]]):
            beta_wave = 0.0
            eta_wave = 0.0
            T_0_wave = 0.0
            lambda_wave = 0.0
        else:
            beta_wave = self.beta_wave
            eta_wave = self.eta_wave
            T_0_wave = self.T_0_wave
            lambda_wave = self.lambda_wave

        head0 = self.OS.eta[2]
        state_env = np.array([self.V_c/0.5,  angle_to_pi(self.beta_c-head0)/(math.pi),      # currents
                              self.V_w/15.0, angle_to_pi(self.beta_w-head0)/(math.pi),      # winds
                              angle_to_pi(beta_wave-head0)/(math.pi), eta_wave/2.0, T_0_wave/7.0, lambda_wave/100.0,    # waves
                              self.H/100.0])    # depth

        # ------------------------- aggregate information ------------------------
        self.state = np.concatenate([state_OS, state_path, state_env]).astype(np.float32)

    def _calculate_reward(self, a):
        self.r = 0.0
        self.r_ye = 0.0
        self.r_ce = 0.0
        self.r_comf = 0.0

    def _done(self):
        """Returns boolean flag whether episode is over."""
        d = False

        # OS hit land
        if self.H <= self.OS.critical_depth:
            d = True

        # reach end of path
        #elif self.OS.glo_wp3_idx >= (self.GlobalPath.n_wps-1):
        elif self.OS.glo_wp3_idx >= 420:
            d = True

        # artificial done signal
        elif self.step_cnt >= self._max_episode_steps:
            d = True

        # viz
        if d:
            s = "calm" if "calm" in self.AIS_data_file else "storm"
            self.plotter.dump(name=f"Pipeline_{s}_{str(int(self.base_speed))}")
        return d

    #def render(self, data=None):
    #    if self.step_cnt % 100 == 0:
    #        print(self.step_cnt)
