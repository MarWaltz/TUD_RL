from tud_rl.envs._envs.HHOS_Fnc import HHOSPlotter
from tud_rl.envs._envs.HHOS_RiverPlanning_Env import *


class HHOS_RiverPlanning_Validation(HHOS_RiverPlanning_Env):
    """Does not consider any environmental disturbances since this is considered by the local-path following unit."""
    def __init__(self, 
                 scenario : bool,
                 river_curve: str):
        self.scenario    = scenario
        self.river_curve = river_curve
        
        assert self.scenario in range(1, 7), "Unknown validation scenario for the river."
        assert self.river_curve in ["straight", "left", "right"], "Unknown river curvature."

        # vessel train
        if self.scenario == 1:
            self.N_TSs = 4
        
        # overtake the overtaker
        elif self.scenario == 2:
            self.N_TSs = 2

        # overtaking under oncoming traffic
        elif self.scenario == 3:
            self.N_TSs = 3
        
        # overtake the overtaker under oncoming traffic
        elif self.scenario == 4:
            self.N_TSs = 4

        # getting overtaken
        elif self.scenario == 5:
            self.N_TSs = 1

        # static obstacles
        elif self.scenario == 6:
            self.N_TSs = 5

        super().__init__(N_TSs_max=self.N_TSs, N_TSs_random=False, 
                         w_ye=.0, w_ce=.0, w_coll=.0, w_rule=.0, w_comf=.0)
        
        # path characteristics
        if river_curve == "straight":
            self.path_config = {"n_seg_path" : 10, "straight_wp_dist" : 50, "straight_lmin" :1000, "straight_lmax" :1000, 
                                "phi_min" : None, "phi_max" : None, "rad_min" : None, "rad_max" : None, "build" : "straight"}
        elif river_curve == "left":
            self.path_config = {"n_seg_path" : 2, "straight_wp_dist" : 50, "straight_lmin" : 50, "straight_lmax" : 50, 
                                "phi_min" : 120, "phi_max" : 120, "rad_min" : 2000, "rad_max" : 2000, "build" : "left_curved"}
        else:
            self.path_config = {"n_seg_path" : 2, "straight_wp_dist" : 50, "straight_lmin" :50, "straight_lmax" : 50, 
                                "phi_min" : 120, "phi_max" : 120, "rad_min" : 2000, "rad_max" : 2000, "build" : "right_curved"}

        # depth configuration
        self.depth_config = {"validation" : True}

        self._max_episode_steps = 100

    def reset(self):
        s = super().reset()

        # viz
        TS_info = {}
        for i, TS in enumerate(self.TSs):
            TS_info[f"TS{str(i)}_N"] = TS.eta[0]
            TS_info[f"TS{str(i)}_E"] = TS.eta[1]
            TS_info[f"TS{str(i)}_head"] = TS.eta[2]
            TS_info[f"TS{str(i)}_V"] = TS._get_V()

        self.plotter = HHOSPlotter(sim_t=self.sim_t, a=0.0, OS_N=self.OS.eta[0], OS_E=self.OS.eta[1], OS_head=self.OS.eta[2], OS_V=self.OS._get_V(), OS_u=self.OS.nu[0],\
                OS_v=self.OS.nu[1], OS_r=self.OS.nu[2], glo_ye=self.glo_ye, glo_course_error=self.glo_course_error, **TS_info)
        return s

    def step(self, a):
        """Performs a given action in the environment and transitions to the next state."""
        s, r, d, info = super().step(a, control_TS=True)

        # viz
        if not d:
            TS_info = {}
            for i, TS in enumerate(self.TSs):
                TS_info[f"TS{str(i)}_N"] = TS.eta[0]
                TS_info[f"TS{str(i)}_E"] = TS.eta[1]
                TS_info[f"TS{str(i)}_head"] = TS.eta[2]
                TS_info[f"TS{str(i)}_V"] = TS._get_V()

            self.plotter.store(sim_t=self.sim_t, a=float(self.a), OS_N=self.OS.eta[0], OS_E=self.OS.eta[1], OS_head=self.OS.eta[2], OS_V=self.OS._get_V(), OS_u=self.OS.nu[0],\
                    OS_v=self.OS.nu[1], OS_r=self.OS.nu[2], glo_ye=self.glo_ye, glo_course_error=self.glo_course_error, **TS_info)
        return s, r, d, info

    def _handle_respawn(self, TS: TargetShip):
        return TS

    def _init_TSs(self):
        self.TSs : List[TargetShip]= []
        for n in range(self.N_TSs):
            self.TSs.append(self._get_TS_river(scenario=self.scenario, n=n))
        
        # deterministic behavior in evaluation
        for TS in self.TSs:
            TS.random_moves = False

    def _done(self):
        d = super()._done()

        # viz
        if d:
            self.plotter.DepthData     = self.DepthData
            self.plotter.GlobalPath    = self.GlobalPath
            self.plotter.RevGlobalPath = self.RevGlobalPath
            self.plotter.dump(name="Plan_river_" + self.river_curve + "_" + str(self.scenario))
        return d

    def render(self, data=None):
        pass
