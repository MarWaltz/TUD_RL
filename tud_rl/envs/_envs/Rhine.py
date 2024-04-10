import math
import pickle
import random
from copy import deepcopy
from typing import List

import gym
import numpy as np
import seaborn as sns
from gym import spaces
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

from tud_rl.envs._envs.HHOS_Fnc import (NM_to_meter, angle_to_2pi, ate,
                                        bng_abs, dtr, get_init_two_wp,
                                        xy_from_polar)
from tud_rl.envs._envs.MMG_KVLCC2 import KVLCC2
from tud_rl.envs._envs.MMG_TargetShip import Path, TargetShip
from tud_rl.envs._envs.VesselFnc import rtd
from tud_rl.envs._envs.VesselPlots import rotate_point


class Rhine(gym.Env):
    """This environment contains an agent steering a KVLCC2 vessel over the Rhine based on image input."""
    def __init__(self, data_file:str, N_TSs_random:bool, N_TSs_max:int):
        super().__init__()

        # simulation settings
        self.delta_t = 5.0     # simulation time interval (in s)
        self.base_speed = 3.0  # default surge velocity (in m/s)

        self.enc_range_min = NM_to_meter(0.25)    # lower distance when we consider encounter situations on the river
        self.enc_range_max = NM_to_meter(0.50)

        self.N_TSs_random = N_TSs_random
        self.N_TSs_max = N_TSs_max
        self.VFG_K_TS = 0.005       

        # load data
        self.data_file = data_file
        self._load_data()

        # visualization
        self.default_cols = sns.color_palette("husl", 30)
        self.img_width  = 320 *2 # px
        self.img_heigth = 320 *2 # px
        self.mpp = 3 # meter per px

    def _load_data(self):
        with open(self.data_file, "rb") as f:
            self.data = pickle.load(f)

        # make left path have increasing indices from north to south, and right part the other opposite 
        self.LeftPath = Path(level = "global", 
                             north = self.data["axis_n"][self.data["axis_km_left"]],
                             east  = self.data["axis_e"][self.data["axis_km_left"]])
        self.RightPath = Path(level = "global", 
                              north = self.data["axis_n"][self.data["axis_km_left"]],
                              east  = self.data["axis_e"][self.data["axis_km_left"]])
        self.LeftPath.reverse(0.0)
        self.RightPath.reverse(0.0)

    def reset(self):
       self.step_cnt = 0           # simulation step counter
       self.sim_t    = 0           # overall passed simulation time (in s)

       # init OS and traveling direction
       OS_goes_n = bool(random.getrandbits(1))
       
       if OS_goes_n:
           # set OS on the right path
           i = np.random.randint(low=100, high=self.RightPath.n_wps-100)
           N_init = self.RightPath.north[i]
           E_init = self.RightPath.east[i]
       else:
           # set OS on the left path
           i = np.random.randint(low=100, high=self.LeftPath.n_wps-100)
           N_init = self.LeftPath.north[i]
           E_init = self.LeftPath.east[i]

       print(OS_goes_n)

       # consider different speeds in training
       if "Validation" in type(self).__name__:
           spd = self.base_speed
       else:
           spd = float(np.random.uniform(0.8, 1.2)) * self.base_speed

       self.OS = KVLCC2(N_init    = N_init, 
                        E_init    = E_init, 
                        psi_init  = 0.0,
                        u_init    = spd,
                        v_init    = 0.0,
                        r_init    = 0.0,
                        delta_t   = self.delta_t,
                        N_max     = np.infty,
                        E_max     = np.infty,
                        nps       = None,
                        full_ship = False,
                        ship_domain_size = 2)

       # store path directly as OS attribute
       self.OS.goes_n = OS_goes_n
       self.OS.path    = deepcopy(self.RightPath if OS_goes_n else self.LeftPath)
       self.OS.Revpath = deepcopy(self.LeftPath if OS_goes_n else self.RightPath)

       # init waypoints of OS for path
       self.OS = self._init_wps(self.OS)

       # set heading with noise in training
       hdg = bng_abs(N0=self.OS.path.north[self.OS.wp1_idx], E0=self.OS.path.east[self.OS.wp1_idx],
                     N1=self.OS.path.north[self.OS.wp2_idx], E1=self.OS.path.east[self.OS.wp2_idx])

       if "Validation" in type(self).__name__:
           self.OS.eta[2] = hdg
       else:
           self.OS.eta[2] = angle_to_2pi(hdg + dtr(np.random.uniform(-25.0, 25.0)))

       # depth updating
       self._update_disturbances()

       # set nps to near-convergence
       self.OS.nps = self.OS._get_nps_from_u(u=self.OS.nu[0], psi=self.OS.eta[2])

       # init other vessels
       self._init_TSs()

       # init state
       self._set_state()
       return self.state

    def _init_wps(self, vessel:KVLCC2):
        """Initializes the waypoints on the vessel's path based on the position of the vessel.
        Returns the vessel."""
        path = vessel.path
        vessel.wp1_idx, vessel.wp1_N, vessel.wp1_E, vessel.wp2_idx, vessel.wp2_N, \
            vessel.wp2_E = get_init_two_wp(n_array=path.north, e_array=path.east, a_n=vessel.eta[0], a_e=vessel.eta[1])
        try:
            vessel.wp3_idx = vessel.wp2_idx + 1
            vessel.wp3_N = path.north[vessel.wp3_idx] 
            vessel.wp3_E = path.east[vessel.wp3_idx]
        except:
            vessel.wp3_idx = vessel.wp2_idx
            vessel.wp3_N = path.north[vessel.wp3_idx] 
            vessel.wp3_E = path.east[vessel.wp3_idx]
        return vessel

    def _init_TSs(self):
        # scenario = 0 means all TS random, no manual configuration
        if self.N_TSs_random:
            assert self.N_TSs_max == 10, "Go for maximum 10 TSs in HHOS planning."
            self.N_TSs = np.random.choice(self.N_TSs_max + 1) # np.random.choice([0, 1, 2, 3], p=[0.1, 0.3, 0.3, 0.3])
        else:
            self.N_TSs = self.N_TSs_max

        # sample TSs
        self.TSs : List[TargetShip]= []
        for n in range(self.N_TSs):
            TS = None
            while TS is None:
                #try:
                TS = self._get_TS_river()
                #except:
                #    pass
            self.TSs.append(TS)

    def _get_TS_river(self):
        """Places a target ship by setting a 
            1) traveling direction,
            2) distance on the global path,
        depending on the scenario. All ships spawn in front of the agent.
        Args:
            scenario (int):  considered scenario
            n (int):      index of the spawned vessel
        Returns: 
            KVLCC2."""
        #------------------ set distances, directions, offsets from path, and nps ----------------------
        # Note: An offset is some float. If it is negative (positive), the vessel is placed on the 
        #       right (left) side of the global path.

        # random
        speedy = bool(np.random.choice([0, 1], p=[0.85, 0.15]))

        if speedy: 
            rev_dir = False
            spd     = np.random.uniform(1.3, 1.5) * self.base_speed
            d       = self.enc_range_max + np.random.uniform(low=-NM_to_meter(0.2), high=NM_to_meter(0.2))
        else:
            rev_dir = bool(random.getrandbits(1))
            spd     = np.random.uniform(0.4, 0.8) * self.base_speed

            if rev_dir:
                d = 3*self.enc_range_max + np.random.uniform(low=-NM_to_meter(0.4), high=NM_to_meter(0.4))
            else:
                d = self.enc_range_max + np.random.uniform(low=-NM_to_meter(0.2), high=NM_to_meter(0.2))
        offset = np.random.uniform(-20.0, 50.0)

        # get wps
        if speedy:
            path = deepcopy(self.OS.Revpath)
            wp1, wp1_N, wp1_E, wp2, wp2_N, wp2_E = get_init_two_wp(n_array = path.north, 
                                                                   e_array = path.east, 
                                                                   a_n     = self.OS.eta[0], 
                                                                   a_e     = self.OS.eta[1])
        else:
            wp1 = self.OS.wp1_idx
            wp1_N = self.OS.wp1_N
            wp1_E = self.OS.wp1_E

            wp2 = self.OS.wp2_idx
            wp2_N = self.OS.wp2_N
            wp2_E = self.OS.wp2_E

            path = deepcopy(self.OS.path)

        # determine starting position
        ate_init = ate(N1=wp1_N, E1=wp1_E, N2=wp2_N, E2=wp2_E, NA=self.OS.eta[0], EA=self.OS.eta[1])
        d_to_nxt_wp = path.wp_dist(wp1, wp2) - ate_init
        orig_seg = True

        while True:
            if d > d_to_nxt_wp:
                d -= d_to_nxt_wp
                wp1 += 1
                wp2 += 1
                d_to_nxt_wp = path.wp_dist(wp1, wp2)
                orig_seg = False
            else:
                break

        # path angle
        pi_path_spwn = bng_abs(N0=path.north[wp1], E0=path.east[wp1], N1=path.north[wp2], E1=path.east[wp2])

        # still in original segment
        if orig_seg:
            E_add, N_add = xy_from_polar(r=ate_init+d, angle=pi_path_spwn)
        else:
            E_add, N_add = xy_from_polar(r=d, angle=pi_path_spwn)

        # determine position
        N_TS = path.north[wp1] + N_add
        E_TS = path.east[wp1] + E_add
        
        # jump on the other path: either due to speedy or opposing traffic
        if speedy or rev_dir:
            if rev_dir:
                pathJump = deepcopy(self.OS.Revpath)
            elif speedy:
                pathJump = deepcopy(self.OS.path)
            
            _, wp1_N, wp1_E, _, wp2_N, wp2_E = get_init_two_wp(n_array=pathJump.north, e_array=pathJump.east, a_n=N_TS, a_e=E_TS)
            ate_TS = ate(N1=wp1_N, E1=wp1_E, N2=wp2_N, E2=wp2_E, NA=N_TS, EA=E_TS)

            E_add, N_add = xy_from_polar(r=ate_TS, angle=bng_abs(N0=wp1_N, E0=wp1_E, N1=wp2_N, E1=wp2_E))
            N_TS += N_add
            E_TS += E_add

        # consider offset
        TS_head = angle_to_2pi(pi_path_spwn + math.pi) if rev_dir or speedy else pi_path_spwn

        if offset != 0.0:
            ang = TS_head - math.pi/2 if offset > 0.0 else TS_head + math.pi/2
            E_add_rev, N_add_rev = xy_from_polar(r=abs(offset), angle=angle_to_2pi(ang))
            N_TS += N_add_rev
            E_TS += E_add_rev

        # generate TS
        TS = TargetShip(N_init    = N_TS,
                        E_init    = E_TS,
                        psi_init  = TS_head,
                        u_init    = spd,
                        v_init    = 0.0,
                        r_init    = 0.0,
                        delta_t   = self.delta_t,
                        N_max     = np.infty,
                        E_max     = np.infty,
                        nps       = None,
                        full_ship = False,
                        ship_domain_size = 2)
        TS.goes_n = self.OS.goes_n if not rev_dir else not self.OS.goes_n

        # set path and waypoints
        TS.path = deepcopy(self.OS.Revpath) if rev_dir else deepcopy(self.OS.path)

        wp1, wp1_N, wp1_E, wp2, wp2_N, wp2_E = get_init_two_wp(n_array = TS.path.north, 
                                                               e_array = TS.path.east, 
                                                               a_n     = TS.eta[0], a_e     = TS.eta[1])
        TS.wp1_idx = wp1
        TS.wp1_N = wp1_N
        TS.wp1_E = wp1_E
        TS.wp2_idx = wp2
        TS.wp2_N = wp2_N
        TS.wp2_E = wp2_E
        TS.wp3_idx = wp2 + 1
        TS.wp3_N = TS.path.north[wp2 + 1]
        TS.wp3_E = TS.path.east[wp2 + 1]

        # predict converged speed of sampled TS
        TS.nps = TS._get_nps_from_u(TS.nu[0], psi=TS.eta[2])
        return TS

    def _update_disturbances(self):
        pass

    def step(self, a):
        """Takes an action and performs one step in the environment.
        Returns new_state, r, done, {}."""
        # control action
        self._manual_control(a)

        # TO-DO: remove this part
        self.OS.eta[2] = bng_abs(N0=self.OS.wp1_N, E0=self.OS.wp1_E, N1=self.OS.wp2_N, E1=self.OS.wp2_E)

        # update agent dynamics (TO-DO: Add currents)
        self.OS._upd_dynamics()

        # update environmental effects
        self._update_disturbances()

        # update OS waypoints (purely for generating target ships)
        self.OS:KVLCC2= self._init_wps(self.OS)

        for i, TS in enumerate(self.TSs):
            # update waypoints
            try:
                self.TSs[i] = self._init_wps(TS)
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

        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += self.delta_t

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward()
        d = self._done()
        return self.state, self.r, d, {}

    def _handle_respawn(self, TS:TargetShip):
        return TS

    def _manual_control(self, a:np.ndarray):
        """Manually controls heading and surge of the own ship."""
        return
        a = a.flatten()
        self.a = a

        # make sure array has correct size
        assert len(a) == 1, "There needs to be one action for the planner."

        # heading control
        assert -1 <= float(a[0]) <= 1, "Unknown action."
        self.OS.eta[2] = angle_to_2pi(self.OS.eta[2] + float(a[0])*self.d_head_scale)

    def _set_state(self):
        self.state = None

    def _calculate_reward(self):
        self.r = 0.0

    def _done(self):
        return False

    def __str__(self, OS_lat, OS_lon) -> str:
        u, v, r = self.OS.nu
        course = self.OS._get_course()

        ste = f"Step: {self.step_cnt}"
        if hasattr(self, "TSs"):
            ste += f", TSs: {len(self.TSs)}"
        pos = f"Lat [°]: {OS_lat:.4f}, Lon [°]: {OS_lon:.4f}, " + r"$\psi$ [°]: " + f"{rtd(self.OS.eta[2]):.2f}"  + r", $\chi$ [°]: " + f"{rtd(course):.2f}"
        vel = f"u [m/s]: {u:.3f}, v [m/s]: {v:.3f}, r [rad/s]: {r:.3f}"
        out = ste + ", " + pos + "\n" + vel
        
        if hasattr(self, "DepthData"):
            depth = f"H [m]: {self.H:.2f}"
            out = out + "," + depth

        if hasattr(self, "CurrentData"):
            current = r"$V_{\rm current}$" + f" [m/s]: {self.V_c:.2f}, " + r"$\psi_{\rm current}$" + f" [°]: {rtd(self.beta_c):.2f}"
            out = out + current
        return out

    def _render_ship(self, ax:plt.Axes, vessel:KVLCC2, color:str):
        """Draws the ship on the axis, including ship domain. Returns the ax."""
        # quick access
        l = vessel.Lpp/2
        b = vessel.B/2
        N0, E0, head0 = self.OS.eta
        N1, E1, head1 = vessel.eta

        # get rectangle/polygon end points in UTM
        A = (E1 - b, N1 + l)
        B = (E1 + b, N1 + l)
        C = (E1 - b, N1 - l)
        D = (E1 + b, N1 - l)

        # rotate them according to heading
        A = rotate_point(x=A[0], y=A[1], cx=E1, cy=N1, angle=-head1)
        B = rotate_point(x=B[0], y=B[1], cx=E1, cy=N1, angle=-head1)
        C = rotate_point(x=C[0], y=C[1], cx=E1, cy=N1, angle=-head1)
        D = rotate_point(x=D[0], y=D[1], cx=E1, cy=N1, angle=-head1)

        # draw the polygon
        xs = [A[0], B[0], D[0], C[0], A[0]]
        ys = [A[1], B[1], D[1], C[1], A[1]]
        ax.plot(xs, ys, color=color, linewidth=2.0)
        return ax

    def fill_around_px(self, img:np.ndarray, x:int, y:int, meters_x:int, meters_y:int, value:float) -> np.ndarray:
        """Fills the pixels of an image at point (x, y) with 'value'. In x-direction, use 'meters_x', and y analogously.
        This considers unrotated objects, like the OS."""
        px_x = math.ceil((meters_x/2) / self.mpp)
        px_y = math.ceil((meters_y/2) / self.mpp)
        img[y-px_y : y+px_y+1, x-px_x : x+px_x+1] = value
        return img

    def fill_TS(self, img:np.ndarray, d:float, rel_bng:float, hdg:float, fill_value:float):
        """Draws a target ship into the image based on its Euclidean distance (d) and relative bearing to the OS (rel_bng), alongside the TS's heading (hdg)."""
        # compute TS position
        dx, dy = xy_from_polar(r=d, angle=rel_bng)
        x = OS_X + dx
        y = OS_Y - dy # y-axis for imgs is reversed

        # compute corner points
        c0_x = x - SHIP_B/2
        c0_y = y + SHIP_L/2

        c1_x = x + SHIP_B/2
        c1_y = y + SHIP_L/2

        c2_x = x + SHIP_B/2
        c2_y = y - SHIP_L/2

        c3_x = x - SHIP_B/2
        c3_y = y - SHIP_L/2

        # rotate them accordingly
        c_x, c_y = rotate_point(x=np.array([c0_x, c1_x, c2_x, c3_x, c0_x]),
                                y=np.array([c0_y, c1_y, c2_y, c3_y, c0_y]),
                                cx=x, cy=y, angle=hdg)
        # go back to pixel level
        c_x = (c_x/MPP).astype(np.int32)
        c_y = (c_y/MPP).astype(np.int32)
        
        # fill values
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        draw.polygon(xy=[e for e in zip(c_x, c_y)], fill=fill_value)
        return np.asarray(img)

    def _generate_depth_img(self, crit_value:float=None) -> np.ndarray:
        """Computes the depth map from the BEV, where the OS is located in the middle of the img.
        Args:
            crit_value(float): If given, returns a binary img indicating only whether depth is below
                               this value or not. If None, returns complete depth data img.
        Returns:
            np.ndarray([self.img_width, self.img_heigth])"""

        # construct regular grid
        N0, E0, head0 = self.OS.eta
        n_min, n_max = N0 - self.mpp * self.img_heigth/2, N0 + self.mpp * self.img_heigth/2
        e_min, e_max = E0 - self.mpp * self.img_width/2,  E0 + self.mpp * self.img_width/2

        n_arr = np.linspace(n_min, n_max, num=self.img_heigth)
        e_arr = np.linspace(e_min, e_max, num=self.img_width)

        # select relevant data
        i = np.logical_and(np.abs(self.data["depth_n"] - N0) <= (self.mpp * self.img_heigth * 0.75),
                            np.abs(self.data["depth_e"] - E0) <= (self.mpp * self.img_width * 0.75))  
        # Note: We take more data than needed here!
        
        n_unstruc = self.data["depth_n"][i]
        e_unstruc = self.data["depth_e"][i]
        d_unstruc = self.data["depth_d"][i]
        
        # rotate to have correct BEV (rotation before interpolation)
        e_unstruc_rot, n_unstruc_rot = rotate_point(x=e_unstruc, y=n_unstruc, cx=E0, cy=N0, angle=head0)
        n_grid, e_grid = np.meshgrid(n_arr, e_arr)

        # interpolate unstructured data
        img = griddata(points=(n_unstruc_rot, e_unstruc_rot), values=d_unstruc, xi=(n_grid, e_grid), 
                        method="linear", fill_value=0.0, rescale=False)

        # rotate it again
        img = np.rot90(img)
        
        if crit_value is None:
            return img
        else:
            return np.where(img >= crit_value, 1.0, 0.0)

    def render(self, data=None):
        """Renders the current environment. Note: The 'data' argument is needed since a recent update of the 'gym' package."""
        # check whether figure has been initialized
        if len(plt.get_fignums()) == 0:
            self.f, self.ax1 = plt.subplots(1, 1, figsize=(10, 10))
            #self.f2, self.ax2 = plt.subplots(1, 1, figsize=(10, 10))
            #self.f3, self.ax3 = plt.subplots(1, 1, figsize=(10, 10))

            plt.ion()
            plt.show()

        if self.step_cnt % 1 == 0:

            # ------------------------------ ship movement --------------------------------
            for ax in [self.ax1]:
                ax.clear()

                # general information
                #ax.text(0.125, 0.8675, self.__str__(OS_lat=OS_lat, OS_lon=OS_lon), fontsize=10, transform=plt.gcf().transFigure)
                ax.set_xlabel("East",  fontsize=10)
                ax.set_ylabel("North", fontsize=10)

                #--------------- depth plot ---------------------
                # tricontourf 
                N0, E0, head0 = self.OS.eta
                n_min, n_max = N0 - self.mpp * self.img_heigth/2, N0 + self.mpp * self.img_heigth/2
                e_min, e_max = E0 - self.mpp * self.img_width/2,  E0 + self.mpp * self.img_width/2

                i = np.logical_and(np.abs(self.data["depth_n"] - N0) <= (self.mpp * self.img_heigth * 0.75),
                                   np.abs(self.data["depth_e"] - E0) <= (self.mpp * self.img_width * 0.75))  
                
                ax.tricontourf(self.data["depth_e"][i], self.data["depth_n"][i], self.data["depth_d"][i])
                ax.scatter(self.data["axis_e"], self.data["axis_n"], color=self.default_cols[0], s=8)
                ax.set_xlim(e_min, e_max)
                ax.set_ylim(n_min, n_max)

                # imgshow
                #img = self._generate_depth_img(crit_value=self.OS.critical_depth)
                #self.ax2.imshow(img, vmin=0.0, vmax=1.0)
                #self.ax2.imshow(img, vmin=0.0, vmax=11.3746404)
                #self.ax2.set_title("Rotation before interpolation")

                #------------------ set ships ------------------------
                # OS
                ax = self._render_ship(ax=ax, vessel=self.OS, color=self.default_cols[0])

                # TSs
                if hasattr(self, "TSs"):
                    for i, TS in enumerate(self.TSs):
                        ax = self._render_ship(ax=ax, vessel=TS, color=self.default_cols[i+1])

                #--------------------- Current data ------------------------
                pass

            #plt.gca().set_aspect('equal')
            plt.pause(0.001)
