import os
import random
import time
import xml.etree.ElementTree as etxml
from copy import copy
from typing import List

import cv2
import gym
import numpy as np
import pybullet as pb
import pybullet_data
from crazyflies_bullet.src.crazyflies_bullet.utils import ED, get_assets_path
from gym import spaces
from pybullet_utils import bullet_client

from tud_rl.agents.base import BaseAgent
from tud_rl.envs._envs.VesselFnc import (angle_to_pi, bng_rel, cpa, dtr,
                                         xy_from_polar)


class Crazyfly:
    def __init__(self, 
                 bc:bullet_client.BulletClient, 
                 pos:np.ndarray, 
                 rpy:np.ndarray,
                 spd:float,
                 incident_dist:float=None,
                 file_name:str = "cf21x_bullet.urdf",
                 use_graphics:bool=True) -> None:
        self.bc  = bc

        self.pos = pos
        self.rpy = rpy
        self.spd = spd
        self.quat = pb.getQuaternionFromEuler(rpy)
        
        self.use_graphics = use_graphics

        if use_graphics:
            self.file_name = file_name
            self.file_name_path = os.path.join(get_assets_path(), file_name)
            self._load_assets()
            self._parse_robot_parameters()

            # GUI variables
            self.draw_n = 50
            self.quat0  = pb.getQuaternionFromEuler(np.array([0., 0., 0.]))
            self.axis_x = -1
            self.axis_y = -1
            self.axis_z = -1
            self.incident_dist = incident_dist
            
            self.old_col = [0., 0., 0., 1.] # black
            self._draw_circle()

        self.delta_yaw = dtr(5.0)

    def act(self, action:np.ndarray):
        """Drone moves always by self.spd [m/s * 1s] in the xy-plane. The 1D-array action contains the change in heading."""
        delta_yaw = np.clip(float(action), -1.0, 1.0) * self.delta_yaw

        # set new heading
        yaw = (self.rpy[2] + delta_yaw) % (2*np.pi)

        # compute new position
        x_add, y_add = xy_from_polar(r=self.spd, angle=yaw)
        x = self.pos[0] + x_add
        y = self.pos[1] + y_add
        z = self.pos[2]

        # set new position
        self.goTo(pos=np.array([x, y, z]), rpy=np.array([0., 0., yaw]))

    def _transformCoords(self, pos:np.ndarray, rpy:np.ndarray):
        """Transforms 2D-coordinates from a system with y pointing north, x point east, and angles are positiv
        from north clockwise to a new system with y pointing east, x pointing south, and angles are positiv from south
        anticlockwise."""
        x, y, z = pos[0:3]
        x_new = -y
        y_new = x
        yaw_new = (np.pi/2 - rpy[2]) % (2*np.pi)
        return np.array([x_new, y_new, z]), np.array([rpy[0], rpy[1], yaw_new])

    def goTo(self, pos:np.ndarray, rpy:np.ndarray):
        """Overwrites the drone's current xyz-position and rpy-angles."""
        # internal arguments
        self.pos = pos
        self.rpy = rpy

        # pybullet update
        if self.use_graphics:
            pos_t, rpy_t = self._transformCoords(pos=self.pos, rpy=self.rpy)
            quat_t = pb.getQuaternionFromEuler(rpy_t)
            self.bc.resetBasePositionAndOrientation(
                self.body_unique_id,
                posObj=pos_t,
                ornObj=quat_t
            )
            self._draw_circle()

    def _draw_circle(self):
        ref = np.zeros((self.draw_n, 3))
        thetas = np.linspace(start=0., stop=2*np.pi, num=self.draw_n)

        pos_t, _ = self._transformCoords(pos=self.pos, rpy=self.rpy)
        ref[:, 0] = pos_t[0] + self.incident_dist * np.cos(thetas) # x
        ref[:, 1] = pos_t[1] + self.incident_dist * np.sin(thetas) # y
        ref[:, 2] = 1. # z

        # color depends on task
        if not hasattr(self, "fly_to_goal"):
            col = [0., 0., 0., 1.]  # black
        else:
            if self.fly_to_goal == 1.0:
                col = [0., 1., 0., 1.] # green
            else:
                col = [0., 0., 0., 1.] # black

        # move circle and change color
        if hasattr(self, "circ_ids"):
            for k, id in enumerate(self.circ_ids):
                pb.resetBasePositionAndOrientation(id, ref[k], self.quat0)
                if col != self.old_col:
                    pb.changeVisualShape(id, -1, rgbaColor=col)
        else:
            self.circ_ids = []
            for k in range(self.draw_n):
                id = self.bc.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=self.bc.createVisualShape(
                        self.bc.GEOM_SPHERE,
                        radius=0.01,
                        rgbaColor=col, # black
                    ),
                    basePosition=ref[k]
                )
                self.circ_ids.append(id)
        self.old_col = col

    def show_local_frame(self):
        if self.use_graphics:
            AXIS_LENGTH = 0.25
            self.axis_x = self.bc.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[AXIS_LENGTH, 0, 0],
                lineColorRGB=[1, 0, 0], # red
                parentObjectUniqueId=self.body_unique_id,
                parentLinkIndex=-1,
                replaceItemUniqueId=self.axis_x
                )
            self.axis_y = self.bc.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, AXIS_LENGTH, 0],
                lineColorRGB=[0, 1, 0], # green
                parentObjectUniqueId=self.body_unique_id,
                parentLinkIndex=-1,
                replaceItemUniqueId=self.axis_y,
                )
            self.axis_z = self.bc.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, 0, AXIS_LENGTH],
                lineColorRGB=[0, 0, 1], # blue
                parentObjectUniqueId=self.body_unique_id,
                parentLinkIndex=-1,
                replaceItemUniqueId=self.axis_z
                )

    def _load_assets(self) -> int:
        """Loads the robot description file (URDF) into the simulation."""
        assert self.file_name_path.endswith('.urdf')
        assert os.path.exists(self.file_name_path), \
            f'Did not find {self.file_name} at: {get_assets_path()}'

        self.body_unique_id = self.bc.loadURDF(
            self.file_name_path,
            self.pos,
            pb.getQuaternionFromEuler(self.rpy),
            # Important Note: take inertia from URDF...
            flags=pb.URDF_USE_INERTIA_FROM_FILE
        )

    def _parse_robot_parameters(self
                                ) -> None:
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.
        """
        URDF_TREE = etxml.parse(self.file_name_path).getroot()
        self.M = float(URDF_TREE[1][0][1].attrib['value'])
        self.L = float(URDF_TREE[0].attrib['arm'])
        self.THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        # self.THRUST2WEIGHT_RATIO = 2.5
        self.IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        self.IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        self.IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        self.J = np.diag([self.IXX, self.IYY, self.IZZ])
        self.J_INV = np.linalg.inv(self.J)
        self.KF = float(URDF_TREE[0].attrib['kf'])
        self.KM = float(URDF_TREE[0].attrib['km'])
        self.COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        self.COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        self.COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        self.COLLISION_Z_OFFSET = self.COLLISION_SHAPE_OFFSETS[2]
        self.MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        self.GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        self.PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        self.DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])  # [kg /rad]
        self.DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])  # [kg /rad]
        self.DRAG_COEFF = np.array([self.DRAG_COEFF_XY, self.DRAG_COEFF_XY, self.DRAG_COEFF_Z])  # [kg /rad]
        self.DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        self.DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        self.DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])


class Destination:
    def __init__(self, bc:bullet_client.BulletClient, use_graphics:bool) -> None:
        # params
        self.bc = bc
        self.use_graphics = use_graphics
        
        # size
        self.radius          = 0.3 # [m]
        self.restricted_area = 0.3 # [m]
        self.spawn_radius    = 2.0 # [m]
        self.respawn_radius  = 2.4 # [m]

        # position
        self.pos = np.array([0., 0., 1.])

        # drawing
        if use_graphics:
            self.n1 = 50
            self.n2 = 200
            self.ref = np.zeros((self.n1, 3))
            thetas = np.linspace(start=0., stop=2*np.pi, num=self.n1)

            self.ref[:, 0] = self.radius * np.cos(thetas) # x
            self.ref[:, 1] = self.radius * np.sin(thetas) # y
            self.ref[:, 2] = 1. # z

            self.ref2 = np.zeros((self.n2, 3))
            thetas = np.linspace(start=0., stop=2*np.pi, num=self.n2)

            self.ref2[:, 0] = self.spawn_radius * np.cos(thetas) # x
            self.ref2[:, 1] = self.spawn_radius * np.sin(thetas) # y
            self.ref2[:, 2] = 1. # z
            self._inital_draw()

        # timing
        self._t_close = 60       # cnt, time the destination is closed after an aircraft has entered 
        self._t_nxt_open = 0     # cnt, current time until the destination opens again
        self._t_open_since = 0   # cnt, current time since the vertiport is open
        self._was_open = True
        self.open()

    def _inital_draw(self):
        self.obj_ids = []
        for k in range(self.n1):
            obj_id = self.bc.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=self.bc.createVisualShape(
                    self.bc.GEOM_SPHERE,
                    radius=0.01,
                    rgbaColor=[0., 1., 0., 1.], # green
                ),
                basePosition=self.ref[k]
            )
            self.obj_ids.append(obj_id)

        for k in range(self.n2):
            self.bc.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=self.bc.createVisualShape(
                    self.bc.GEOM_SPHERE,
                    radius=0.01,
                    rgbaColor=[0., 0., 0., 1.], # black
                ),
                basePosition=self.ref2[k]
            )

    def _draw(self):
        if self.color == "green":
            col = [0., 1., 0., 1.]
        elif self.color == "red":
            col = [1., 0., 0., 1.]

        for k in range(self.n1):
            pb.changeVisualShape(self.obj_ids[k], -1, rgbaColor=col)

    def reset(self):
        self.open()

    def step(self, drones: List[Crazyfly]):
        """Updates status of the destination.
        Returns:
            np.ndarray([number_of_planes,]): who entered a closed destination
            np.ndarray([number_of_planes,]): who entered an open destination
            bool: whether destination just openeded again"""
        just_opened = False

        # count time until next opening
        if self._is_open is False:
            self._t_nxt_open -= 1
            if self._t_nxt_open <= 0:
                self.open()
                just_opened = True
        else:
            self._t_open_since += 1

        # store opening status
        self._was_open = copy(self._is_open)

        # check who entered a closed or open destination
        entered_close = np.zeros(len(drones), dtype=bool)
        entered_open  = np.zeros(len(drones), dtype=bool)

        for i, p in enumerate(drones):
            if p.D_dest <= self.radius:            
                if self._is_open:
                    entered_open[i] = True
                else:
                    entered_close[i] = True

        #  close only if the correct AC entered
        for i, p in enumerate(drones):
            if entered_open[i] and p.fly_to_goal == 1.0:
                self.close()
        return entered_close, entered_open, just_opened

    def open(self):
        self._t_open_since = 0
        self._t_nxt_open = 0
        self._is_open = True
        self.color = "green"
        self._draw()
    
    def close(self):
        self._t_open_since = 0
        self._is_open = False
        self._t_nxt_open = copy(self._t_close)
        self.color = "red"
        self._draw()

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


class CrazyflySimple(gym.Env):
    """Environment to simulate a Crazyfly 2.1 of Bitcraze. 
    This environment is highly simplified and does not consider the dynamics."""
    def __init__(self, 
                 w_coll:float,
                 w_goal:float,
                 w_comf:float,
                 r_goal_norm:float=0.2,
                 N_drones:int=2, 
                 use_graphics:bool=False):
        super(CrazyflySimple, self).__init__()

        # params
        self.N_drones = N_drones
        self.N_drones_max = N_drones
        self.use_graphics = use_graphics
        self.w_coll = w_coll
        self.w_goal = w_goal
        self.w_comf = w_comf
        self.r_goal_norm = r_goal_norm
        self.w = self.w_coll + self.w_goal + self.w_comf

        self.validate = True
        self.video = False

        # initialize and setup PyBullet
        if use_graphics:
            self.bc = self._setup_client_and_physics()
        else:
            self.bc = None

        # speed limits
        self.base_spd  = 0.025
        self.delta_spd = 0.005

        # accident definitions
        self.accident_dist = 0.1 # m
        self.incident_dist = 0.2 # m

        # setup drones
        self.drones = [Crazyfly(
            bc=self.bc, 
            pos=np.array([0., 0., 1.]), 
            rpy=np.zeros(3), 
            spd = self.base_spd,
            incident_dist=self.incident_dist,
            use_graphics=use_graphics) 
                       for _ in range(N_drones)]

        # setup destination
        self.dest = Destination(bc=self.bc, use_graphics=use_graphics)

        # config
        self.history_length = 2
        self.OS_obs     = 3     # abs bng goal, rel bng goal, dist goal, fly to goal
        self.obs_per_TS = 6     # distance, relative bearing, speed difference, heading intersection, DCPA, TCPA
        self.obs_size   = self.OS_obs + self.obs_per_TS*(self.N_drones-1)

        self.observation_space = spaces.Box(low  = np.full(self.obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(self.obs_size,  np.inf, dtype=np.float32))
        self.act_size = 1
        self.action_space = spaces.Box(low  = np.full(self.act_size, -1.0, dtype=np.float32), 
                                       high = np.full(self.act_size, +1.0, dtype=np.float32))
        if self.validate:
            self._max_episode_steps = 10_000
        else:
            self._max_episode_steps = 250

    def _setup_client_and_physics(self) -> bullet_client.BulletClient:
        # setup client
        bc = bullet_client.BulletClient(connection_mode=pb.GUI)

        # add open_safety_gym/envs/data to the PyBullet data path
        bc.setAdditionalSearchPath(get_assets_path())

        # disable GUI debug visuals
        bc.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        bc.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
        bc.setGravity(0, 0, -9.81)

        # also add PyBullet's data path
        bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.PLANE_ID = bc.loadURDF("plane.urdf")
        
        # load 10x10 Walls
        pb.loadURDF(os.path.join(get_assets_path(), "room_10x10.urdf"), useFixedBase=True)

        # set camera position
        bc.resetDebugVisualizerCamera(
            cameraTargetPosition=(0.0, 0.0, 0.0),
            cameraDistance=3.5,
            cameraYaw=90,
            cameraPitch=-70
        )
        return bc

    def reset_drone(self, i:int, drone:Crazyfly) -> Crazyfly:
        # sample bearing and speed
        if self.validate:
            qdr = dtr([0., 72., 144., 216., 288.][i])
            spd = self.base_spd
        else:
            qdr = np.random.uniform(0.0, 2*np.pi)
            spd = np.random.uniform(self.base_spd-self.delta_spd, self.base_spd+self.delta_spd) # m/s

        # determine origin
        x, y = xy_from_polar(r=self.dest.spawn_radius, angle=qdr)

        # add noise to yaw
        yaw = (qdr + np.pi) % (2*np.pi)

        if not self.validate:
            sgn = 1 if bool(random.getrandbits(1)) else -1
            yaw = (yaw + sgn * dtr(np.random.uniform(20.0, 45.0))) % (2*np.pi)

        # reset speed and fly_to_goal, and move drone
        drone.spd = spd
        drone.fly_to_goal = -1.
        drone.goTo(pos=np.array([self.dest.pos[0] + x, self.dest.pos[1] + y, 1.]), 
                   rpy=np.array([0., 0., yaw]))
        
        # compute initial distance to destination
        drone.D_dest     = ED(pos1=drone.pos, pos2=self.dest.pos)
        drone.D_dest_old = copy(drone.D_dest)
        return drone

    def reset(self):
        """Resets environment to initial state."""
        # counter
        self.step_cnt = 0

        # disable rendering before resetting
        if self.use_graphics:
            self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 0)

        # reset drone state
        for i, drone in enumerate(self.drones):
            self.drones[i] = self.reset_drone(i, drone)

        # reset destination
        self.dest.reset()

        # interface to high-level module including goal decision
        self._high_level_control()

        # enable rendering again after resetting
        if self.use_graphics:  
            self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 1)

            if self.video:
                self._capture_frame()

        # init state
        self._set_state()
        self.state_init = self.state
        return self.state

    def _capture_frame(self):
        # Capture the current frame as an image
        img = pb.getCameraImage(width=1920, height=1080)[2]

        # Save the frame as an image file
        cv2.imwrite(f"frames/frame_{self.step_cnt}.png", img)

    def _high_level_control(self):
        """Decides who out of the current flight taxis should fly toward the goal."""
        if self.validate:
            if len(self.drones) > 0:

                # check whether one guy has already a go-signal
                if all([d.fly_to_goal == -1.0 for d in self.drones]):

                    idx = np.argmin([d.D_dest for d in self.drones])
                    for i, _ in enumerate(self.drones):
                        if i == idx:
                            self.drones[i].fly_to_goal = 1.0
                        else:
                            self.drones[i].fly_to_goal = -1.0
        else:
            for i, _ in enumerate(self.drones):
                if i == 0 and self.step_cnt >= 200:
                    self.drones[i].fly_to_goal = 1.0
                else:
                    self.drones[i].fly_to_goal = -1.0

    def _handle_respawn(self, entered_open:np.ndarray):
        """Respawns planes when they correctly entered the open destination area or left the whole map."""
        # check map-leaving and successful vertiport entry
        if self.validate:
            for i, d in enumerate(self.drones):
                if (entered_open[i] and d.fly_to_goal == 1.0) or d.D_dest >= self.dest.respawn_radius:
                    self.drones.pop(i)
                    self.N_drones = len(self.drones)

        # check only map-leaving
        else:
            for i, drone in enumerate(self.drones):
                if drone.D_dest >= self.dest.respawn_radius:
                    self.drones[i] = self.reset_drone(drone)

    def _set_state(self):
        if len(self.drones) == 0:
            self.state = None
            return

        # state observation of id0 will be used for learning
        self.state = self._get_state(0)

        for i, d in enumerate(self.drones):

            # compute current state
            if i == 0:
                d.s = self.state
            else:
                d.s = self._get_state(i)

            # update history
            if not hasattr(d, "s_hist"):
                d.s_hist = np.zeros((self.history_length, self.obs_size))
                d.hist_len = 0
            else:
                if d.hist_len == self.history_length:
                    d.s_hist = np.roll(d.s_hist, shift=-1, axis=0)
                    d.s_hist[self.history_length - 1] = d.s_old
                else:
                    d.s_hist[d.hist_len] = d.s_old
                    d.hist_len += 1
            
            # safe old state
            d.s_old = copy(d.s)

    def _get_state(self, i:int) -> np.ndarray:
        """Computes the state from the perspective of the i-th agent of the internal plane array."""

        # select plane of interest
        d = self.drones[i]

        # relative bearing to goal, distance, fly to goal
        rel_bng_goal = bng_rel(N0=d.pos[1], E0=d.pos[0], N1=self.dest.pos[1], E1=self.dest.pos[0], head0=d.rpy[2], to_2pi=False)/np.pi
        d_goal   = ED(pos1=d.pos, pos2=self.dest.pos)/self.dest.spawn_radius
        task     = d.fly_to_goal
        s_i = np.array([rel_bng_goal, d_goal, task])

        # information about other planes
        TS_info = []
        for j, other in enumerate(self.drones):
            if i != j:
                # relative speed
                v_r = (other.spd - d.spd)/(2*self.delta_spd)

                # relative bearing
                bng = bng_rel(N0=d.pos[1], E0=d.pos[0], N1=other.pos[1], E1=other.pos[0], head0=d.rpy[2], to_2pi=False)/np.pi

                # distance
                dist = ED(pos1=d.pos, pos2=other.pos)/self.dest.spawn_radius

                # heading intersection
                C_T = angle_to_pi(other.rpy[2] - d.rpy[2])/np.pi

                # CPA metrics
                DCPA, TCPA = cpa(NOS=d.pos[1], EOS=d.pos[0], NTS=other.pos[1], ETS=other.pos[0], chiOS=d.rpy[2], chiTS=other.rpy[2],
                                 VOS=d.spd, VTS=other.spd)
                DCPA = DCPA / 0.2
                TCPA = TCPA / 60.0

                # aggregate
                TS_info.append([dist, bng, v_r, C_T, DCPA, TCPA])

        # no TS is in sight: pad a 'ghost ship' to avoid confusion for the agent
        if len(TS_info) == 0:
            TS_info.append([1.0, -1.0, -1.0, -1.0, 1.0, -1.0])

        # sort array according to distance
        TS_info = np.hstack(sorted(TS_info, key=lambda x: x[0], reverse=True)).astype(np.float32)

        # pad NA's as usual in single-agent LSTMRecTD3
        desired_length = self.obs_per_TS * (self.N_drones_max-1)
        TS_info = np.pad(TS_info, (0, desired_length - len(TS_info)), "constant", constant_values=np.nan).astype(np.float32)

        s_i = np.concatenate((s_i, TS_info))
        return s_i

    def step(
            self,
            a:BaseAgent
    ) -> tuple:
        # counter
        self.step_cnt += 1

        # move drone
        for i, d in enumerate(self.drones):

            # spatial-temporal recurrent
            act = a.select_action(s        = d.s, 
                                  s_hist   = d.s_hist, 
                                  a_hist   = None, 
                                  hist_len = d.hist_len)

            # move plane
            d.act(act)

            # save action of id0 for comfort reward
            if i == 0:
                a0 = act[0]

        # update distances to destination
        for drone in self.drones:
            drone.D_dest_old = copy(drone.D_dest)
            drone.D_dest = ED(pos1=drone.pos, pos2=self.dest.pos)

        # check destination entries
        _, entered_open, just_opened = self.dest.step(self.drones)

        # get reward
        self.compute_reward(a0)

        # respawning
        self._handle_respawn(entered_open)

        # high-level control
        if self.validate:
            if just_opened:
                self._high_level_control()
        else:
            self._high_level_control()

        # video
        if self.video:
            self._capture_frame()

        # set new state, compute reward and done signal
        self._set_state()
        done = self.done()
        return self.state, float(self.r[0]), done, {}
    
    def compute_reward(self, a0:float):
        r_coll = np.zeros((self.N_drones, 1), dtype=np.float32)
        r_goal = np.zeros((self.N_drones, 1), dtype=np.float32)
        r_comf = np.zeros((self.N_drones, 1), dtype=np.float32)

        # ------ collision reward ------
        D_matrix = np.ones((len(self.drones), len(self.drones))) * np.inf
        for i, di in enumerate(self.drones):
            for j, dj in enumerate(self.drones):
                if i != j and i == 0:
                    D_matrix[i][j] = ED(pos1=di.pos, pos2=dj.pos)

        for i, di in enumerate(self.drones):
            if i != 0:
                continue

            D = float(np.min(D_matrix[i]))

            if D <= self.accident_dist:
                r_coll[i] -= 10.0

            elif D <= self.incident_dist:
                r_coll[i] -= 10.0

            else:
                r_coll[i] -= 5*np.exp(-(D-self.incident_dist)**2/(0.1203412)**2) 
                # approximately yields reward of -5 at 0.2 and -0.01 at 0.5m
                # b = function(d, r){
                # sqrt((d-0.2)^2/(-log(r/5)))
                # }

            # off-map
            if di.D_dest > self.dest.spawn_radius: 
                r_coll[i] -= 5.0

        # ------ goal reward ------
        for i, d in enumerate(self.drones):

            if i != 0:
                continue
            
            # goal-approach reward for the one who should fly toward the goal
            if d.fly_to_goal == 1.0:
                r_goal[i] = (d.D_dest_old - d.D_dest)/self.r_goal_norm
            
            # punish others for getting into the restricted area
            elif d.D_dest <= self.dest.restricted_area:
                r_goal[i] = -5.0

        #--------------- comfort reward --------------------
        r_comf[0] = -(a0)**4

        # aggregate reward components
        if self.w == 0.0:
            r = np.zeros((self.N_drones, 1), dtype=np.float32)
        else:
            r = (self.w_coll*r_coll + self.w_goal*r_goal + self.w_comf*r_comf)/self.w

        # store
        self.r = r
        self.r_coll = r_coll
        self.r_goal = r_goal
        self.r_comf = r_comf

    def done(self):
        # all planes left
        if len(self.drones) == 0:
            return True
        
        # id-0 successfully reached the goal
        #elif self.drones[0].D_dest <= self.dest.radius and self.drones[0].fly_to_goal == 1:
        #    return True

        # artificial done signal
        elif self.step_cnt >= self._max_episode_steps:
            return True
        return False

    def render(self, mode='human') -> np.ndarray:
        if self.use_graphics:
            time.sleep(0.05)
            for drone in self.drones:
                drone.show_local_frame()
