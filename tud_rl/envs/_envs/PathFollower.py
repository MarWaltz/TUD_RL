import copy
import random
import gym
import csv
import math
import matplotlib
import numpy as np
import mmgdynamics as mmg
import matplotlib.pyplot as plt

from gym import spaces

from matplotlib.figure import Figure
from matplotlib import cm, transforms
from matplotlib.patches import Rectangle, Patch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from mmgdynamics.calibrated_vessels import kvlcc2, seiunmaru
from mmgdynamics.dynamics import calibrate

from getpass import getuser
from collections import deque
from typing import Tuple, Dict, Any, List

# --------- Global config -------------------

TRAIN_ON_TAURUS: bool = False
CNN: bool = False

if TRAIN_ON_TAURUS:
    RIVERDATA = "/home/s2075466/riverdata"
    matplotlib.use("Agg")
else:
    RIVERDATA = f"/home/{getuser()}/Dropbox/TU Dresden/riverdata"
    matplotlib.use("TKAgg")

# --------- Gym Env -------------------------


class PathFollower(gym.Env):
    def __init__(self, mode: str = "step", epi_steps: int = 2000) -> None:
        super().__init__()

        assert mode in ["step", "abs", "cont"]
        self.mode = mode

        # Number of evenly spaced gridpoints for which stream velocity and water depth data is available.
        # All Gridpoints are BASEPOINT_DIST meters apart to span a maximum rhine width of 500 meters
        self.GRIDPOINTS: int = 26
        self.BASEPOINT_DIST: int = 20

        # Convergance rate of vector field for vector field guidance
        self.K: float = 0.005

        # Derivative pentalty constant
        self.C: float = -1

        # Constant for tanh of cte
        self.T = 0.01

        # Rudder increment/decrement per action [deg]
        self.RUDDER_INCR: int = 5
        self.MAX_RUDDER: int = 20

        # Minimum water under keel [m]
        self.MIN_UNDER_KEEL: float = 1.5

        # Current timestep of episode
        self.timestep: int = 1

        # Prepare the river data by importing it from file
        self.coords, self.metrics = import_river(RIVERDATA)

        # River X coordinate
        self.rx: np.ndarray = np.array([self.coords[row][0]
                                        for row, _ in enumerate(self.coords)])
        self.rx = self.rx.reshape((-1, self.GRIDPOINTS))

        # River Y coordinate
        self.ry: np.ndarray = np.array([self.coords[row][1]
                                        for row, _ in enumerate(self.coords)])
        self.ry = self.ry.reshape((-1, self.GRIDPOINTS))

        # Extract water depth and reshape
        self.wd: np.ndarray = np.array([math.sqrt(self.metrics[row][3])
                                        for row, _ in enumerate(self.metrics)])
        self.wd = self.wd + 5  # Lower the rhine by 2 meters to prevent the dynamics to crash
        self.wd = self.wd.reshape((-1, self.GRIDPOINTS))
        self.max_wd = np.max(self.wd)

        # Stream direction
        self.str_diry: np.ndarray = np.array(
            [self.metrics[row][1] for row, _ in enumerate(self.metrics)])
        self.str_diry = self.str_diry.reshape((-1, self.GRIDPOINTS))

        self.str_dirx: np.ndarray = np.array(
            [self.metrics[row][2] for row, _ in enumerate(self.metrics)])
        self.str_dirx = self.str_dirx.reshape((-1, self.GRIDPOINTS))

        # Extract stream velocity and reshape
        self.str_vel: np.ndarray = np.array([self.metrics[row][6]
                                             for row, _ in enumerate(self.metrics)])
        self.str_vel = self.str_vel.reshape((-1, self.GRIDPOINTS))
        #self.str_vel = sigmoid(self.str_vel)
        # self.str_vel = np.clip(self.str_vel,0,1)

        # Index list of the path to follow
        self.path_index: np.ndarray = np.empty(self.wd.shape[0], dtype=int)

        # Path roughness (Use only every nth data point of the path)
        self.ROUGHNESS: int = 10

        # The path_index variable gets filled here.
        self.path: Dict[str, List[float]] = self.get_river_path()
        self.red_path: Dict[str, List[float]] = self.smooth(copy.deepcopy(self.path),
                                                            self.ROUGHNESS, alpha=0.08)

        # Plotting offsets for CNN image generation
        self.lookahead = 70
        self.lookbehind = 70

        # Vessel set-up ------------------------------------------------

        # Rudder angle [rad]
        self.delta = 0.0

        # Propeller revolutions [s⁻¹]
        self.nps = 7.0

        # Overall vessel speed [m/s]
        self.speed = 0

        # Vessel to be simulated
        # seiunmaru  # calibrate(GMS,1000)
        self.vessel: Dict[str, float] = kvlcc2

        # Movement heading can include drift angles
        self.movement_heading = 0.0

        # Heading the longitudinal vessel axis points to
        self.aghead = 0.0

        # Get the border to port and starboard where
        # the water depth equals the ship draft
        self.build_fairway_border()

        # Initial values
        self.ivs: np.ndarray = self.reset_ivs()

        self.history = History()

        # Gym Action and observation space -------------------------------------
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(43207 if CNN else 49,)
        )

        # Set action space according to mode
        if self.mode == "step":
            self.n_actions = 3
            self.action_space = spaces.Discrete(self.n_actions)
        elif self.mode == "abs":
            self.n_actions = 9
            self.action_space = spaces.Discrete(self.n_actions)
        elif self.mode == "cont":
            self.n_actions = 1
            self.action_space = spaces.Box(
                low=-1,
                high=1,
                shape=(1,)
            )
        self.max_episode_steps = epi_steps

    def reset(self) -> np.ndarray:

        def start_x(x): return self.path["x"][x]
        def start_y(y): return self.path["y"][y]

        self.DIR = random.choice([1, -1])
        self.DIR = -1

        # Index of the current waypoint that is in use. Starts with its starting value
        # self.waypoint_idx = self.STARTIND
        mid_path_index = len(self.path["idx"])//2
        if self.DIR == 1:
            self.waypoint_idx = random.randrange(
                10,
                mid_path_index)
        elif self.DIR == -1:
            self.waypoint_idx = random.randrange(
                mid_path_index,
                len(self.path["idx"]) - 100)

        # Get the index of waypoint behind agent
        self.lwp_idx = self.get_lwp_idx(self.waypoint_idx)

        # Last waypoint and next waypoint
        self.lwp = self.get_wp(self.lwp_idx)
        self.nwp = self.get_wp(self.lwp_idx, 1)

        # Agent position plus some random noise
        self.agpos = (
            (start_x(self.waypoint_idx) + random.uniform(-10, 10)),
            (start_y(self.waypoint_idx) + random.uniform(-10, 10))
        )

        # Agent position last timestep
        self.agpos_lt = self.agpos

        # Set agent heading to path heading plus some small noise (5° = 1/36*pi rad)
        random_angle = 1/36*math.pi
        self.aghead = self.path_angle(
            p1=(start_x(self.waypoint_idx),
                start_y(self.waypoint_idx)),
            p2=(start_x(self.waypoint_idx+1),
                start_y(self.waypoint_idx+1))) + random.uniform(
                    -random_angle, random_angle)

        self.movement_heading = self.aghead

        # Construct frames of the input space
        # in order to build image vectors out of them
        self.construct_frames()

        # Create the vessel obj and place its center at the agent position
        if not TRAIN_ON_TAURUS:
            self.init_vessel()

        # Get current water depth and stream velocity
        # for the current postition
        # Water depth in [m]
        # Stream velocity in [m/s]
        self.curr_wd, self.curr_str_vel, self.curr_str_dir = self.get_river_metrics()

        # Cross track error [m]
        self.cte = self.cross_track_error()

        # Desired course [rad]
        self.dc = self.desired_course(self.cte, self.K)

        # Heading error [rad]
        self.course_error = self.heading_error(self.dc)

        self.ivs = self.reset_ivs()

        self.state = self.build_state()

        return self.state

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:

        action = float(action) if self.mode == "cont" else int(action)

        # Reset the done flag on every step
        self.done: bool = False
        if self.timestep >= self.max_episode_steps:
            self.done = True

        if self.timestep != 1 and self.timestep % 5 != 0:
            action = self.history.action[-1]

        # Store old delta
        delta_old = self.delta

        self.map_action(action)

        # Update the initial values for the rudder angle
        self.ivs[3] = self.delta

        # Simulate the vessel with the MMG model for one second (default)
        try:
            sol = mmg.step(
                X=self.ivs,
                params=self.vessel,
                nps_old=self.nps,
                delta_old=delta_old,
                fl_psi=self.rel_current_attack_angle(),
                water_depth=self.curr_wd,
                # water_depth=None,
                # fl_vel=None,
                fl_vel=self.get_str_vel(),
                atol=1e-5,
                rtol=1e-5,
                sps=1
            )

            # Unpack values of solver
            u, v, r, *_ = sol

        except ValueError:
            sol = self.ivs
            self.done = True
            u, v, r = 0, 0, 0

        for val, name in zip([u, v, r, self.ivs[:-1], self.delta, action, self.timestep],
                             ["u", "v", "r", "ivs", "delta", "action", "timestep"]):
            self.history.append(val, name)

        # Update Speed
        self.speed = math.sqrt(u**2+v**2)

        # Update state
        self.update_vessel_heading(r)
        self.update_movement_heading(u, v)
        self.update_position(u, v)

        # Check if agent crossed a waypoint
        if self.crossed_waypoint():
            self.update_waypoints()
            self.construct_frames()

        # Get new river metrics
        self.curr_wd, self.curr_str_vel, self.curr_str_dir = self.get_river_metrics()

        # Cross track error [m]
        self.cte = self.cross_track_error()

        # Desired course [rad]
        self.dc = self.desired_course(self.cte, self.K)

        # Heading error [rad]
        self.course_error = self.heading_error(self.dc)

        reward = self.calculate_reward()
        # if self.curr_wd<=2.5 and self.curr_str_vel<=0.3:
        #    self.done = True

        # Set initial values to the result of the last iteration
        self.ivs = np.hstack(sol)

        # Rebuild state
        self.state = self.build_state()

        if self.done:
            self.timestep = 1
        else:
            self.timestep += 1

        # print(round(self.curr_str_dir,2))
        return self.state, reward, self.done, {}

    def map_action(self, action: int) -> None:
        """Maps an integer action to a change in rudder angle
        depending on the mode of training

        Args:
            action (int): Integer value representing an action
        """

        def rad(a): return a/180*math.pi

        if self.mode == "step":
            # Clip rudder angle to plus-minus 35°
            if not self.delta <= -self.MAX_RUDDER or self.delta >= self.MAX_RUDDER:
                if action == 0:  # Steer to port
                    # Unneccessary but left in for clarity
                    self.delta += float(0)
                elif action == 1:
                    self.delta -= rad(float(self.RUDDER_INCR))
                elif action == 2:  # Steer to starboard
                    self.delta += rad(float(self.RUDDER_INCR))

        elif self.mode == "abs":
            action -= int(self.n_actions/2)
            self.delta = rad(float(action*self.RUDDER_INCR))

        elif self.mode == "cont":
            rud = action * self.RUDDER_INCR
            self.delta += rad(rud)

    def calculate_reward(self) -> float:

        draft = self.vessel["d"]
        wd = self.curr_wd

        if wd - draft < self.MIN_UNDER_KEEL:
            self.done = True
            border = -20
        else:
            border = 0

        # Derivative penalty
        if self.mode != "step":
            try:
                old_action = self.history.action[-2]
                action = self.history.action[-1]
                deriv = abs(action - old_action)
                if self.mode == "cont":
                    span = abs(
                        self.action_space.high[0]-self.action_space.low[0])
                    deriv_rew = ((deriv**2)/span) * self.C
                else:
                    # deriv_rew = (deriv/(self.n_actions-1))
                    deriv_rew = (deriv/(self.n_actions-1))**2 * self.C
                    # deriv_rew = inv_logit(deriv_rew,coef = 2) * self.C
            except IndexError:
                deriv_rew = 0
        else:
            deriv_rew = 0

        k_rot = 100
        r_rot = math.exp(-k_rot * abs(self.history.r[-1]))
        print(np.round(max(self.history.r), 4))

        # Hyperparameter controling the steepness
        # of the exponentiated cross track error
        k_cte = 0.1
        r_cte = math.exp(-k_cte * abs(self.cte))

        # Reward for heading angle
        k_head = 10
        r_ang = math.exp(-k_head * abs(self.course_error))

        return r_cte + r_ang + border  # + r_rot

    def calculate_reward2(self) -> None:

        draft = self.vessel["d"]
        wd = self.curr_wd

        # if wd - draft < self.MIN_UNDER_KEEL:
        #    coll = -10
        # else: coll = 0

        k = 10
        r_cog = 1/(1+k*abs(self.course_error/0.15*math.pi))
        r_dist = 1/(1+k*abs(self.cte/100))
        r_rot = 1/(1+k*abs(self.history.r[-1]/0.015*math.pi))

        return r_cog*r_dist*r_rot  # + coll

    def get_river_path(self) -> dict:
        """Generate the path for the vessel to follow.
        For now this is (approx) the deepest point of the fairway
        for the entire river part

        In this case I will scan through the middle third of the rivers
        crossection and find the deepest point in that corridor.
        This point then gets mapped to its corresponding
        x and y coordinates, which then gets returned

        Args:
            roughness (int): Roughness of the path (return only every nth
            data point of the generated path)

        Returns:
            dict: Dict with x and y coordinates of the path
        """

        # Dictionary to hold the path
        path = {
            "x": np.empty(self.wd.shape[0]),
            "y": np.empty(self.wd.shape[0]),
            "idx": np.arange(self.wd.shape[0])
        }

        # To use only the middle portion of the river
        offset = self.GRIDPOINTS // 3

        # print("Building waypoints...")
        for col in range(self.wd.shape[0]):
            # Get middle thrid per column
            frame = self.wd[col][offset:2*offset+1]
            max_ind = np.argmax(frame)  # Find deepest index
            max_ind += offset  # Add back offset

            # Average to river midpoint to keep the path
            # more centered on the river
            max_ind = (max_ind + self.GRIDPOINTS//2)//2

            # Add the index to the path index list
            self.path_index[col] = int(max_ind)

            # Get the coordinates of the path by returning the x and y coord
            # for the calculated max index per column
            path["x"][col] = self.rx[col][max_ind]
            path["y"][col] = self.ry[col][max_ind]

        # Exponential smoothing for coordinates
        return path

    def build_fairway_border(self) -> None:
        """Build a border to port and starboard of the vessel
        by checking if the water is deep enough to drive.

        This function runs once during class initialization
        and scans through the entire water depth array 'self.wd'
        to find all points for which the water depth is lower than
        the minimum allowed water under keel.
        """

        draft = self.vessel["d"]

        free_space: np.ndarray[bool] = self.wd - \
            draft < self.MIN_UNDER_KEEL

        lower_poly = np.full(free_space.shape[0], int(0))
        upper_poly = np.full(free_space.shape[0], self.GRIDPOINTS - 1)

        max_index = self.GRIDPOINTS - 1
        for i in range(free_space.shape[0]):
            mid_index: int = self.path_index[i]
            searching_upper: bool = True
            searching_lower: bool = True
            for lo in range(mid_index):
                if free_space[i][mid_index - lo] and searching_lower:
                    lower_poly[i] = mid_index - lo + 1
                    searching_lower = False
            for high in range(max_index - mid_index):
                if free_space[i][mid_index + high] and searching_upper:
                    upper_poly[i] = mid_index + high
                    searching_upper = False

        self.port_border: dict = {
            "x": np.array([self.rx[i][upper_poly[i]]
                           for i in range(self.wd.shape[0])]),
            "y": np.array([self.ry[i][upper_poly[i]]
                           for i in range(self.wd.shape[0])])
        }

        self.star_border: dict = {
            "x": np.array([self.rx[i][lower_poly[i]]
                           for i in range(self.wd.shape[0])]),
            "y": np.array([self.ry[i][lower_poly[i]]
                           for i in range(self.wd.shape[0])])
        }

        # self.star_border = self.smooth(
        #     self.star_border, 1, alpha=0.5)
        # self.port_border = self.smooth(
        #     self.port_border, 1, alpha=0.5)

    def smooth(self, path: Dict[str, np.ndarray],
               every_nth: int = 2, alpha: float = 0.05) -> Dict[str, np.ndarray]:
        """Smooth a given dictionary path

        Args:
            path (dict): dict with x and y coords in it
            every_nth (int): Use only every nth data point of the path
            in the output array

        Returns:
            dict: same dict but with smoothed coords
        """
        x, y, idx = path["x"], path["y"], path["idx"]

        alpha = alpha
        x = self.exponential_smoothing(x, alpha)
        y = self.exponential_smoothing(y, alpha)

        # Only use every nth data point of the path
        if every_nth > 1:
            x = x[::every_nth]
            y = y[::every_nth]
            idx = idx[::every_nth]

        path["x"], path["y"], path["idx"] = x, y, idx

        return path

    def build_state(self) -> np.ndarray:
        """Build the state space

        Returns:
            np.ndarray: state space
        """
        u, v, r, delta = self.ivs[:-1]  # nps is not needed
        if CNN:
            img = self.img_from_state()
            img = np.concatenate(img).ravel()
            features = np.hstack(
                [
                    u/3,
                    v,
                    r/0.01,
                    delta,
                    math.tanh(self.T*self.cte),
                    self.course_error,
                    self.DIR
                ]
            )
            return np.hstack([img, features])

        if self.timestep < 10:
            return np.hstack(
                [
                    u/3, v, r/0.01, delta,
                    np.zeros(40),
                    math.tanh(self.T*self.cte),
                    self.course_error,
                    self.curr_wd - self.vessel["d"],
                    self.DIR,
                    self.rel_current_attack_angle()
                ]
            )
        else:
            return np.hstack(
                [
                    u/3, v, r/0.01, delta,
                    np.hstack(self.history.ivs),
                    math.tanh(self.T*self.cte),
                    self.course_error,
                    self.curr_wd - self.vessel["d"],
                    self.DIR,
                    self.rel_current_attack_angle()
                ]
            )

    def reset_ivs(self) -> np.ndarray:

        return np.array(
            [
                3.0,  # Longitudinal vessel speed [m/s]
                0.0,  # Lateral vessel speed [m/s]
                0.0,  # Yaw rate acceleration [rad/s]
                0.0,  # Rudder angle [rad]
                self.nps  # Propeller revs [s⁻¹]
            ]
        )

    def abs_current_attack_angle(self) -> float:
        """Get the attack angle of river currents.
        We assume the river flow direction is the
        same as the path heading.
        Therefore we calulate the current attack angle
        as the difference between the agent heading
        and the path heading

        Returns:
            float: attack angle of current [rad]
        """

        dbw = self.dist(self.lwp, self.nwp)
        dist_to_last = self.dist(self.agpos, self.lwp)

        dist_on_wp = math.sqrt(dist_to_last**2 - self.cte**2)

        path_heading = self.path_angle(self.lwp, self.nwp)
        next_heading = self.path_angle(
            self.nwp, self.get_wp(self.lwp_idx, 2))

        if path_heading - next_heading <= -math.pi:
            path_heading += 2*math.pi
        elif path_heading - next_heading >= math.pi:
            next_heading += 2*math.pi

        frac = dist_on_wp/dbw
        cta = (1-frac)*path_heading + frac*next_heading

        return cta - math.pi

    def rel_current_attack_angle(self) -> float:

        return self.rel_ang_diff(self.aghead, self.curr_str_dir)

    def get_str_vel(self) -> float:
        """Return the stream velocity based
        on the agent direction

        Returns:
            float: Stream velocity
        """

        return self.curr_str_vel  # if self.DIR==1 else -self.curr_str_vel

    @ staticmethod
    def rel_ang_diff(a1: float, a2: float) -> float:
        """Relative angle difference for an angle range of [0,2*pi]

        Args:
            a1 (float): Angle in radians or degrees
            a2 (float): Angle in radians or degrees

        Returns:
            float: absolute diff in angles or degs
        """

        if abs(a1-a2) <= math.pi:
            if a1 <= a2:
                z = abs(a1-a2)
            else:
                z = a2-a1
        else:
            if a1 < a2:
                z = abs(a1-a2) - 2*math.pi
            else:
                z = 2*math.pi - abs(a1-a2)
        return float(z)

    def moved_forward(self) -> bool:
        """Deprecated

        Returns:
            bool: Moved towards the goal or away from it
        """

        d = self.dist

        wp = self.get_wp(self.waypoint_idx, 4)

        # Distance from current agent position to anchor waypoint
        ag_cur = d(wp, self.agpos)
        # Distance from last agent position to anchor waypoint
        ag_lst = d(wp, self.agpos_lt)

        if ag_cur < ag_lst:
            return True  # Moved towards waypoint
        else:
            return False  # Moved away from waypoint

    def path_angle(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Get the angle in radians from the ordinate
        for two points forming a straight line.
        Angles in qaudrand one are positve

        Args:
            p1 (Tuple): Coordinates of first point. Form: (x,y)
            p2 (Tuple): Coordinates of second point. Form: (x,y)

        Returns:
            float: Angle in radians
        """

        x1, y1, x2, y2 = *p1, *p2

        Y1 = max(y1, y2)
        Y2 = min(y1, y2)

        dist = math.sqrt((x1-x2)**2 + (Y1-Y2)**2)
        angle = math.acos((Y1-Y2)/(dist))

        if x2 < x1 and y2 < y1:
            course = math.pi + angle
        elif x2 > x1 and y2 > y1:
            course = angle
        elif x2 < x1 and y2 > y1:
            course = 2*math.pi - angle
        elif x2 > x1 and y2 < y1:
            course = math.pi - angle

        course = course if self.DIR == 1 else course + math.pi

        return course if course <= 2*math.pi else course - 2*math.pi

    @ staticmethod
    def dist(p1: Tuple, p2: Tuple) -> float:
        """Euler distance

        Args:
            p1 (Tuple): (x,y) of 1st point
            p2 (Tuple): (x,y) of 2nd point

        Returns:
            float: distance between points
        """

        x1, y1 = p1
        x2, y2 = p2

        return math.sqrt((y1-y2)**2 + (x1-x2)**2)

    def cross_track_error(self) -> float:
        """Calculate the cross track error for vector field guidance

        Source:


        Returns:
            float: Cross track error
        """

        d = self.dist

        self.distwp = d(self.agpos, self.lwp)

        agx, agy = self.agpos  # agent postition
        lwpx, lwpy = self.lwp  # last waypoint pos
        nwpx, nwpy = self.nwp  # next waypoint

        # Calculate distance to path as the height of
        # the triangle formed by three points (agpos,lwp,nwp)
        two_a = (nwpx-lwpx)*(lwpy-agy)-(lwpx-agx)*(nwpy-lwpy)

        return two_a/d(self.lwp, self.nwp)

    def desired_course(self, cte: float, k: float) -> float:
        """Calculate the desired course of the vessel

        Args:
            cte (float): cross track error to current path
            k (float): convergance rate of vector field

        Returns:
            float: Desired vessel course in radians
        """
        # Use 10 waypoints (200m) ahead as anchor
        ahead = 1

        dbw = self.dist(self.lwp, self.get_wp(self.lwp_idx, ahead))
        dist_to_last = self.dist(self.agpos, self.lwp)

        dist_on_wp = math.sqrt(dist_to_last**2 - self.cte**2)

        path_heading = self.path_angle(self.lwp, self.nwp)
        next_heading = self.path_angle(
            self.nwp, self.get_wp(self.lwp_idx, ahead + 1))

        if path_heading - next_heading <= -math.pi:
            path_heading += 2*math.pi
        elif path_heading - next_heading >= math.pi:
            next_heading += 2*math.pi

        fact = dist_on_wp/dbw

        tan_cte = math.atan(
            k * -cte) if self.DIR == 1 else math.pi-math.atan(k * cte)
        head = (1-fact)*path_heading + fact*next_heading
        dc = tan_cte + head

        dc = dc if dc < 2*math.pi else dc - 2*math.pi
        dc = dc if dc > 0 else 2*math.pi - dc
        # print(dc)
        return dc

    def crossed_waypoint(self) -> bool:

        # Get the next waypoint in the dir of the vessel that is
        # not already a waypoint
        # self.waypoint_idx = index of waylpoint astern the vessel
        # self.waypoint_idx + 1 = index of waypoint forward the vessel
        # self.waypoint_idx + 2 = index of waypoint to check
        to_check = self.get_wp(self.lwp_idx, 2)

        dist_to_last = self.dist(self.agpos, self.lwp)
        dist_to_check = self.dist(self.agpos, to_check)

        # Last waypoint is still the closest -> do nothing
        if dist_to_last < dist_to_check:
            return False
        else:
            return True

    def update_waypoints(self) -> None:
        """Update the waypoints according to the agents position.
        This function checks whether the waypoint behind the vessel
        is closer than the waypoint 2 indices ahead.

        If the distance to the two-steps-ahead waypoint is closer,
        we know that the vessel has crossed the next waypoint in
        front of it. Therefore last and next waypoint switch, and the
        waypoint to be checked becomes the next waypoint.
        """

        to_check = self.get_wp(self.lwp_idx, 2)
        # Waypoint to check is closer than the old one
        # -> switch waypoints
        self.lwp = self.nwp
        self.nwp = to_check
        if self.DIR == 1:
            self.lwp_idx += 1
        elif self.DIR == -1:
            self.lwp_idx -= 1

    def get_lwp_idx(self, index: int, plus_n: int = 0) -> Tuple[int]:
        """Get a waypoint based on its index

        Args:
            index (int): Index of the waypoint
            plus_n (int): waypoint n ahead of the current
                          dependent on vessel direction

        Returns:
            Tuple: x and y coordinates of the waypoint
        """
        red_indices = self.red_path["idx"]
        for idx, _ in enumerate(red_indices):
            if red_indices[idx] > index:
                if self.DIR == 1:
                    return idx - 1
                elif self.DIR == -1:
                    return idx

    def get_wp(self, index: int, plus_n: int = 0) -> Tuple[float, float]:

        p = self.red_path["idx"]
        if self.DIR == 1:
            x = self.path["x"][p[index + plus_n]]
            y = self.path["y"][p[index + plus_n]]
        elif self.DIR == -1:
            x = self.path["x"][p[index - plus_n]]
            y = self.path["y"][p[index - plus_n]]

        return x, y

    def update_position(self, u: float, v: float) -> None:
        """Transform the numerical integration result from
        vessel fixed coordinate system to earth-fixed coordinate system
        and update the agent's x and y positions.

        Args:
            res (Tuple[float,float]): Tuple (u,v) of integration output:
                u: Longitudinal velocity
                v: Lateral velocity
        """

        # Update absolute positions
        vx = math.cos(self.aghead) * u - math.sin(self.aghead) * v
        vy = math.sin(self.aghead) * u + math.cos(self.aghead) * v

        # Unpack to access values
        agx, agy = self.agpos

        vx, vy = self.swap_xy(vx, vy)

        agx += vx
        agy += vy

        # Repack
        self.agpos = float(agx), float(agy)

        # Update the exterior points of the vessel for plotting
        if not TRAIN_ON_TAURUS:
            self.update_exterior()

    def update_exterior(self) -> None:
        """Update the corner points of the vessel according to
        movement
        """

        # Rectangle of the heading transformed vessel
        self.vessel_rect = Rectangle(self.ship_anchor(),
                                     width=self.vessel["B"],
                                     height=self.vessel["Lpp"],
                                     rotation_point="center",
                                     angle=360-self.aghead*180/math.pi,
                                     color="black")

        # Corner points of the vessel rectangle
        self.vessel_exterior_xy = self.vessel_rect.get_corners()

    def init_vessel(self) -> None:
        """Wrapper to inititalize the vessel"""
        self.update_exterior()

    def update_vessel_heading(self, r: float) -> None:
        """Update the heading by adding the yaw rate to
        the heading.
        Additionally check correct the heading if
        it is larger or smaller than |360°|
        """

        self.aghead += r

        full_circle = 2*math.pi

        if self.aghead > full_circle:
            self.aghead -= full_circle
        elif self.aghead < -full_circle:
            self.aghead += full_circle

    def update_movement_heading(self, u: float, v: float) -> None:

        self.movement_heading = math.atan2(v, u) + self.aghead

    @ staticmethod
    def swap_xy(x: float, y: float) -> Tuple[float, float]:
        """Swaps x and y coordinate in order to assign
        longitudinal motion to the ascending y axis

        Args:
            x (float): old x coord
            y (float): old y coord

        Returns:
            Tuple[float, float]: swapped x,y coords
        """

        return y, x

    def get_river_metrics(self) -> Tuple[float, float]:
        """Get the mean water depth and stream velocity under the vessel:
        Water depth:
            We first grid-search through the x-y coordinate vector
            to find the column and index of the closest known point
            to the current position of the vessel.

            Then the column number and index are used to find
            the corresponding water depth value.

            This is done for each of the four exterior points
            of the vessel

            This is a exhausive search algorithm for now. To
            map each point of the vessel to its corresponding
            water depth, around 200 distances are needed be calculated.

        Stream velocity:
            Since the stream velocity array is of the same shape as the
            water depth array, therefore resembling the same positions,
            we can just take the column and index value for the water depth
            and plug it into the velocity array.

        Returns:
            float: Mean water depth under vessel
        """

        width = 20

        start_idx = self.path["idx"][self.red_path["idx"][self.lwp_idx]]
        search_range = np.arange(start_idx-width, start_idx+width+1)

        # Shorten the distance function
        d = self.dist

        if TRAIN_ON_TAURUS:
            dist = [1000]  # [dist]
            col_idx = [(0, 0)]  # [(col, idx)]
            ptc = [list(self.agpos)]
        else:
            dist = [1000] * 4  # [dist]
            col_idx = [(0, 0)] * 4  # [(col, idx)]
            ptc = self.vessel_exterior_xy

        for idx, point in enumerate(ptc):
            for col in search_range:
                for colidx in range(self.GRIDPOINTS):
                    x = self.rx[col][colidx]
                    y = self.ry[col][colidx]
                    dtp = d((x, y), point)
                    if dtp < dist[idx]:
                        dist[idx] = dtp
                        col_idx[idx] = col, colidx

        wd = [self.wd[col][idx] for col, idx in col_idx]
        str_vel = [self.str_vel[col][idx] for col, idx in col_idx]

        str_dirx = [self.str_dirx[col][idx] for col, idx in col_idx]
        str_diry = [self.str_diry[col][idx] for col, idx in col_idx]
        str_dir = np.arctan2(str_diry, str_dirx)

        return np.mean(wd), np.mean(str_vel), np.mean(str_dir)

    def heading_error(self, desired_heading: float) -> float:
        """Calculate the error from current to desired heading

        Args:
            desired_heading (float): desired heading [rad]

        Returns:
            float: error in [rad]
        """
        return self.rel_ang_diff(desired_heading, self.movement_heading)

    def exponential_smoothing(self, x: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """Simple exponential smoothing

        Args:
            x (np.ndarray): array of input values
            alpha (float, optional): Smoothing alpha. Defaults to 0.05.

        Returns:
            np.ndarray: smoothed array
        """
        s = np.zeros_like(x)

        for idx, x_val in enumerate(x):
            if idx == 0:
                s[idx] = x[idx]
            else:
                s[idx] = alpha * x_val + (1-alpha) * s[idx-1]

        return s

    def render(self, mode: str = "human") -> None:
        if mode == "human":

            if not plt.get_fignums():
                self.fig = plt.figure()
                self.fig.patch.set_facecolor("#212529")
                self.ax: plt.Axes = self.fig.add_subplot(1, 1, 1)

            self.ax.clear()
            self.ax.contourf(self.rx_frame, self.ry_frame,
                             self.wd_frame, cmap=cm.ocean)
            self.ax.quiver(self.rx_frame, self.ry_frame,
                           self.str_diry_frame, self.str_dirx_frame,
                           scale=200, headwidth=1.5)
            self.ax.plot(
                self.exponential_smoothing(self.path_frame[0], alpha=0.08),
                self.exponential_smoothing(self.path_frame[1], alpha=0.08),
                color="red",
                marker=None)
            # self.ax.plot(*self.agpos, color="yellow", marker="o", lw=15)
            self.ax.plot(self.port_border["x"],
                         self.port_border["y"], color="maroon")
            self.ax.plot(self.star_border["x"],
                         self.star_border["y"], color="maroon")

            self.ax.arrow(
                *self.draw_heading(self.aghead, len=100),
                color="yellow", width=5,
                label=f"Vessel Heading: {np.round(self.aghead*180/math.pi,2)}°")
            self.ax.arrow(
                *self.draw_heading(self.dc),
                color="green", width=5,
                label=f"Desired Heading: {np.round(self.dc*180/math.pi,2)}°")
            self.ax.arrow(
                *self.draw_heading(self.movement_heading),
                color="orange", width=5,
                label=f"Movement Heading: {np.round(self.movement_heading*180/math.pi,2)}°")

            self.ax.add_patch(self.vessel_rect)

            handles, _ = self.ax.get_legend_handles_labels()
            speed_count = Patch(
                color="white", label=f"Vessel Speed: {round(self.speed,2)} m/s")
            wuk = round(self.curr_wd - self.vessel["d"], 2)
            wd_below_keel = Patch(
                color="white", label=f"Water under Keel: {wuk} m")
            cta = np.round(self.curr_str_dir*180/math.pi, 2)
            cta_patch = Patch(
                color="white", label=f"Current Attack Angle: {cta}°")

            handles.append(speed_count)
            handles.append(wd_below_keel)
            handles.append(cta_patch)
            self.ax.legend(handles=handles)
            self.ax.set_facecolor("#363a47")

            zoom = 800
            self.ax.set_xlim(self.agpos[0] - zoom *
                             1.5, self.agpos[0] + zoom*1.5)
            self.ax.set_ylim(self.agpos[1] - zoom, self.agpos[1] + zoom)

            plt.pause(0.001)

        else:
            raise NotImplementedError(
                "Currently no other mode than 'human' is available.")

    def img_from_state(self) -> None:

        max_brightness = 255
        mh = self.movement_heading

        fig = Figure(figsize=(3, 3), dpi=40, layout="tight")
        fig.patch.set_facecolor("#000000")
        width, height = fig.get_size_inches() * fig.get_dpi()
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        # Base orientation of plot
        base = ax.transData
        # Rotational transform object
        rot = transforms.Affine2D().rotate(mh)

        # Man rotation of agent pos to use it for zooming
        agx = self.agpos[0]*math.cos(mh)-self.agpos[1]*math.sin(mh)
        agy = self.agpos[0]*math.sin(mh)+self.agpos[1]*math.cos(mh)

        zoom = 1000

        def path():
            ax.clear()
            ax.set_xlim(agx - zoom, agx + zoom)
            ax.set_ylim(agy - zoom/50, agy + zoom)
            ax.margins(-.05, -.05)
            ax.axis('off')
            ax.plot(agx, agy, c="w", marker=".", lw=1)
            ax.plot(*self.path_frame, lw=2, c="w", transform=rot+base)
            canvas.draw()
            raw = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
            return raw.reshape(int(height), int(width), 3)

        def water_depth():
            ax.clear()
            ax.set_xlim(agx - zoom, agx + zoom)
            ax.set_ylim(agy - zoom/50, agy + zoom)
            ax.margins(-.05, -.05)
            ax.axis('off')
            ax.plot(agx, agy, c="w", marker=".", lw=1)
            ax.contourf(
                self.rx_frame,
                self.ry_frame,
                self.wd_frame,
                cmap=cm.Greys_r,
                transform=rot+base)
            canvas.draw()
            raw = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
            return raw.reshape(int(height), int(width), 3)

        def str_vel():
            ax.clear()
            ax.set_xlim(agx - zoom, agx + zoom)
            ax.set_ylim(agy - zoom/50, agy + zoom)
            ax.margins(-.05, -.05)
            ax.axis('off')
            ax.plot(agx, agy, c="w", marker=".", lw=1)
            ax.contourf(
                self.rx_frame,
                self.ry_frame,
                self.str_vel_frame,
                cmap=cm.Greys_r,
                transform=rot+base)
            canvas.draw()
            raw = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
            return raw.reshape(int(height), int(width), 3)

        p = path()[:, :, 0]
        wd = water_depth()[:, :, 0]
        vel = str_vel()[:, :, 0]

        return np.array([p, wd, vel]) / max_brightness

    def img(self) -> np.ndarray:

        fig = Figure(figsize=(3, 3), dpi=40, layout="tight")
        # ax = fig.add_subplot(1, 1, 1)
        fig.patch.set_facecolor("#000000")
        width, height = fig.get_size_inches() * fig.get_dpi()
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.axis('off')

        # Base orientation of plot
        base = ax.transData
        # Rotational transform object
        rot = transforms.Affine2D().rotate_deg(self.movement_heading*180/math.pi)
        # ax.plot(*self.path_frame,c="w",lw=2,transform=rot+base)
        # ax.plot(*self.star_border_frame,c="r",lw=1,transform=rot+base)
        # ax.plot(*self.port_border_frame,c="r",lw=1,transform=rot+base)
        ax.contourf(self.rx_frame, self.ry_frame, self.wd_frame,
                    cmap=cm.Greys_r, transform=rot+base)
        # Man rotation of agent pos to use it for zooming
        agx = self.agpos[0]*math.cos(self.movement_heading) - \
            self.agpos[1]*math.sin(self.movement_heading)
        agy = self.agpos[0]*math.sin(self.movement_heading) + \
            self.agpos[1]*math.cos(self.movement_heading)
        # ax.plot(agx,agy, color="w", marker=".", lw=2)
        zoom = 1000
        ax.set_xlim(agx - zoom, agx + zoom)
        ax.set_ylim(agy - zoom/50, agy + zoom)
        ax.margins(-.05, -.05)

        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(height), int(width), 3)
        img = img.copy()
        img = -img[:, :, 0] + 255  # invert colors
        return img

    def plot_img(self, img: np.ndarray) -> None:

        if not plt.get_fignums():
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

        self.ax.clear()
        self.ax.imshow(img, cmap=cm.gray_r)

        plt.pause(0.01)

    def construct_frames(self) -> None:
        """Construct frames of every metric
        that are used to render CNN input images
        """

        def make(*, obj: Any, la: int, lb: int) -> Any:
            """Frame constructor basis function

            Args:
                obj (Any): metric to frame
                la (int): lookahead distance
                lb (int): lookbehind distance

            Returns:
                Any: Object of same type as `obj`
                     but cut according to la and lb
            """
            if not isinstance(obj, dict):
                wp = self.path["idx"][self.red_path["idx"][self.lwp_idx]]
                if self.DIR == 1:
                    return obj[wp-lb:wp+la]
                else:
                    return obj[wp-la:wp+lb]
            else:
                wp = self.path["idx"][self.red_path["idx"][self.lwp_idx]]
                if self.DIR == 1:
                    return (
                        obj["x"][wp-lb:wp+la],
                        obj["y"][wp-lb:wp+la]
                    )
                else:
                    return (
                        obj["x"][wp-la:wp+lb],
                        obj["y"][wp-la:wp+lb]
                    )

        self.path_frame = make(
            obj=self.path,
            la=self.lookahead,
            lb=self.lookbehind
        )
        self.wd_frame = make(
            obj=self.wd - self.vessel["d"],
            la=self.lookahead,
            lb=self.lookbehind
        )

        self.str_vel_frame = make(
            obj=self.str_vel,
            la=self.lookahead,
            lb=self.lookbehind
        )

        self.rx_frame = make(
            obj=self.rx,
            la=self.lookahead,
            lb=self.lookbehind
        )

        self.ry_frame = make(
            obj=self.ry,
            la=self.lookahead,
            lb=self.lookbehind
        )

        self.port_border_frame = make(
            obj=self.port_border,
            la=self.lookahead,
            lb=self.lookbehind
        )

        self.star_border_frame = make(
            obj=self.star_border,
            la=self.lookahead,
            lb=self.lookbehind
        )

        self.str_dirx_frame = make(
            obj=self.str_dirx,
            la=self.lookahead,
            lb=self.lookbehind
        )

        self.str_diry_frame = make(
            obj=self.str_diry,
            la=self.lookahead,
            lb=self.lookbehind
        )

    def draw_heading(self, angle: float, len: int = 200) -> Tuple[float, float, float, float]:

        x, y = self.agpos

        endx = len * math.sin(angle)
        endy = len * math.cos(angle)

        return float(x), float(y), endx, endy

    def ship_anchor(self) -> Tuple:
        """Build a coordinate tuple resembling the
        anchor point of this rectangle (used to plot the vessel)
        :                + - width - +
        :   y            |           |
        :   |            |           |
        :   |            |           |
        :   |_____x      |           |
        :              height        |
        :                |           |
        :             (anchor)------ +

        Returns:
            Tuple: coordinates of the anchor
        """

        agx, agy = self.agpos

        # Length and Breadth of the simulated vessel
        L, B = self.vessel["Lpp"], self.vessel["B"]

        anchor = agx - B/2, agy - L/2

        return anchor


def import_river(path: str) -> Tuple[Dict[str, list], Dict[str, list]]:
    """Import river metrics from two separate files

    Args:
        path (str): destination of the files

    Returns:
        Tuple[list, list]: list of coordinates and list of metrics
        (water depth, stream velocity)
    """
    data = {
        "coords": [],
        "metrics": []
    }

    if not path.endswith("/"):
        path += "/"

    for file in ["coords", "metrics"]:
        with open(path + file + ".txt", "r") as f:
            reader = csv.reader(f, delimiter=' ')
            # print(f"Loading {file}.txt...",end="\r")
            for row in reader:
                tmp = [float(entry) for entry in row]
                data[file].append(tmp)

    return np.array(data["coords"]), np.array(data["metrics"])


def sigmoid(x: float) -> float:
    def inner(x):
        return 1/(1+np.exp(-x))
    vec = np.vectorize(inner)
    return vec(x)


class History:

    u: float
    v: float
    r: float
    delta: float
    ivs: np.ndarray
    action: int
    timestep: int

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        print("\n".join(f"{key}: {type(val)}" for key,
              val in vars(self).items()))

    def append(self, val: Any, attr: str, mode: str = "train") -> None:

        assert mode in ["train", "test"], "Unknown mode."

        if not hasattr(self, attr):
            setattr(self, attr, [] if mode != "train" else deque(maxlen=10))

        item = getattr(self, attr)
        if isinstance(item, (list, deque)):
            item.append(val)
        else:
            raise RuntimeError(
                "Can only append to 'list' and 'deque', but {} was found".format(type(item)))


fully_loaded_GMS = {
    "m":        3614.89,  # Displacement
    "d":        3.1891,  # Draft
    "A_R":      5.29,  # Rudder Area
    "B":        11.45,  # Width
    "Lpp":      110,  # Length
    "C_b":      0.8,  # Block coefficient
    "t_P":      0.2,  # Thrust deduction factor
    "D_p":      1.751,  # Propeller diameter
    "eta":      0.960,  # Ratio of propeller diameter to rudder span
    "w_P0":     0.4,  # Wake fraction coefficient
    "f_alpha":  2.45  # Rudder lift gradient coefficient
}

GMS = {
    "m":        1735,  # Displacement
    "d":        1.631,  # Draft
    "A_R":      4.29,  # Rudder Area
    "B":        11.45,  # Width
    "Lpp":      110,  # Length
    "C_b":      0.9,  # Block coefficient
    "t_P":      0.2,  # Thrust deduction factor
    "D_p":      1.751,  # Propeller diameter
    "eta":      0.960,  # Ratio of propeller diameter to rudder span
    "w_P0":     0.4,  # Wake fraction coefficient
    "f_alpha":  1.88  # Rudder lift gradient coefficient
}

if __name__ == "__main__":
    COLOR = 'white'
    matplotlib.rcParams['text.color'] = COLOR
    matplotlib.rcParams['axes.labelcolor'] = COLOR
    matplotlib.rcParams['xtick.color'] = COLOR
    matplotlib.rcParams['ytick.color'] = COLOR
   # plt.rcParams['figure.dpi'] = 14

    p = PathFollower()
    s = p.reset()
    rangeq = 2000
    ac = [2, 2] + [0] * 1998
    for i in range(rangeq):
        s2, r, d, _ = p.step(0)
        p.render()
        # p.plot_img(p.img())
        # p.img_from_state()

    # p.render(stay=True)
