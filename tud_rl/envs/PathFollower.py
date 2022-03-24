import random
import gym
import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib

from gym import spaces
from matplotlib import cm
from matplotlib.patches import Rectangle
from mmgdynamics import dynamics as mmg
from mmgdynamics.calibrated_vessels import GMS1
from getpass import getuser
from typing import Tuple, Dict, Any


TRAIN_ON_TAURUS: bool = False

if TRAIN_ON_TAURUS:
    RIVERDATA = "/home/s2075466/riverdata"
else:
    RIVERDATA = f"/home/{getuser()}/Dropbox/TU Dresden/riverdata"
    matplotlib.use("TKAgg")
    
class PathFollower(gym.Env):
    def __init__(self, mode: str = "step") -> None:
        super().__init__()
        
        assert mode in ["step","abs","cont"]
        self.mode = mode

        # Number of evenly spaced gridpoints for which stream velocity and water depth data is available. 
        # All Gridpoints are BASEPOINT_DIST meters apart to span a maximum rhine width of 500 meters 
        self.GRIDPOINTS: int = 26
        self.BASEPOINT_DIST: int = 20
        self.STARTIND: int = 50
        
        # Convergance rate of vector field for vector field guidance
        self.K: float = 0.005
        
        # Derivative pentalty constant
        self.C: float = -1
        
        # Rudder increment/decrement per action [deg]
        self.RUDDER_INCR: int = 3
        self.MAX_RUDDER: int = 20
        
        # Minimum water under keel [m]
        self.MIN_UNDER_KEEL: float = 1.5
        
        # Current timestep of episode
        self.timestep: int = 1

        # Prepare the river data by importing it from file
        self.coords, self.metrics = self.import_river(RIVERDATA)
        
        # River X coordinate
        self.rx: np.ndarray = np.array([self.coords[row][0] 
                            for row,_ in enumerate(self.coords)])
        self.rx = self.rx.reshape((-1, self.GRIDPOINTS))

        # River Y coordinate    
        self.ry: np.ndarray = np.array([self.coords[row][1] 
                            for row,_ in enumerate(self.coords)])
        self.ry = self.ry.reshape((-1, self.GRIDPOINTS))

        # Extract water depth and reshape
        self.wd: np.ndarray = np.array([math.sqrt(self.metrics[row][1]**2 + self.metrics[row][2]**2) 
                                for row,_ in enumerate(self.metrics)])
        self.wd = self.wd + 2 # Lower the rhine by 3 meters to prevent the dynamics to crash
        self.wd = self.wd.reshape((-1, self.GRIDPOINTS))

        # Extract stream velocity and reshape
        self.str_vel: np.ndarray = np.array([self.metrics[row][2] 
                                 for row,_ in enumerate(self.metrics)])
        self.str_vel = self.str_vel.reshape((-1, self.GRIDPOINTS))
        
        # Index list of the path to follow
        self.path_index: np.ndarray = np.empty(self.wd.shape[0],dtype=int)

        # Path roughness (Use only every nth data point of the path)
        self.ROUGHNESS: int = 5

        self.path: Dict[str,list[float]] = self.get_river_path(roughness=self.ROUGHNESS)
        
        
        # Vessel set-up ------------------------------------------------
        
        # Propeller revolutions [s⁻¹]
        self.nps = 5.0
        
        # Rudder angle [rad]
        self.delta = 0.0
        
        # Vessel to be simulated
        self.vessel: Dict[str,float] = GMS1
        
        # Get the border to port and starboard where the water depth
        # equal to the ship draft
        self.build_fairway_border()
        
        # Initial values 
        self.ivs: np.ndarray = np.array(
            [
                4.0, # Longitudinal vessel speed [m/s]
                0.0, # Lateral vessel speed [m/s]
                0.0, # Yaw rate acceleration [rad/s]
                self.delta, # Rudder angle [rad]
                self.nps # Propeller revs [s⁻¹]
            ]
        )
        
        self.history = History()
        
        # Gym Action and observation space -------------------------------------
        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (9,)
        )
        
        if self.mode == "step":
            self.n_actions = 3
            self.action_space = spaces.Discrete(self.n_actions)
        elif self.mode == "abs":
            self.n_actions = 9
            self.action_space = spaces.Discrete(self.n_actions)
        elif self.mode == "cont":
            self.n_actions = 1
            self.action_space = spaces.Box(
                low   = -1,
                high  =  1,
                shape = (1,)
            )
        self.max_episode_steps = 10000
    
    def reset(self) -> np.ndarray:
        
        start_x = lambda x: self.path["x"][x]
        start_y = lambda y: self.path["y"][y]

        # Index of the current waypoint that is in use. Starts with its starting value
        #self.waypoint_idx = self.STARTIND
        self.waypoint_idx = random.randrange(50,500)

        # Last waypoint and next waypoint
        self.lwp = self.get_wp(self.waypoint_idx)
        self.nwp = self.get_wp(self.waypoint_idx + 1)
        
        # Agent position plus some random noise
        self.agpos = (
            (start_x(self.waypoint_idx) + start_x(self.waypoint_idx + 1)) / 2 + random.uniform(-50,50), 
            (start_y(self.waypoint_idx) + start_y(self.waypoint_idx + 1)) / 2 + random.uniform(-50,50)
            )

        # Agent position last timestep
        self.agpos_lt = self.agpos

        # Set agent heading to path heading plus some small noise (5° = 1/36*pi rad)
        random_angle = 1/36*math.pi
        self.aghead = self.path_angle(p1=(start_x(self.waypoint_idx),
                                          start_y(self.waypoint_idx)),
                                      p2=(start_x(self.waypoint_idx+1),
                                          start_y(self.waypoint_idx+1))) + random.uniform(-random_angle,
                                                                                           random_angle)
        
        # Create the vessel obj and place its center to self.agpos
        if not TRAIN_ON_TAURUS:
            self.init_vessel()
        
        # Get current water depth and stream velocity 
        # for the current postition 
        # Water depth in [m]
        # Stream velocity in [m/s]
        self.curr_wd, self.curr_str_vel = self.get_river_metrics()
        
        # Cross track error [m]
        self.cte = self.cross_track_error()

        # Desired course [rad]
        self.dc = self.desired_course(self.cte, self.K)

        # Heading error [rad]
        self.course_error = self.heading_error(self.dc)
        
        
        self.state = self.build_state()

        return self.state

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        
        action = float(action) if self.mode=="cont" else int(action)
        
        # Reset the done flag on every step
        self.done: bool = False
        if self.timestep >= self.max_episode_steps:
            self.done = True
        
        # Store old delta
        delta_old = self.delta
        
        self.map_action(action)

        # Update the initial values for the rudder angle
        self.ivs[3] = self.delta

        # Simulate the vessel with the MMG model for one second (default)
        sol = mmg.step(
            X = self.ivs,
            params = self.vessel,
            nps_old = self.nps,
            delta_old = delta_old,
            fl_psi = self.current_attack_angle(),
            water_depth = self.curr_wd,
            #water_depth = None,
            #fl_vel = None,
            fl_vel = self.curr_str_vel,
            atol = 1e-4, 
            rtol = 1e-4,
            sps = 1
        )
        
        # Unpack values of solver
        u, v, r, *_ = sol
        
        for val,name in zip([u,v,r,self.delta,action,self.timestep],
                            ["u","v","r","delta","action","timestep"]):
            self.history.append(val,name)
        
        # Update state
        self.update_heading(r)
        self.update_position((u,v))
        self.update_waypoints()
        self.curr_wd, self.curr_str_vel = self.get_river_metrics()
        
        # Cross track error [m]
        self.cte = self.cross_track_error()

        # Desired course [rad]
        self.dc = self.desired_course(self.cte, self.K)

        # Heading error [rad]
        self.course_error = self.heading_error(self.dc)
        
        reward = self.calculate_reward()
        
        # Set initial values to the result of the last iteration
        self.ivs = np.hstack(sol)
        
        # Rebuild state
        self.state = self.build_state()
        
        if self.done:
            self.timestep = 1
        else:
            self.timestep += 1
        
        print(round(reward,2),self.timestep)
        return self.state, reward, self.done, {}
    
    def map_action(self, action: int) -> None:
        """Maps an integer action to a change in rudder angle
        depending on the mode of training

        Args:
            action (int): Integer value representing an action
        """
        
        rad = lambda a: a/180*math.pi
        
        if self.mode == "step":
            # Clip rudder angle to plus-minus 35°
            if not self.delta <= self.MAX_RUDDER or self.delta >= self.MAX_RUDDER:
                if action == 0: # Steer to port
                    self.delta += float(0) # Unneccessary but left in for clarity
                elif action == 1:
                    self.delta -= rad(float(self.RUDDER_INCR))
                elif action == 2: # Steer to starboard
                    self.delta += rad(float(self.RUDDER_INCR))

        elif self.mode == "abs":
            action -= int(self.n_actions/2)
            self.delta = rad(float(action*self.RUDDER_INCR))
            
        elif self.mode == "cont":
            rud = action * self.MAX_RUDDER
            self.delta += rad(rud)
            
        
    def calculate_reward(self) -> float:
        
        draft = self.vessel["d"]
        wd = self.curr_wd
        
        if wd - draft < self.MIN_UNDER_KEEL:
            self.done = True
            return -100

        # Derivative penalty
        try:
            old_action = self.history.action[-2]
            action = self.history.action[-1]
            deriv = abs(action - old_action)
            deriv_rew = deriv**2 if self.mode=="cont" else (deriv/(self.n_actions-1))**2
            deriv_rew = self.C * deriv_rew
        except:
            deriv_rew = 0
        
        # Hyperparameter controling the steepness 
        # of the exponentiated cross track error
        kcte = -0.1
        cte_rew = math.exp(kcte * abs(self.cte))
        
        # Reward for heading angle
        khead = -2
        if self.course_error > 0:
            ang_rew = math.exp(khead * self.course_error)
        else:
            ang_rew = math.exp(-khead * self.course_error)
            
        # Reward for avoiding shallow waters
        #wdr = -np.max(self.wd)/wd
            
        return cte_rew + deriv_rew + ang_rew
    
    def get_river_path(self, roughness: int)-> dict:
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
            "y": np.empty(self.wd.shape[0])
        }

        # To use only the middle portion of the river
        offset = self.GRIDPOINTS // 3

        #print("Building waypoints...")
        for col in range(self.wd.shape[0]):
            frame = self.wd[col][offset:2*offset+1] # Get middle thrid per column
            max_ind = np.argmax(frame) # Find deepest index
            max_ind += offset # Add back offset
            
            # Add the index to the path index list
            self.path_index[col] = int(max_ind)
            
            # Get the coordinates of the path by returning the x and y coord
            # for the calculated max index per column
            path["x"][col] = self.rx[col][max_ind]
            path["y"][col] = self.ry[col][max_ind]
            
        # Exponential smoothing for coordinates 
        path = self.smooth(path, roughness)

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
        
        free_space: np.ndarray[bool] = self.wd - draft - self.MIN_UNDER_KEEL < 0
        
        lower_poly = np.full(free_space.shape[0],int(0))
        upper_poly = np.full(free_space.shape[0],self.GRIDPOINTS - 1)
        
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
        
        self.star_border = self.smooth(self.star_border,self.ROUGHNESS)
        self.port_border = self.smooth(self.port_border,self.ROUGHNESS)
        

    def smooth(self, path: Dict[str,np.ndarray], every_nth: int = 2) -> Dict[str,np.ndarray]:
        """Smooth a given dictionary path

        Args:
            path (dict): dict with x and y coords in it
            every_nth (int): Use only every nth data point of the path
            in the output array

        Returns:
            dict: same dict but with smoothed coords
        """
        x,y = path["x"], path["y"]

        alpha = 0.1
        x = self.exponential_smoothing(x,alpha)
        y = self.exponential_smoothing(y,alpha)

        # Only use every nth data point of the path
        if every_nth > 1:
            x = x[::every_nth]
            y = y[::every_nth]

        path["x"], path["y"] = x, y

        return path

    def build_state(self) -> np.ndarray:
        """Build the state space

        Returns:
            np.ndarray: state space
        """
        
        return np.hstack(
            [
                self.ivs[:-1], # nps is not needed
                self.aghead,
                self.cte,
                self.course_error,
                self.curr_wd - self.vessel["d"],
                self.curr_str_vel
            ]
        )

    def current_attack_angle(self) -> float:
        """Get the attack angle of river currents.
        We assume the river flow direction is the
        same as the path heading. 
        Therefore we calulate the current attack angle
        as the difference between the agent heading
        and the path heading

        Returns:
            float: attack angle of current [rad]
        """
        
        return self.path_angle(self.lwp,self.nwp) - self.aghead
        
    
    def moved_forward(self) -> bool:
        """Deprecated

        Returns:
            bool: Moved towards the goal or away from it
        """
        
        d = self.dist
        
        wp = self.get_wp(self.waypoint_idx + 4)
        
        # Distance from current agent position to anchor waypoint
        ag_cur = d(wp, self.agpos) 
        # Distance from last agent position to anchor waypoint
        ag_lst = d(wp, self.agpos_lt)
        
        if ag_cur < ag_lst:
            return True # Moved towards waypoint
        else: 
            return False # Moved away from waypoint
    
    @staticmethod
    def path_angle(p1: Tuple[float,float], p2: Tuple[float,float]) -> float:
        """Get the angle in radians from the ordinate
        for two points forming a straight line.
        Angles in qaudrand one are positve

        Args:
            p1 (Tuple): Coordinates of first point. Form: (x,y)
            p2 (Tuple): Coordinates of second point. Form: (x,y)

        Returns:
            float: Angle in radians
        """
        
        x1,y1,x2,y2 = *p1, *p2
        
        Y1 = max(y1,y2)
        Y2 = min(y1,y2)

        dist = math.sqrt((x1-x2)**2 + (Y1-Y2)**2)
        angle = math.acos((Y1-Y2)/(dist))
        
        if x2 < x1 and y2 < y1:
            return -math.pi + angle
        elif x2 > x1 and y2 > y1:
            return angle
        elif x2 < x1 and y2 > y1:
            return -angle
        elif x2 > x1 and y2 < y1:
            return math.pi - angle
    
    @staticmethod
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
        
        agx, agy = self.agpos # agent postition
        lwpx, lwpy = self.lwp # last waypoint pos
        nwpx, nwpy = self.nwp # next waypoint

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
        
        dbw = self.dist(self.lwp, self.nwp)
        dist_to_last = self.dist(self.agpos,self.lwp)
        
        path_heading = self.path_angle(self.lwp,self.nwp)
        next_heading = self.path_angle(self.nwp,self.get_wp(self.waypoint_idx+2))
        
        fact = dist_to_last/dbw
        
        tan_cte = math.atan(k * -cte)
        head = (1-fact)*path_heading + fact*next_heading

        return tan_cte + head
        
    def update_waypoints(self) -> None:
        """Update the waypoints according to the agents position.
        This function checks whether the waypoint behind the vessel
        is closer than the waypoint 2 indices ahead.
        
        If the distance to the two-steps-ahead waypoint is closer,
        we know that the vessel has crossed the next waypoint in
        front of it. Therefore last and next waypoint switch, and the
        waypoint to be checked becomes the next waypoint.
        """
        
        # Get the next waypoint in the dir of the vessel that is
        # not already a waypoint
        # self.waypoint_idx = index of waylpoint astern the vessel
        # self.waypoint_idx + 1 = index of waypoint forward the vessel
        # self.waypoint_idx + 2 = index of waypoint to check
        to_check = self.get_wp(self.waypoint_idx + 2)
        
        dist_to_last = self.dist(self.agpos, self.lwp)
        dist_to_check = self.dist(self.agpos, to_check)
        
        # Last waypoint is still the closest -> do nothing
        if dist_to_last < dist_to_check:
            return
        
        # Waypoint to check is closer than the old one
        # -> switch waypoints
        self.lwp = self.nwp
        self.nwp = to_check
        self.waypoint_idx += 1
        
    def get_wp(self, index: int) -> Tuple[float, float]:
        """Get a waypoint based on its index

        Args:
            index (int): Index of the waypoint

        Returns:
            Tuple: x and y coordinates of the waypoint
        """
        
        x = self.path["x"][index-1]
        y = self.path["y"][index-1]
        
        return x,y
    
    def update_position(self, res: Tuple[float,float]) -> None:
        """Transform the numerical integration result from 
        vessel fixed coordinate system to earth-fixed coordinate system
        and update the agent's x and y positions.

        Args:
            res (Tuple[float,float]): Tuple (u,v) of integration output:
                u: Longitudinal velocity
                v: Lateral velocity
        """
        
        u, v = res

        # Update absolute positions
        vx = math.cos(self.aghead) * u - math.sin(self.aghead) * v
        vy = math.sin(self.aghead) * u + math.cos(self.aghead) * v

        # Unpack to access values
        agx, agy = self.agpos 
        
        vx, vy = self.swap_xy(vx,vy)
        
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
                                     color="yellow")

        # Corner points of the vessel rectangle
        self.vessel_exterior_xy = self.vessel_rect.get_corners()
        
    def init_vessel(self) -> None:
        """Wrapper to inititalize the vessel"""
        self.update_exterior()
        
        
    def update_heading(self, r: float) -> None:
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

    @staticmethod
    def swap_xy(x: float, y: float) -> Tuple[float, float]:
        """Swaps x and y coordinate in order to assign
        longitudinal motion to the ascending y axis

        Args:
            x (float): old x coord
            y (float): old y coord

        Returns:
            Tuple[float, float]: swapped x,y coords
        """

        return y,x
        
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
        
        start_idx = int(self.waypoint_idx * self.ROUGHNESS)
        width = 4
        
        # Shorten the distance function
        d = self.dist
        
        search_range = np.arange(start_idx-width, start_idx+width+1)
        
        if TRAIN_ON_TAURUS:
            dist = [1000] # [dist]
            col_idx = [(0,0)] # [(col, idx)]
            ptc = [list(self.agpos)]
        else:
            dist = [1000] * 4 # [dist]
            col_idx = [(0,0)] * 4 # [(col, idx)]
            ptc = self.vessel_exterior_xy
            
        for idx, point in enumerate(ptc):
            for col in search_range:
                for colidx in range(self.GRIDPOINTS):
                    x = self.rx[col][colidx]
                    y = self.ry[col][colidx]
                    dtp = d((x,y),point)
                    if dtp < dist[idx]:
                        dist[idx] = dtp
                        col_idx[idx] = col, colidx
        
        wd = [self.wd[col][idx] for col,idx in col_idx]
        str_vel = [self.str_vel[col][idx] for col,idx in col_idx]

        return np.mean(wd), np.mean(str_vel)
                        

    def heading_error(self, desired_heading: float) -> float:
        """Calculate the error from current to desired heading

        Args:
            desired_heading (float): desired heading [rad]

        Returns:
            float: error in [rad]
        """
        return desired_heading - self.aghead
        
    
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

    def render(self, stay: bool, mode: str = "human") -> None:
        if mode == "human":

            if not plt.get_fignums():
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(1, 1, 1)
                self.fig.patch.set_facecolor("#1d1f26")
                self.ax.contourf(self.rx, self.ry, self.wd, cmap = cm.winter_r)
                self.ax.plot(self.path["x"],self.path["y"], color="red")
                self.ax.plot(self.port_border["x"],self.port_border["y"], color="maroon")
                self.ax.plot(self.star_border["x"],self.star_border["y"], color="maroon")
                self.ax.plot(*self.agpos, color="yellow", marker="o", lw=15)
                self.ax.add_patch(self.vessel_rect)

                self.ax.arrow(*self.draw_heading(self.aghead), 
                              color="yellow", width=5, 
                              label=f"Current Heading: {np.round(self.aghead*180/math.pi,2)}°")
                self.ax.arrow(*self.draw_heading(self.dc), 
                              color="green", width=5, 
                              label=f"Desired Heading: {np.round(self.dc*180/math.pi,2)}°")
                self.ax.legend()
                self.ax.set_facecolor("#363a47")
                
                zoom = 1000
                self.ax.set_xlim(self.agpos[0] - zoom,self.agpos[0] + zoom)
                self.ax.set_ylim(self.agpos[1] - zoom,self.agpos[1] + zoom)
                
                if stay:
                    plt.show()
                else:
                    plt.pause(0.1)

            else:
                self.ax.clear()                
                self.ax.contourf(self.rx, self.ry, self.wd, cmap = cm.winter_r)
                self.ax.plot(self.path["x"],self.path["y"], color="red",marker=".")
                self.ax.plot(*self.agpos, color="yellow", marker="o", lw=15)
                self.ax.plot(self.port_border["x"],self.port_border["y"], color="maroon")
                self.ax.plot(self.star_border["x"],self.star_border["y"], color="maroon")
                self.ax.add_patch(self.vessel_rect)

                self.ax.arrow(*self.draw_heading(self.aghead), 
                              color="yellow", width=5, 
                              label=f"Current Heading: {np.round(self.aghead*180/math.pi,2)}°")
                self.ax.arrow(*self.draw_heading(self.dc), 
                              color="green", width=5, 
                              label=f"Desired Heading: {np.round(self.dc*180/math.pi,2)}°")
                self.ax.legend()
                self.ax.set_facecolor("#363a47")
                    
                zoom = 1000
                self.ax.set_xlim(self.agpos[0] - zoom,self.agpos[0] + zoom)
                self.ax.set_ylim(self.agpos[1] - zoom,self.agpos[1] + zoom)
                    
                plt.pause(0.1)

        else:
            raise NotImplementedError("Currently no other mode than 'human' is available.")

    def draw_heading(self, angle: float)-> Tuple[float,float,float,float]:
        
        x,y = self.agpos
        
        endx = 200 * math.sin(angle)
        endy = 200 * math.cos(angle)
        
        return float(x),float(y), endx, endy
    
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
            
    @staticmethod
    def import_river(path: str) -> Tuple[Dict[str,list], Dict[str,list]]:
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

        for file in ["coords","metrics"]:
            with open(path + file + ".txt", "r") as f:
                reader = csv.reader(f, delimiter=' ')
                #print(f"Loading {file}.txt...",end="\r")
                for row in reader:
                    tmp = [float(entry) for entry in row]
                    data[file].append(tmp)

        return data["coords"], data["metrics"]

class History:  
    def __init__(self) -> None:
        pass
        
    def __repr__(self) -> str:
        print("\n".join(f"{key}: {type(val)}" for key, val in vars(self).items()))
    
    def append(self, val: Any, attr: str) -> None:
        
        if not hasattr(self,attr):
            setattr(self,attr, [])

        item = getattr(self,attr)
        if isinstance(item,list):
            item.append(val)
        else:
            raise RuntimeError("Can only append to 'list', but {} was found".format(type(item)))
            
            

if __name__ == "__main__":
    COLOR = 'white'
    matplotlib.rcParams['text.color'] = "black"
    matplotlib.rcParams['axes.labelcolor'] = COLOR
    matplotlib.rcParams['xtick.color'] = COLOR
    matplotlib.rcParams['ytick.color'] = COLOR

    p = PathFollower()
    s = p.reset()
    rangeq=500
    ac = [0,0] + [1] * 498
    for i in range(rangeq):
        #s2,r,d,_ = p.step(random.randrange(3))
        s2,r,d,_ = p.step([0])
        #print(round(r,2),p.timestep)
        p.render(stay=False)
    
    #p.render(stay=True)