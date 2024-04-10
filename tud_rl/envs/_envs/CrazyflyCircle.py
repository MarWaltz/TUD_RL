import os

import gym
import numpy as np
import pybullet as pb
import pybullet_data
from crazyflies_bullet.src.crazyflies_bullet.crazyfly import CrazyFlie
from crazyflies_bullet.src.crazyflies_bullet.physics import PyBulletPhysics
from crazyflies_bullet.src.crazyflies_bullet.sensors import SensorNoise
from crazyflies_bullet.src.crazyflies_bullet.utils import (
    LowPassFilter, deg2rad, get_assets_path, get_quaternion_from_euler)
from gym import spaces
from pybullet_utils import bullet_client


class CrazyflyCircle(gym.Env):
    """Environment to realistically simulate a Crazyfly 2.1 of Bitcraze that should fly in a circle."""
    def __init__(self,
                 aggregate_phy_steps: int = 2,
                 domain_randomization: float = -1,  # deactivated when negative value
                 latency: float = 0.015,  # [s]
                 motor_time_constant: float = 0.080,  # [s]
                 motor_thrust_noise: float = 0.05,  # noise in % added to thrusts
                 observation_frequency: int = 100,
                 observation_noise=1,
                 sim_freq: int = 200,
                 use_graphics=True):
        super(CrazyflyCircle, self).__init__()

        # set params
        self.domain_randomization = domain_randomization
        self.use_graphics = use_graphics
        
        # default simulation constants (in capital letters)
        self.G = 9.81
        self.RAD_TO_DEG = 180 / np.pi
        self.DEG_TO_RAD = np.pi / 180
        self.SIM_FREQ = sim_freq  # default: 200Hz
        self.TIME_STEP = 1. / self.SIM_FREQ  # default: 0.005

        # physics parameters depend on the task
        self.time_step = self.TIME_STEP
        self.number_solver_iterations = 5
        self.aggregate_phy_steps = aggregate_phy_steps

        # === Setup sensor and observation settings
        self.observation_frequency = observation_frequency
        self.obs_rate = int(sim_freq // observation_frequency)
        self.gyro_lpf = LowPassFilter(gain=1., time_constant=2/sim_freq,
                                      sample_time=1/sim_freq)

        # initialize and setup PyBullet
        self.bc = self._setup_client_and_physics()
        self.stored_state_id = -1

        # spawn crazyflie
        self.agent_params = dict(
            aggregate_phy_steps = self.aggregate_phy_steps,
            latency             = latency,
            motor_time_constant = motor_time_constant,
            motor_thrust_noise  = motor_thrust_noise,
            time_step           = self.time_step,
        )
        self._setup_simulation()

        # sensor noise
        self.observation_noise = observation_noise
        use_observation_noise = observation_noise > 0
        self.sensor_noise = SensorNoise(bypass=not use_observation_noise)

        # config
        self.obs_size = 12   # xyz, xyz_dot, rpy, rpy_dot

        self.observation_space = spaces.Box(low  = np.full(self.obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(self.obs_size,  np.inf, dtype=np.float32))
        self.act_size = 1
        self.action_space = spaces.Box(low  = np.full(self.act_size, -1.0, dtype=np.float32), 
                                       high = np.full(self.act_size, +1.0, dtype=np.float32))
        self._max_episode_steps = 200

    def _setup_client_and_physics(self) -> bullet_client.BulletClient:
        if self.use_graphics:
            bc = bullet_client.BulletClient(connection_mode=pb.GUI)
        else:
            bc = bullet_client.BulletClient(connection_mode=pb.DIRECT)

        # add open_safety_gym/envs/data to the PyBullet data path
        bc.setAdditionalSearchPath(get_assets_path())

        # disable GUI debug visuals
        bc.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        bc.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
        bc.setPhysicsEngineParameter(
            fixedTimeStep=self.time_step,
            numSolverIterations=self.number_solver_iterations,
            deterministicOverlappingPairs=1,
            numSubSteps=1)
        bc.setGravity(0, 0, -9.81)
        # bc.setDefaultContactERP(0.9)
        return bc

    def _setup_simulation(self) -> None:
        """Create world layout, spawn agent and obstacles."""
        # reset some variables that might be changed by DR -- this avoids errors
        # when calling the render() method after training.
        self.g = self.G
        self.time_step = self.TIME_STEP

        # also add PyBullet's data path
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.PLANE_ID = self.bc.loadURDF("plane.urdf")
        
        # Load 10x10 Walls
        pb.loadURDF(os.path.join(get_assets_path(), "room_10x10.urdf"), useFixedBase=True)

        # init drone
        self.init_xyz = np.array([0., 0., 1.], dtype=np.float64)
        self.init_rpy = np.zeros(3, dtype=np.float64)
        self.init_quaternion = get_quaternion_from_euler(self.init_rpy)
        self.init_xyz_dot = np.zeros(3, dtype=np.float64)
        self.init_rpy_dot = np.zeros(3, dtype=np.float64)

        self.drone_state0 = dict(
            xyz        = self.init_xyz,
            rpy        = self.init_rpy,
            xyz_dot    = self.init_xyz_dot,
            rpy_dot    = self.init_rpy_dot
        )
        self.drone = CrazyFlie(bc=self.bc, **{**self.agent_params, **self.drone_state0})
        
        # setup forward dynamics
        self.physics = PyBulletPhysics(drone=self.drone, bc=self.bc, time_step=self.time_step)

        # setup task specifics
        self._setup_task_specifics()

    def _setup_task_specifics(self):
        """Initialize task specifics. Called by _setup_simulation()."""
        # === Set camera position
        #self.bc.resetDebugVisualizerCamera(
        #    cameraTargetPosition=(self.drone.xyz[0], self.drone.xyz[1], self.drone.xyz[2]),#(0.0, 0.0, 1.0),
        #    cameraDistance=0.5,
        #    cameraYaw=90,
        #    cameraPitch=-80
        #)
        self.bc.resetDebugVisualizerCamera(
            cameraTargetPosition=(0.0, 0.0, 0.0),
            cameraDistance=1.5,
            cameraYaw=90,
            cameraPitch=-70
        )

    def apply_domain_randomization(self) -> None:
        """ Apply domain randomization at the start of every new episode.

        Initialize simulation constants used for domain randomization. The
        following values are reset at the beginning of every epoch:
            - physics time step
            - Thrust-to-weight ratio
            - quadrotor mass
            - diagonal of inertia matrix
            - motor time constant
            - yaw-torque factors km1 and km2
        """
        def drawn_new_value(default_value,
                            factor=self.domain_randomization,
                            size=None):
            """Draw a random value from a uniform distribution."""
            bound = factor * default_value
            bounds = (default_value - bound, default_value + bound)
            return np.random.uniform(*bounds, size=size)

        if self.domain_randomization > 0:
            # physics parameter
            self.time_step = drawn_new_value(self.TIME_STEP)
            self.physics.set_parameters(
                    time_step=self.time_step,
                    number_solver_iterations=self.number_solver_iterations,
                )

            # === Drone parameters ====
            self.drone.m = drawn_new_value(self.drone.M)
            J_diag = np.array([self.drone.IXX, self.drone.IYY, self.drone.IZZ])
            J_diag_sampled = drawn_new_value(J_diag, size=3)
            self.drone.J = np.diag(J_diag_sampled)
            self.drone.J_INV = np.linalg.inv(self.drone.J)

            self.drone.force_torque_factor_0 = drawn_new_value(
                self.drone.FORCE_TORQUE_FACTOR_0)
            self.drone.force_torque_factor_1 = drawn_new_value(
                self.drone.FORCE_TORQUE_FACTOR_1)

            if self.drone.use_motor_dynamics:
                # set A, B, K according to new values of T_s, T
                mtc = drawn_new_value(self.drone.MOTOR_TIME_CONSTANT, size=4)
                t2w = drawn_new_value(self.drone.THRUST2WEIGHT_RATIO, size=4)
                self.drone.update_motor_dynamics(
                    new_motor_time_constant=mtc,
                    new_sampling_time=self.time_step,
                    new_thrust_to_weight_ratio=t2w
                )
            # set new mass and inertia to PyBullet
            self.bc.changeDynamics(
                bodyUniqueId=self.drone.body_unique_id,
                linkIndex=-1,
                mass=self.drone.m,
                localInertiaDiagonal=J_diag_sampled
            )
        else:
            pass

    def compute_observation(self) -> np.ndarray:
        """Returns the current observation of the environment."""
        if self.observation_noise > 0:  # add noise only for positive values

            if self.step_cnt % self.obs_rate == 0:
                # === 100 Hz Part ===
                # update state information with 100 Hz (except for rpy_dot)
                # apply noise to perfect simulation state:
                xyz, vel, rpy, omega, acc = self.sensor_noise.add_noise(
                    pos=self.drone.xyz,
                    vel=self.drone.xyz_dot,
                    rot=self.drone.rpy,
                    omega=self.drone.rpy_dot,
                    acc=np.zeros(3),  # irrelevant
                    dt=1 / self.SIM_FREQ
                )
                #quat = np.asarray(self.bc.getQuaternionFromEuler(rpy))
                self.state = np.concatenate(
                    [xyz, vel, rpy, omega])
            else:
                # === 200 Hz Part ===
                # This part runs with >100Hz, re-use Kalman Filter values:
                xyz, vel, rpy = self.state[0:3], self.state[3:6], self.state[6:9]

                # read Gyro data with >100 Hz and add noise:
                omega = self.sensor_noise.add_noise_to_omega(
                    omega=self.drone.rpy_dot, dt=1 / self.SIM_FREQ)

            # apply low-pass filtering to gyro (happens with 200Hz):
            omega = self.gyro_lpf.apply(omega)
            obs = np.concatenate(
                [xyz, vel, rpy, omega])
        else:
            # no observation noise is applied
            obs = np.concatenate([self.drone.xyz, self.drone.xyz_dot, 
                                  self.drone.rpy, self.drone.rpy_dot])
        return obs

    def task_specific_reset(self):
        pos     = self.init_xyz.copy()
        xyz_dot = self.init_xyz_dot.copy()
        rpy_dot = self.init_rpy_dot.copy()
        quat    = self.init_quaternion.copy()
        np.random.seed(0)
        # pos
        pos_lim = 0.05
        pos += np.random.uniform(-pos_lim, pos_lim, size=3)

        # quat
        init_angle = deg2rad(20)  # default: 20°
        rpy = np.random.uniform(-init_angle, init_angle, size=3)
        yaw_init = 0.1 * np.pi  # use 2*pi as init
        rpy[2] = np.random.uniform(-yaw_init, yaw_init)
        quat = self.bc.getQuaternionFromEuler(rpy)

        # vel
        vel_lim = 0.1
        xyz_dot += np.random.uniform(-vel_lim, vel_lim, size=3)

        # angle rate
        rpy_dot_lim = deg2rad(50)  # default: 50°/s
        rpy_dot[:2] = np.random.uniform(-rpy_dot_lim, rpy_dot_lim, size=2)
        rpy_dot[2]  = np.random.uniform(-deg2rad(20), deg2rad(20))

        # set drone internals
        self.drone.x = np.random.normal(self.drone.HOVER_X, scale=0.02,
                                        size=(4,))
        self.drone.y = self.drone.K * self.drone.x
        self.drone.action_buffer = np.clip(
            np.random.normal(self.drone.HOVER_ACTION, 0.02,
                                size=self.drone.action_buffer.shape), -1, 1)

        # init low pass filter with new values:
        self.gyro_lpf.set(x=rpy_dot)

        # update pos and quat
        self.bc.resetBasePositionAndOrientation(
            self.drone.body_unique_id,
            posObj=pos,
            ornObj=quat
        )

        # update vel and omega
        R = np.array(self.bc.getMatrixFromQuaternion(quat)).reshape(3, 3)
        self.bc.resetBaseVelocity(
            self.drone.body_unique_id,
            linearVelocity=xyz_dot,
            # PyBullet assumes world frame, so local frame -> world frame
            angularVelocity=R.T @ rpy_dot
        )

    def reset(self):
        """Resets environment to initial state."""
        # counter
        self.step_cnt = 0

        # disable rendering before resetting
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 0)
        if self.stored_state_id >= 0:
            self.bc.restoreState(self.stored_state_id)
        else:
            # Restoring a saved state circumvents the necessity to load all
            # bodies again..
            self.stored_state_id = self.bc.saveState()

        # resets drone controller and action counter
        self.drone.reset()

        # reset drone state (pos, vel, rpy, omega)
        self.task_specific_reset()

        # DR
        self.apply_domain_randomization()

        # init low pass filter(s) with new values:
        self.gyro_lpf.set(x=self.drone.rpy_dot)

        # collect information from PyBullet simulation
        self.drone.update_information()

        # enable rendering again after resetting
        if self.use_graphics:  
            self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 1)
        
        # init state
        self._set_state()
        self.state_init = self.state
        return self.state

    def _set_state(self):
        self.state = self.compute_observation()

    def step(
            self,
            action: np.ndarray
    ) -> tuple:
        """Step the simulation's dynamics once forward.

        This method follows the interface of the OpenAI Gym.

        Parameters
        ----------
        action: array
            Holding the control commands for the agent.

        Returns
        -------
        observation (object)
            Agent's observation of the current environment
        reward (float)
            Amount of reward returned after previous action
        done (bool)
            Whether the episode has ended, handled by the time wrapper
        info (dict)
            contains auxiliary diagnostic information such as the cost signal
        """
        for _ in range(self.aggregate_phy_steps):

            #   calculate observations aggregate_phy_steps-times to correctly
            #   estimate drone state (due to gyro filter)
            self.physics.step_forward(action)

            # Note: do not delete the following line due to >100 Hz sensor noise
            self.compute_observation()
            self.step_cnt += 1

        # set new state, compute reward and done signal
        self._set_state()
        r = self.compute_reward(action)
        done = self.done()
        return self.state, r, done, {}
    
    def compute_reward(self, action):
        return 0.0

    def done(self):
        if self.step_cnt >= self._max_episode_steps:
            return True
        return False

    def render(self, mode='human') -> np.ndarray:
        """Show PyBullet GUI visualization.

        Render function triggers the PyBullet GUI visualization.
        Camera settings are managed by Task class.

        Note: For successful rendering call env.render() before env.reset()

        Parameters
        ----------
        mode: str

        Returns
        -------
        array
            holding RBG image of environment if mode == 'rgb_array'
        """
        # close direct connection to physics server and
        # create new instance of physics with GUI visuals
        if self.use_graphics:
            #self.bc.disconnect()
            #self.bc = self._setup_client_and_physics(graphics=True)
            #self._setup_simulation(
            #    physics=self.input_parameters['physics']
            #)
            self.drone.show_local_frame()
            # Save the current PyBullet instance as save state
            # => This avoids errors when enabling rendering after training
            self.stored_state_id = self.bc.saveState()
