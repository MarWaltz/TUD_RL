import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt


class Entity(object):
    def __init__(self):
        self.density = 25.0
        self.mass = 1.0

class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()
        self.size = 0.15
        self.movable = False
        self.collide = False
        self.id = 0.25
        self.color = "orange"

class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        self.size = 0.05
        self.max_speed = None
        self.movable = True
        self.collide = True
        self.u_range = 1.0
        self.id = 0.35
        self.color = "green"

class CoopNavigation(gym.Env):
    """Implements the cooperative navigation game from Lowe et al (2017)."""
    def __init__(self, N_agents=3, cont_acts=False):
        super(CoopNavigation, self).__init__()

        # simulation setup
        self.x_max = 2
        self.y_max = 2
        self.dim_world = 2
        self.dim_color = 3
        self.dt = 0.1
        self.damping = 0.25
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

        # config
        assert N_agents == 3, "Always have three agents in CoopNavigation."
        self.N_agents = N_agents
        self.N_landmarks = N_agents

        # obs and act size for a single agent
        obs_size = 19
        self.observation_space = spaces.Box(low  = np.full(obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(obs_size, np.inf, dtype=np.float32))

        self.cont_acts = cont_acts
        if cont_acts:
            act_size = 2
            self.action_space = spaces.Box(low  = np.full(act_size, -1.0, dtype=np.float32), 
                                           high = np.full(act_size, +1.0, dtype=np.float32))
        else:
            self.action_space = spaces.Discrete(5)

        self._max_episode_steps = 200

    def reset(self):
        """Resets environment to initial state."""
        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # generate agents
        self.agents = [Agent() for _ in range(self.N_agents)]
        for agent in self.agents:
            agent.pos = np.random.uniform(-1.0, 1.0, self.dim_world)
            agent.vel = np.zeros(self.dim_world)

        # generate landmarks
        self.landmarks = [Landmark() for _ in range(self.N_landmarks)]
        for l in self.landmarks:
            l.pos = np.random.uniform(-1.0, 1.0, self.dim_world)

        # init state
        self._set_state()
        self.state_init = self.state
        return self.state
  
    def step(self, a):
        """a is np.array([N_agents, action_dim]), where action_dim = 2 in continuous case, else np.array(N_agents,)."""
        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += self.dt
 
        # apply agent physical controls
        if self.cont_acts:
            p_force = a
        else:
            p_force = np.zeros((self.N_agents, self.dim_world))
            for i, a_i in enumerate(a):
                if a_i == 0:
                    p_force[i, 0] = -1.0
                elif a_i == 1:
                    p_force[i, 0] = +1.0
                elif a_i == 2:
                    p_force[i, 1] = -1.0
                elif a_i == 3:
                    p_force[i, 1] = 1.0
                elif a_i == 4:
                    pass # no-op

        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        
        # integrate physical state
        self.integrate_state(p_force)

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward()
        d = self._done()
        return self.state, self.r, d, {}
 
    def _is_collision(self, agent1 : Agent, agent2 : Agent):
        delta_pos = agent1.pos - agent2.pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def _calculate_reward(self):
        self.r = np.zeros((self.N_agents, 1), dtype=np.float32)

        for i, agent in enumerate(self.agents):
            r_i = 0.0

            # collective reward for landmark distance
            for l in self.landmarks:
                dists = [np.sqrt(np.sum(np.square(a.pos - l.pos))) for a in self.agents]
                r_i -= min(dists)
            
            # collision penalty
            for a in self.agents:
                if a is not agent:
                    if self._is_collision(agent, a):
                        r_i -= 1.0

            self.r[i] = r_i

    def _set_state(self):
        """State contains own position, velocity, relative positions of agents and landmakrs, and their id's. 
        Overall 19 features. Thus, state is np.array([N_agents, 19])."""
        self.state = np.zeros((self.N_agents, 19), dtype=np.float32)

        for i, agent in enumerate(self.agents):
            # position and velocity
            s_i = np.concatenate((agent.pos, agent.vel))

            # relative position and id of other agents
            for other in self.agents:
                if agent is not other:
                    s_i = np.concatenate((s_i, other.pos-agent.pos, np.array([other.id])))
            
            # relative position and id of landmarks
            for l in self.landmarks:
                s_i = np.concatenate((s_i, l.pos-agent.pos, np.array([l.id])))

            self.state[i] = s_i

    def apply_environment_force(self, p_force):
        for i, agent in enumerate(self.agents):
            for j, other_agent in enumerate(self.agents):
                if(i >= j): 
                    continue
                [f_1, f_2] = self.get_collision_force(agent, other_agent)
                p_force[i] += f_1
                p_force[j] += f_2 
        return p_force

    def integrate_state(self, p_force):
        for i, agent in enumerate(self.agents):
            agent.vel *= (1 - self.damping)
            agent.vel += (p_force[i] / agent.mass) * self.dt
            if agent.max_speed is not None:
                speed = np.sqrt(np.square(agent.vel[0]) + np.square(agent.vel[1]))
                if speed > agent.max_speed:
                    agent.vel *= (agent.max_speed/speed)
            agent.pos += agent.vel * self.dt  

    def get_collision_force(self, agent1, agent2):
        # don't collide against itself
        if (agent1 is agent2):
            return [0.0, 0.0]
        
        # compute actual distance between entities
        delta_pos = agent1.pos - agent2.pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        
        # minimum allowable distance
        dist_min = agent1.size + agent2.size

        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_1 = +force
        force_2 = -force
        return [force_1, force_2]

    def _done(self):
        # someone left the map
        #if any([agent.is_off_map() for agent in self.agents]):
        #    return True

        # artificial done signal
        if self.step_cnt >= self._max_episode_steps:
            return True
        return False

    def render(self, mode=None):
        """Renders the current environment."""

        # plot every nth timestep
        if self.step_cnt % 10 == 0: 
            
            # init figure
            if len(plt.get_fignums()) == 0:
                self.f, self.ax1 = plt.subplots(1, 1, figsize=(10, 10))
                plt.ion()
                plt.show()           
            
            # set screen
            self.ax1.clear()
            self.ax1.set_xlim(-self.x_max-0.5, self.x_max+0.5)
            self.ax1.set_ylim(-self.y_max-0.5, self.y_max+0.5)
            self.ax1.set_xlabel("x")
            self.ax1.set_ylabel("y")

            # plot agents and landmarks
            for l in self.landmarks:
                circ = plt.Circle((l.pos[0], l.pos[1]), l.size, color=l.color)
                self.ax1.add_patch(circ)

            for agent in self.agents:
                circ = plt.Circle((agent.pos[0], agent.pos[1]), agent.size, color=agent.color)
                self.ax1.add_patch(circ)
            
            plt.pause(0.1)
