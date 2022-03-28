from tud_rl.envs.FossenEnv import *


class FossenEnvScenarioOne(FossenEnv):
    """This environment contains four agents, each steering a CyberShip II. They spawn in a N-E-S-W positions and should all turn right."""

    def __init__(self, N_TSs=3, cnt_approach="tau", state_pad=np.nan):
        super().__init__(N_TSs=N_TSs, cnt_approach=cnt_approach, state_pad=state_pad)

    def reset(self):
        """Resets environment to initial state."""

        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # init four goals
        self.goals = [{"N" : 10, "E" : 50},
                      {"N" : 50, "E" : 10},
                      {"N" : 90, "E" : 50},
                      {"N" : 50, "E" : 90}]

        # init four agents
        self.agents = []
        for i in range(self.N_TSs + 1):
            
            if i == 0:
                head   = np.pi
                N_init = 90
                E_init = 50

            elif i == 1:
                head   = 3/2 * np.pi
                N_init = 50
                E_init = 90
            
            elif i == 2:
                head   = 0
                N_init = 10
                E_init = 50
            
            elif i == 3:
                head   = np.pi/2
                N_init = 50
                E_init = 10
            
            self.agents.append(CyberShipII(N_init       = N_init, 
                                           E_init       = E_init, 
                                           psi_init     = head,
                                           u_init       = 0.0,
                                           v_init       = 0.0,
                                           r_init       = 0.0,
                                           delta_t      = self.delta_t,
                                           N_max        = self.N_max,
                                           E_max        = self.E_max,
                                           cnt_approach = self.cnt_approach,
                                           tau_u        = 3.0))

        # set longitudinal speed to near-convergence
        for agent in self.agents:
            agent.nu[0] = agent._u_from_tau_u(agent.tau_u)

        # determine current COLREG situations
        # Note: We have four agents in this scenario. Thus, treating each as a OS, all have three TSs from their perspective
        #       We need to update all those COLREG modes.
        self.TS_COLREGs_all = [[0] * 3] * 4
        self._set_COLREGs()

        # init aggregated state
        self._set_aggregated_state()
        self.state_init = self.state_agg

        return self.state_agg


    def _set_aggregated_state(self):
        """Sets the overall state in this scenario."""
        self.state_agg = []

        for idx, agent in enumerate(self.agents):

            # agent acts as OS
            self.OS = agent
            self.TSs = [TS_agent for TS_idx, TS_agent in enumerate(self.agents) if idx != TS_idx]
            self.TS_COLREGs = self.TS_COLREGs_all[idx]
            self.goal = self.goals[idx]
            
            # compute state from this perspective
            self._set_state()
            self.state_agg.append(self.state)


    def _set_COLREGs(self):
        """Computes for each target ship the current COLREG situation and stores it internally.
        Again: Each agent acts as OS and has three TS, respectively, leading to overall 4 * 3 COLREG modes."""

        # overwrite old situations
        self.TS_COLREGs_all_old = copy.deepcopy(self.TS_COLREGs_all)

        # compute new ones
        self.TS_COLREGs_all = []

        for idx, agent in enumerate(self.agents):

            TS_COLREGs_agent = []

            for TS_idx, TS_agent in enumerate(self.agents):

                if idx == TS_idx:
                    continue
                TS_COLREGs_agent.append(self._get_COLREG_situation(OS=agent, TS=TS_agent))

            self.TS_COLREGs_all.append(TS_COLREGs_agent)


    def step(self, a):
        """The action a is a list of four integers, each representing an action for an agent."""

        # perform control action
        [self.agents[idx]._control(act) for idx, act in enumerate(a)]

        # update resulting tau
        [agent._set_tau() for agent in self.agents]

        # update dynamics
        [agent._upd_dynamics() for agent in self.agents]

        # handle map-leaving of agents
        #self.agents = [self._handle_map_leaving(agent, respawn=False, mirrow=False, clip=True)[0] for agent in self.agents]

        # update COLREG scenarios
        self._set_COLREGs()

        # compute state, reward, done        
        self._set_aggregated_state()
        self._calculate_reward()
        d = self._done()

        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += self.delta_t
        
        return self.state_agg, self.r, d, {}


    def _calculate_reward(self):
        """We don't train anymore, just eyetest. No reward needed."""
        self.r = 0


    def _done(self):
        """Returns boolean flag whether episode is over."""

        # artificial done signal
        if self.step_cnt >= self._max_episode_steps:
            return True
        return False


    def render(self):
        """Renders the current environment."""

        # plot every nth timestep
        if self.step_cnt % 1 == 0: 

            # check whether figure has been initialized
            if len(plt.get_fignums()) == 0:
                self.fig = plt.figure(figsize=(10, 7))
                self.gs  = self.fig.add_gridspec(1, 1)
                self.ax0 = self.fig.add_subplot(self.gs[0, 0]) # ship
                plt.ion()
                plt.show()
            
            # ------------------------------ ship movement --------------------------------
            # clear prior axes, set limits and add labels and title
            self.ax0.clear()
            self.ax0.set_xlim(-5, self.E_max + 5)
            self.ax0.set_ylim(-5, self.N_max + 5)
            self.ax0.set_xlabel("East")
            self.ax0.set_ylabel("North")

            # set agents and goals
            for idx, agent in enumerate(self.agents):
                
                col = plt.rcParams["axes.prop_cycle"].by_key()["color"][idx]
                N0, E0, head0 = agent.eta
               
                rect = self._get_rect(E = E0, N = N0, width = agent.width, length = agent.length, heading = head0,
                                      linewidth=1, edgecolor=col, facecolor='none')
                self.ax0.add_patch(rect)

                # add jets according to COLREGS
                for COLREG_deg in [5, 112.5, 247.5, 355]:
                    self.ax0 = self._plot_jet(axis = self.ax0, E=E0, N=N0, l = self.sight, 
                                              angle = head0 + dtr(COLREG_deg), color=col, alpha=0.3)

                for COLREG_deg in [67.5, 175, 185, 292.5]:
                    self.ax0 = self._plot_jet(axis = self.ax0, E=E0, N=N0, l = self.sight, 
                                            angle = head0 + dtr(COLREG_deg), color=col, alpha=0.3)

                self.ax0.scatter(self.goals[idx]["E"], self.goals[idx]["N"], color=col)

            plt.pause(0.001)
