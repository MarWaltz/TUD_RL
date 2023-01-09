import copy

import torch
import torch.optim as optim
import tud_rl.common.nets as nets
from tud_rl.agents._continuous.LSTMTD3 import LSTMTD3Agent
from tud_rl.common.configparser import ConfigFile


class LSTMRecTD3Agent(LSTMTD3Agent):
    def __init__(self, c: ConfigFile, agent_name):

        # Note: Since the actor and critic weights in the config-file wouldn't fit the plain LSTMTD3,
        #       we need to artificially provide a 'None' entry for them and set the mode to 'train'.
        c_cpy = copy.deepcopy(c)
        c_cpy.overwrite(actor_weights=None)
        c_cpy.overwrite(critic_weights=None)
        c_cpy.overwrite(mode="train")

        # now we can instantiate the parent class and correct the overwritten information, rest as usual
        super().__init__(c_cpy, agent_name)
        self.actor_weights  = c.actor_weights
        self.critic_weights = c.critic_weights
        self.mode = c.mode

        # overwrite nets (Note: 'num_obs_OS' is specific for the HHOS envs.)
        self.num_obs_OS = getattr(c.Agent, agent_name)["num_obs_OS"]
        self.num_obs_TS = getattr(c.Agent, agent_name)["num_obs_TS"]

        if self.state_type == "feature":
            self.actor  = nets.LSTMRecActor(action_dim       = self.num_actions, 
                                            num_obs_OS       = self.num_obs_OS,
                                            num_obs_TS       = self.num_obs_TS,
                                            use_past_actions = self.use_past_actions,
                                            device           = self.device).to(self.device)
            self.critic = nets.LSTMRec_Double_Critic(action_dim       = self.num_actions,
                                                     num_obs_OS       = self.num_obs_OS,
                                                     num_obs_TS       = self.num_obs_TS,
                                                     use_past_actions = self.use_past_actions,
                                                     device           = self.device).to(self.device)

        # number of parameters for actor and critic
        self.n_params = self._count_params(self.actor), self._count_params(self.critic)

        # load prior weights if available
        if self.actor_weights is not None and self.critic_weights is not None:
            self.actor.load_state_dict(torch.load(self.actor_weights, map_location=self.device))
            self.critic.load_state_dict(torch.load(self.critic_weights, map_location=self.device))

        # redefine target nets
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
    
        # freeze target nets with respect to optimizers to avoid unnecessary computations
        for p in self.target_actor.parameters():
            p.requires_grad = False

        for p in self.target_critic.parameters():
            p.requires_grad = False

        # define optimizer
        if self.optimizer == "Adam":
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        else:
            self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=self.lr_actor, alpha=0.95, centered=True, eps=0.01)
            self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=self.lr_critic, alpha=0.95, centered=True, eps=0.01)
