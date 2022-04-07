import copy
import warnings

import torch
import torch.optim as optim
from tud_rl.agents.discrete.DDQN import DDQNAgent
from tud_rl.common.logging_func import *
from tud_rl.common.nets import RecDQN


class RecDQNAgent(DDQNAgent):
    def __init__(self, c, agent_name, logging=True):

        # Note: Since the 'dqn_weights' in the config-file wouldn't fit the plain DQN,
        #       we need to artificially provide a 'None' entry for them and set the mode to 'train'.
        c_cpy = copy.copy(c)
        c_cpy["dqn_weights"] = None
        c_cpy["mode"]        = "train"

        # now we can instantiate the parent class and correct the overwritten information, rest as usual
        super().__init__(c_cpy, agent_name, logging=False)
        self.dqn_weights = c["dqn_weights"]
        self.mode        = c["mode"]

        # checks
        assert not (self.mode == "test" and (self.dqn_weights is None)), "Need prior weights in test mode."

        assert self.state_type == "feature", "RecDQN is currently based on features."

        if self.net_struc is not None:
            warnings.warn("The net structure cannot be controlled via the config-spec for LSTM-based agents.")

        # init DQN
        if self.state_type == "feature":
            self.DQN = RecDQN(num_actions = self.num_actions,
                              N_TSs       = c["env"]["env_kwargs"]["N_TSs"]).to(self.device)

        # init logger and save config
        if logging:
            self.logger = EpochLogger(alg_str = self.name, env_str = self.env_str, info = self.info)
            self.logger.save_config({"agent_name" : self.name, **c})
            
            print("--------------------------------------------")
            print(f"n_params: {self._count_params(self.DQN)}")
            print("--------------------------------------------")
        
        # load prior weights if available
        if self.dqn_weights is not None:
            self.DQN.load_state_dict(torch.load(self.dqn_weights))

        # init target net and counter for target update
        self.target_DQN = copy.deepcopy(self.DQN).to(self.device)
        self.tgt_up_cnt = 0
        
        # freeze target nets with respect to optimizers to avoid unnecessary computations
        for p in self.target_DQN.parameters():
            p.requires_grad = False

        # define optimizer
        if self.optimizer == "Adam":
            self.DQN_optimizer = optim.Adam(self.DQN.parameters(), lr=self.lr)
        else:
            self.DQN_optimizer = optim.RMSprop(self.DQN.parameters(), lr=self.lr, alpha=0.95, centered=True, eps=0.01)
