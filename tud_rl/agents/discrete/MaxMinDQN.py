import torch
from tud_rl.agents.discrete.EnsembleDQN import EnsembleDQNAgent
from tud_rl.common.logging_func import *


class MaxMinDQNAgent(EnsembleDQNAgent):
    def __init__(self, c, agent_name, logging=True):
        super().__init__(c, agent_name, logging=False)
     
        # init logger and save config
        if logging:
            self.logger = EpochLogger(alg_str = self.name, env_str = self.env_str)
            self.logger.save_config({"agent_name" : self.name, **c})

            print("--------------------------------------------")
            print(f"n_params: {self.N * self._count_params(self.EnsembleDQN[0])}")
            print("--------------------------------------------")


    def _ensemble_reduction(self, q_ens):
        """
        Input:  torch.Size([N, batch_size, num_actions])
        Output: torch.Size([batch_size, num_actions])
        """
        return torch.min(q_ens, dim=0).values
