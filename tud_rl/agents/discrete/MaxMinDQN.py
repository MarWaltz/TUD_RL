import torch
from tud_rl.agents.discrete.EnsembleDQN import EnsembleDQNAgent
from tud_rl.common.logging_func import *


class MaxMinDQNAgent(EnsembleDQNAgent):
    def __init__(self, c, agent_name):
        super().__init__(c, agent_name, logging=True)
     
    def _ensemble_reduction(self, q_ens):
        """
        Input:  torch.Size([N, batch_size, num_actions])
        Output: torch.Size([batch_size, num_actions])
        """
        return torch.min(q_ens, dim=0).values
