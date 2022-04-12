import torch
from tud_rl.agents._discrete.EnsembleDQN import EnsembleDQNAgent
from tud_rl.common.configparser import ConfigFile


class MaxMinDQNAgent(EnsembleDQNAgent):
    def __init__(self, c : ConfigFile, agent_name):
        super().__init__(c, agent_name)
     
    def _ensemble_reduction(self, q_ens):
        """
        Input:  torch.Size([N, batch_size, num_actions])
        Output: torch.Size([batch_size, num_actions])
        """
        return torch.min(q_ens, dim=0).values
