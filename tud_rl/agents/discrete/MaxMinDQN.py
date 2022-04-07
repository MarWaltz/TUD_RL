import torch
from tud_rl.agents.discrete.EnsembleDQN import EnsembleDQNAgent


class MaxMinDQNAgent(EnsembleDQNAgent):
    def __init__(self, c, agent_name):
        super().__init__(c, agent_name)
     
    def _ensemble_reduction(self, q_ens):
        """
        Input:  torch.Size([N, batch_size, num_actions])
        Output: torch.Size([batch_size, num_actions])
        """
        return torch.min(q_ens, dim=0).values
