import torch
from tud_rl.agents._discrete.DQN import DQNAgent
from tud_rl.common.configparser import ConfigFile


class DDQNAgent(DQNAgent):
    def __init__(self, c: ConfigFile, agent_name):
        super().__init__(c, agent_name)

    def _compute_target(self, r, s2, d):
        with torch.no_grad():
            a2 = torch.argmax(self.DQN(s2), dim=1).reshape(self.batch_size, 1)
            Q_next = torch.gather(input=self.target_DQN(s2), dim=1, index=a2)
            
            y = r + self.gamma * Q_next * (1 - d)
        return y
