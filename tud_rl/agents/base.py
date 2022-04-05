import pickle
import torch

import numpy as np

from abc import ABC, abstractmethod
from typing import Union

from tud_rl.common.configparser import Configfile
from tud_rl.common.logging_func import EpochLogger
from tud_rl.common.normalizer import Input_Normalizer


class AbstractAgent(ABC):
    """Abstract Base Class for all agents.
    """
    
    inp_normalizer: Input_Normalizer
    logger: EpochLogger
    
    @abstractmethod
    def select_action(self, s: np.ndarray) -> Union[int,np.ndarray]:
        """Select an action for the agent to take.
        Must take in a state and output an action
        """
        raise NotImplementedError
    
    @abstractmethod
    def train(self) -> None:
        """Trains the agent
        """
        raise NotImplementedError
    
    @abstractmethod
    def memorize(self,s: np.ndarray, 
                 a: Union[int,np.ndarray], 
                 r: float, s2: np.ndarray, 
                 d:bool) -> None:
        """Add transitions tuple to the experience 
        replay buffer
        """
        raise NotImplementedError
    
class AbstractBootAgent(AbstractAgent):   
     
    @abstractmethod
    def reset_active_head(self) -> None:
        """Resets the active head for a bootstrapped
        DQN Framework
        """
        raise NotImplementedError

class BaseAgent(AbstractAgent):
    def __init__(self, c: Configfile, agent_name: str, logging: bool):

        # attributes and hyperparameters
        self.name             = agent_name
        self.num_actions      = c.num_actions
        self.mode             = c.mode
        self.state_shape      = c.state_shape
        self.state_type       = c.Env.state_type
        self.input_norm       = c.input_norm
        self.input_norm_prior = c.input_norm_prior
        self.gamma            = c.gamma
        self.optimizer        = c.optimizer
        self.loss             = c.loss
        self.buffer_length    = c.buffer_length
        self.grad_clip        = c.grad_clip
        self.grad_rescale     = c.grad_rescale
        self.act_start_step   = c.act_start_step
        self.upd_start_step   = c.upd_start_step
        self.upd_every        = c.upd_every           # used in training files, purely for logging here
        self.batch_size       = c.batch_size
        self.device           = c.device
        self.env_str          = c.Env.name
        self.info             = c.Env.info
        self.seed             = c.seed
        
        if logging:
            self.logger: EpochLogger = EpochLogger(
                alg_str = self.name, 
                env_str = self.env_str, 
                info = self.info)
            self.logger.save_config({"agent_name" : self.name, **c.config_dict})

        # checks
        assert c.mode in ["train", "test"], "Unknown mode. Should be 'train' or 'test'."

        if self.input_norm:
            assert not (self.mode == "test" and self.input_norm_prior is None), "Please supply 'input_norm_prior' in test mode with input normalization."

        assert self.state_type in ["image", "feature"], "'state_type' can be either 'image' or 'feature'."

        if self.state_type == "image":
            assert len(self.state_shape) == 3 and type(self.state_shape) == tuple, "'state_shape' should be: (in_channels, height, width) for images."

            if self.input_norm:
                raise NotImplementedError("Input normalization is not available for images.")

        assert self.loss in ["SmoothL1Loss", "MSELoss"], "Pick 'SmoothL1Loss' or 'MSELoss', please."
        assert self.optimizer in ["Adam", "RMSprop"], "Pick 'Adam' or 'RMSprop' as optimizer, please."
        assert self.device in ["cpu", "cuda"], "Unknown device."

        # gpu support
        if self.device == "cpu":    
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
            print("Using GPU support.")
        
        # input normalizer
        if self.input_norm:
            
            if self.input_norm_prior is not None:
                with open(self.input_norm_prior, "rb") as f:
                    prior = pickle.load(f)
                self.inp_normalizer = Input_Normalizer(state_dim=self.state_shape, prior=prior)
            else:
                self.inp_normalizer = Input_Normalizer(state_dim=self.state_shape, prior=None)
        
    def _count_params(self, net):
        return sum([np.prod(p.shape) for p in net.parameters()])
