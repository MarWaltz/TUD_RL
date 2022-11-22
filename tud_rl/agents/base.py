from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
import torch

from tud_rl.common.configparser import ConfigFile


class _Agent(ABC):
    """Abstract Base Class for any agent
    defining its strucure.
    """

    @abstractmethod
    def select_action(self, s: np.ndarray) -> Union[int, np.ndarray]:
        """Select an action for the agent to take.
        Must take in a state and output an action.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self) -> None:
        """Trains the agent
        """
        raise NotImplementedError

    @abstractmethod
    def memorize(self, s: np.ndarray,
                 a: Union[int, np.ndarray],
                 r: float, s2: np.ndarray,
                 d: bool) -> None:
        """Add transitions tuple to the experience 
        replay buffer
        """
        raise NotImplementedError

    @abstractmethod
    def print_params(self, n_params: Union[Tuple[int, int], int], case: int) -> None:
        """Prints the number of trainable parameters of an Agent

        Args:
            n_params (int): Number of params of the net
                            If case == 0 (discrete):
                                n_params = n_params
                            If case == 1 (continuous):
                                n_params[0]: n_params actor
                                n_params[1]: n_params critic
            case (int): case [0: discrete, 1: continous]
        """
        raise NotImplementedError


class BaseAgent(_Agent):
    def __init__(self, c: ConfigFile, agent_name: str):

        # attributes and hyperparameters
        self.name             = agent_name
        self.num_actions      = c.num_actions
        self.mode             = c.mode
        self.state_shape      = c.state_shape
        self.state_type       = c.Env.state_type
        self.gamma            = c.gamma
        self.optimizer        = c.optimizer
        self.loss             = c.loss
        self.buffer_length    = c.buffer_length
        self.grad_clip        = c.grad_clip
        self.grad_rescale     = c.grad_rescale
        self.act_start_step   = c.act_start_step
        self.upd_start_step   = c.upd_start_step       
        self.upd_every        = c.upd_every  # used in training files, purely for logging here
        self.batch_size       = c.batch_size
        self.device           = c.device
        self.seed             = c.seed
        self.needs_history    = False

        # checks
        assert c.mode in ["train", "test"], "Unknown mode. Should be 'train' or 'test'."

        assert self.state_type in ["image", "feature"],\
            "'state_type' can be either 'image' or 'feature'."

        if self.state_type == "image":
            assert len(self.state_shape) == 3 and type(self.state_shape) == tuple, \
                "'state_shape' should be: (in_channels, height, width) for images."

        assert self.loss in ["SmoothL1Loss", "MSELoss"], "Pick 'SmoothL1Loss' or 'MSELoss', please."
        assert self.optimizer in ["Adam", "RMSprop"], "Pick 'Adam' or 'RMSprop' as optimizer, please."
        assert self.device in ["cpu", "cuda"], "Unknown device."

        # gpu support
        if self.device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
            print("Using GPU support.")

    def _count_params(self, net):
        """Count the number of parameters of a given net"""
        return sum([np.prod(p.shape) for p in net.parameters()])

    def print_params(self, n_params: Union[Tuple[int, int], int], case: int) -> None:
        """Prints the number of trainable parameters of an Agent

        Args:
            n_params (int): Number of params of the net
            case (int): case [0: discrete, 1: continous]
        """
        if case == 0:
            print("--------------------------------------------")
            print(f"Trainable Parameters: {n_params}")
            print("--------------------------------------------")
        else:
            print("--------------------------------------------")
            print(f"Trainable Parameters Actor: {n_params[0]}\n"
                  f"Trainable Parameters Critic: {n_params[1]}")
            print("--------------------------------------------")
