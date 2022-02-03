import pickle

import numpy as np
import torch
from tud_rl.common.normalizer import Input_Normalizer

class BaseAgent:
    def __init__(self, c, agent_name):

        # attributes and hyperparameters
        self.name             = agent_name
        self.num_actions      = c["num_actions"]
        self.mode             = c["mode"]
        self.state_shape      = c["state_shape"]
        self.state_type       = c["env"]["state_type"]
        self.input_norm       = c["input_norm"]
        self.input_norm_prior = c["input_norm_prior"]
        self.gamma            = c["gamma"]
        self.optimizer        = c["optimizer"]
        self.loss             = c["loss"]
        self.buffer_length    = c["buffer_length"]
        self.grad_clip        = c["grad_clip"]
        self.grad_rescale     = c["grad_rescale"]
        self.act_start_step   = c["act_start_step"]
        self.upd_start_step   = c["upd_start_step"]
        self.upd_every        = c["upd_every"]             # used in training files, purely for logging here
        self.batch_size       = c["batch_size"]
        self.device           = c["device"]
        self.env_str          = c["env"]["name"]
        self.info             = c["env"]["info"]
        self.seed             = c["seed"]

        # checks
        assert c["mode"] in ["train", "test"], "Unknown mode. Should be 'train' or 'test'."
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

    def memorize(self, s, a, r, s2, d):
        """Stores current transition in replay buffer."""
        self.replay_buffer.add(s, a, r, s2, d)
