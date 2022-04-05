from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

@dataclass
class Configfile:
    """Configuration class for storing the parsed 
    yaml config file.
    
    Contains all configuration options possible
    for training with the `tud_rl` package.

    """
    
    # General config
    seed: int
    timesteps: int
    epoch_length: int
    eval_episodes: int
    optimizer: str
    loss: str
    buffer_length: int
    grad_rescale: bool
    mode: str
    grad_clip: bool
    act_start_step: int
    upd_start_step: int
    upd_every: int
    batch_size: int
    device: str        
    state_shape: Tuple
    num_actions: int
    input_norm: bool
    input_norm_prior: Optional[str]
    
    # Discrete training
    img_height: Optional[int]
    img_width: Optional[int]
    dqn_weights: Optional[str]
    gamma: float
    eps_init: float
    eps_final: float
    eps_decay_steps: int
    tgt_update_freq: int
    net_struc: List[Union[Union[int,str],str]]
    lr: float
    
    # Continuous training
    action_high: float
    action_low: float
    lr_actor: float
    lr_critic: float
    tau: float
    actor_weights: Optional[str]
    critic_weights: Optional[str]
    net_struc_actor: List[Union[Union[int,str],str]]
    net_struc_critic: List[Union[Union[int,str],str]]
    
    class Env:
        name: str
        wrappers: List[str]
        wrapper_kwargs: Dict[str,Any]
        max_episode_steps: int
        state_type: str
        env_kwargs: Dict[str,Any]
        info: str
        
    class Agent:
        pass
    
    def __init__(self, path: str) -> None:
        
        self.path = path
        
        # Load the YAML file into a dict
        self.config_dict = self._read_yaml(self.path)
        
        # Set the config file entries as attrs of the class
        self._set_attrs(self.config_dict)
        
    def _set_attrs(self, config_dict: Dict[str,Any]) -> None:
        
        for key,val in config_dict.items():
            if key in ["env","agent"]:
                self._set_subclass(key,val)
            else:
                setattr(self,key,val)
        
    def _read_yaml(self, file_path: str) -> Dict[str,Any]:
        
        with open(file_path) as yaml_file:
            file = yaml_file.read()
        
        return yaml.safe_load(file)
    
    def _set_subclass(self, key: str, d: Dict[str,Any]) -> None:
        
        if key == "env":
            setattr(self,key,self.Env)
            for key, val in d.items():
                setattr(self.Env,key,val)
        elif key == "agent":
            setattr(self,key,self.Agent)
            for key, val in d.items():
                setattr(self.Agent,key,val)