import argparse
import copy
import pickle
import random
import sys
import time

import gym
import numpy as np
#import pybulletgym
import torch
from current_algos.common.eval_plot import plot_from_progress
from current_algos.common.custom_envs import LCP_Environment
from current_algos.common.custom_envs import ObstacleAvoidance_Env
#from current_algos.common.POMDP_wrapper import POMDP_Wrapper
from current_algos.TD3.td3_agent import *

# training config
EVAL_EPISODES = 10      # number of episodes to average per evaluation


def visualize_policy(actor_weights, critic_weights):

    test_env = ObstacleAvoidance_Env()
    test_agent = TD3_Agent(mode      = "test",
                      action_dim     = test_env.action_space.shape[0], 
                      state_dim      = test_env.observation_space.shape[0], 
                      action_high    = test_env.action_space.high[0],
                      action_low     = test_env.action_space.low[0], 
                      actor_weights  = actor_weights, 
                      critic_weights = critic_weights)

    rets = []
    
    for _ in range(EVAL_EPISODES):
        # get initial state
        s = test_env.reset()

        # potentially normalize it
        if test_agent.input_norm:
            s = test_agent.inp_normalizer.normalize(s, mode="test")
        cur_ret = 0

        d = False
        
        while not d:
            # render
            test_env.render(agent_name=test_agent.name)

            # select action
            a = test_agent.select_action(s)
            
            # perform step
            s2, r, d, _ = test_env.step(a)

            # potentially normalize s2
            if test_agent.input_norm:
                s2 = test_agent.inp_normalizer.normalize(s2, mode="test")

            # s becomes s2
            s = s2
            cur_ret += r
        
        # compute average return and append it
        rets.append(cur_ret)
    
    return rets


if __name__ == "__main__":
    
    # set number of torch threads
    torch.set_num_threads(torch.get_num_threads())

    # run main loop
    visualize_policy(critic_weights="DDPG_Agent_critic_weights.pth", actor_weights="DDPG_Agent_actor_weights.pth")
