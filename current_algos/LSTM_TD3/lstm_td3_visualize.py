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
from current_algos.LSTM_TD3.lstm_td3_agent import *

# training config
EVAL_EPISODES = 10      # number of episodes to average per evaluation


def visualize_policy(hide_velocity, actor_weights, critic_weights):

    test_env = ObstacleAvoidance_Env(hide_velocity=hide_velocity)
    test_agent = LSTM_TD3_Agent(mode      = "test",
                      action_dim     = test_env.action_space.shape[0], 
                      obs_dim      = test_env.observation_space.shape[0], 
                      action_high    = test_env.action_space.high[0],
                      action_low     = test_env.action_space.low[0], 
                      actor_weights  = actor_weights, 
                      critic_weights = critic_weights)

    rets = []
    
    for _ in range(EVAL_EPISODES):

        # init history
        o_hist = np.zeros((test_agent.history_length, test_agent.obs_dim))
        a_hist = np.zeros((test_agent.history_length, test_agent.action_dim))
        hist_len = 0
        
        # get initial state
        o = test_env.reset()

        # potentially normalize it
        if test_agent.input_norm:
            o = test_agent.inp_normalizer.normalize(o, mode="test")
        cur_ret = 0

        d = False
        
        while not d:
            # render
            test_env.render(agent_name=test_agent.name)

            # select action
            a = test_agent.select_action(o=o, o_hist=o_hist, a_hist=a_hist, hist_len=hist_len)
            
            # perform step
            o2, r, d, _ = test_env.step(a)

            # potentially normalize o2
            if test_agent.input_norm:
                o2 = test_agent.inp_normalizer.normalize(o2, mode="test")

            # update history
            if hist_len == test_agent.history_length:
                o_hist = np.roll(o_hist, shift = -1, axis = 0)
                o_hist[test_agent.history_length - 1, :] = o

                a_hist = np.roll(a_hist, shift = -1, axis = 0)
                a_hist[test_agent.history_length - 1, :] = a
            else:
                o_hist[hist_len] = o
                a_hist[hist_len] = a
                hist_len += 1
            
            # o becomes o2
            o = o2
            cur_ret += r
        
        # compute average return and append it
        rets.append(cur_ret)
    
    return rets


if __name__ == "__main__":
    
    # set number of torch threads
    torch.set_num_threads(torch.get_num_threads())

    # run main loop
    visualize_policy(hide_velocity=True, critic_weights="LSTM_TD3_Agent_critic_weights.pth", actor_weights="LSTM_TD3_Agent_actor_weights.pth")
