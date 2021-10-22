import argparse
import copy
import pickle
import random
import time

import gym
import gym_minatar
import numpy as np

import torch
from current_algos.common.eval_plot import plot_from_progress
from current_algos.CNN_based.Bootstrapped_DQN_MinAtar.bootstrapped_dqn_agent_MinAtar import *

# training config
TIMESTEPS = 5000000     # overall number of training interaction steps
EPOCH_LENGTH = 5000     # number of time steps between evaluation/logging events
EVAL_EPISODES = 1      # number of episodes to average per evaluation


def visualize_policy(env_str, dqn_weights):
    test_env = gym.make(env_str)

    # careful, MinAtar constructs state as (height, width, in_channels), which is NOT aligned with PyTorch
    state_shape = (test_env.observation_space.shape[2], *test_env.observation_space.shape[0:2])

    test_agent = CNN_Bootstrapped_DQN_Agent(mode         = "test",
                                            num_actions  = test_env.action_space.n, 
                                            state_shape  = state_shape,
                                            kernel       = "gaussian_cdf",
                                            dqn_weights  = dqn_weights)

    for _ in range(EVAL_EPISODES):

        # get initial state
        s = test_env.reset()

        # potentially normalize it
        if test_agent.input_norm:
            s = test_agent.inp_normalizer.normalize(s, mode="test")

        # change s to be of shape (in_channels, height, width) instead of (height, width, in_channels)
        s = np.moveaxis(s, -1, 0)

        cur_ret = 0
        d = False

        if test_env.game_name == "seaquest":
            eval_epi_steps = 0
        
        while not d:
            
            test_env.render()

            if test_env.game_name == "seaquest":
                eval_epi_steps += 1

            # select action
            a = test_agent.select_action(s, active_head=None)
            
            # perform step
            s2, r, d, _ = test_env.step(a)

            # potentially normalize s2
            if test_agent.input_norm:
                s2 = test_agent.inp_normalizer.normalize(s2, mode="test")

            # change s2 to be of shape (in_channels, height, width) instead of (height, width, in_channels)
            s2 = np.moveaxis(s2, -1, 0)

            # s becomes s2
            s = s2
            cur_ret += r

            # break option for seaquest-env
            if test_env.game_name == "seaquest" and eval_epi_steps == 100000:
                break    

visualize_policy(env_str="Breakout-MinAtar-v0", dqn_weights="CNN_OurBootDQN_Agent_gaussian_cdf_DQN_weights.pth")
