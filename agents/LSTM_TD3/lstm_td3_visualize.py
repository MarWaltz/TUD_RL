import argparse
import copy
import pickle
import random
import sys
import time

import gym
import numpy as np
import pybulletgym
import torch
from common.eval_plot import plot_from_progress
from configs.continuous_actions import __path__
from agents.LSTM_TD3.lstm_td3_agent import *
from current_envs.envs import *
from current_envs.wrappers.gym_POMDP_wrapper import gym_POMDP_wrapper



def visualize_policy(c, agent_name, actor_weights, critic_weights):
    
    # measure computation time
    start_time = time.time()
    
    # init envs
    env = gym.make(c["env"]["name"], **c["env"]["env_kwargs"])
    test_env = gym.make(c["env"]["name"], **c["env"]["env_kwargs"])

    # wrappers
    for wrapper in c["env"]["wrappers"]:
        wrapper_kwargs = c["env"]["wrapper_kwargs"][wrapper]
        env = eval(wrapper)(env, **wrapper_kwargs)
        test_env = eval(wrapper)(test_env, **wrapper_kwargs)
    
    # get state_shape
    if c["env"]["state_type"] == "image":
        raise NotImplementedError("Currently, image input is not available for continuous action spaces.")
    
    elif c["env"]["state_type"] == "feature":
        obs_dim = env.observation_space.shape[0]

    # seeding
    print(c["seed"])
    env.seed(c["seed"])
    test_env.seed(c["seed"])
    torch.manual_seed(c["seed"])
    np.random.seed(c["seed"])
    random.seed(c["seed"])

    # init agent
    test_agent = LSTM_TD3_Agent(mode             = "test",
                           action_dim       = env.action_space.shape[0], 
                           action_high      = env.action_space.high[0],
                           action_low       = env.action_space.low[0],
                           obs_dim          = obs_dim,
                           actor_weights    = actor_weights,
                           critic_weights   = critic_weights, 
                           input_norm       = c["input_norm"],
                           input_norm_prior = c["input_norm_prior"],
                           double_critic    = c["agent"][agent_name]["double_critic"],
                           tgt_pol_smooth   = c["agent"][agent_name]["tgt_pol_smooth"],
                           tgt_noise        = c["tgt_noise"],
                           tgt_noise_clip   = c["tgt_noise_clip"],
                           pol_upd_delay    = c["agent"][agent_name]["pol_upd_delay"],
                           gamma            = c["gamma"],
                           tau              = c["tau"],
                           net_struc_actor  = c["net_struc_actor"],
                           net_struc_critic = c["net_struc_critic"],
                           lr_actor         = c["lr_actor"],
                           lr_critic        = c["lr_critic"],
                           buffer_length    = c["buffer_length"],
                           grad_clip        = c["grad_clip"],
                           grad_rescale     = c["grad_rescale"],
                           act_start_step   = c["act_start_step"],
                           upd_start_step   = c["upd_start_step"],
                           upd_every        = c["upd_every"],
                           batch_size       = c["batch_size"],
                           history_length   = c["history_length"],
                           use_past_actions = c["use_past_actions"],
                           device           = c["device"],
                           env_str          = c["env"]["name"],
                           info             = c["env"]["info"],
                           seed             = c["seed"])

    rets = []
    
    for _ in range(1):

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
        epi_steps = 0
        d = False
        
        while not d:

            epi_steps += 1
            
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

            #if epi_steps == 600:
            #    d = True
        
        # compute average return and append it
        rets.append(cur_ret)
    
    return rets


if __name__ == "__main__":
    
    # get config and name of agent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="ski_lstm_td3_mdp.json")
    parser.add_argument("--agent_name", type=str, default="lstm_td3")
    parser.add_argument("--seed", type=int, default=65330)#random.randint(0, 100000))
    args = parser.parse_args()

    # read config file
    with open(__path__._path[0] + "/" + args.config_file) as f:
        c = json.load(f)

    # potentially overwrite seed
    if args.seed is not None:
        c["seed"] = args.seed

    # convert certain keys in integers
    for key in ["seed", "timesteps", "epoch_length", "eval_episodes", "buffer_length", "act_start_step",\
         "upd_start_step", "upd_every", "batch_size", "history_length"]:
        c[key] = int(c[key])

    # handle maximum episode steps
    if c["env"]["max_episode_steps"] == -1:
        c["env"]["max_episode_steps"] = np.inf
    else:
        c["env"]["max_episode_steps"] = int(c["env"]["max_episode_steps"])

    # set number of torch threads
    torch.set_num_threads(torch.get_num_threads())

    # run main loop
    visualize_policy(c, args.agent_name, actor_weights="lstm_td3_agent_actor_weights.pth", critic_weights="lstm_td3_agent_critic_weights.pth")
