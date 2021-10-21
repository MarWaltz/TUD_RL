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
from current_algos.common.envs.ObstacleAvoidance import ObstacleAvoidance_Env
from current_algos.common.envs.Ski import Ski_Env
#from current_algos.common.POMDP_wrapper import POMDP_Wrapper
from current_algos.LSTM_TD3.lstm_td3_agent import *

# training config
TIMESTEPS = 25000000     # overall number of training interaction steps
EPOCH_LENGTH = 5000     # number of time steps between evaluation/logging events
EVAL_EPISODES = 10      # number of episodes to average per evaluation

def evaluate_policy(test_env, test_agent):
    test_agent.mode = "test"
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

def train(env_str, POMDP_type="MDP", lr_critic=0.0001, history_length=2, use_past_actions=False, actor_weights=None, critic_weights=None, seed=0, device="cpu"):
    """Main training loop."""

    # measure computation time
    start_time = time.time()
    
    # init env
    if env_str == "LCP":
        env = ObstacleAvoidance_Env(POMDP_type=POMDP_type)
        test_env = ObstacleAvoidance_Env(POMDP_type=POMDP_type)
        max_episode_steps = env._max_episode_steps

    elif env_str == "Ski":
        env = Ski_Env(POMDP_type=POMDP_type)
        test_env = Ski_Env(POMDP_type=POMDP_type)
        max_episode_steps = env._max_episode_steps
    
    else:
        env = gym.make(env_str)
        test_env = gym.make(env_str)
        max_episode_steps = env._max_episode_steps
    
    # seeding
    env.seed(seed)
    test_env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # init agent
    agent = LSTM_TD3_Agent(mode             = "train",
                           action_dim       = env.action_space.shape[0], 
                           obs_dim          = env.observation_space.shape[0], 
                           action_high      = env.action_space.high[0],
                           action_low       = env.action_space.low[0], 
                           actor_weights    = actor_weights, 
                           critic_weights   = critic_weights, 
                           lr_critic        = lr_critic,
                           history_length   = history_length,
                           use_past_actions = use_past_actions,
                           device           = device)

    # init history
    o_hist = np.zeros((agent.history_length, agent.obs_dim))
    a_hist = np.zeros((agent.history_length, agent.action_dim))
    hist_len = 0
    
    # get initial state and normalize it
    o = env.reset()
    if agent.input_norm:
        o = agent.inp_normalizer.normalize(o, mode="train")

    # init epi step counter and epi return
    epi_steps = 0
    epi_ret = 0

    # main loop
    for total_steps in range(TIMESTEPS):

        epi_steps += 1
        
        # select action
        if total_steps < agent.act_start_step:
            a = np.random.uniform(low=agent.action_low, high=agent.action_high, size=agent.action_dim)
        else:
            a = agent.select_action(o=o, o_hist=o_hist, a_hist=a_hist, hist_len=hist_len)

        # perform step
        o2, r, d, _ = env.step(a)
        
        # Ignore "done" if it comes from hitting the time horizon of the environment
        d = False if epi_steps == max_episode_steps else d

        # potentially normalize o2
        if agent.input_norm:
            o2 = agent.inp_normalizer.normalize(o2, mode="train")
        
        # add epi ret
        epi_ret += r

        # memorize
        agent.memorize(o, a, r, o2, d)
        
        # update history
        if hist_len == agent.history_length:
            o_hist = np.roll(o_hist, shift = -1, axis = 0)
            o_hist[agent.history_length - 1, :] = o

            a_hist = np.roll(a_hist, shift = -1, axis = 0)
            a_hist[agent.history_length - 1, :] = a
        else:
            o_hist[hist_len] = o
            a_hist[hist_len] = a
            hist_len += 1

        # train
        if (total_steps >= agent.upd_start_step) and (total_steps % agent.upd_every == 0):
            for _ in range(agent.upd_every):
                agent.train()

        # o becomes o2
        o = o2

        # end of episode handling
        if d or (epi_steps == max_episode_steps):
 
            # reset noise after episode
            agent.noise.reset()
            
            # reset history
            o_hist = np.zeros((agent.history_length, agent.obs_dim))
            a_hist = np.zeros((agent.history_length, agent.action_dim))
            hist_len = 0
            
            # reset env to initial state and normalize it
            o = env.reset()
            if agent.input_norm:
                o = agent.inp_normalizer.normalize(o, mode="train")
            
            # log episode return
            agent.logger.store(Epi_Ret=epi_ret)
            
            # reset epi steps and epi ret
            epi_steps = 0
            epi_ret = 0
        
        # end of epoch handling
        if (total_steps + 1) % EPOCH_LENGTH == 0:

            epoch = (total_steps + 1) // EPOCH_LENGTH

            # evaluate agent with deterministic policy
            eval_ret = evaluate_policy(test_env=test_env, test_agent=copy.copy(agent))
            for ret in eval_ret:
                agent.logger.store(Eval_ret=ret)

            # log and dump tabular
            agent.logger.log_tabular("Epoch", epoch)
            agent.logger.log_tabular("Timestep", total_steps)
            agent.logger.log_tabular("Runtime_in_h", (time.time() - start_time) / 3600)
            agent.logger.log_tabular("Epi_Ret", with_min_and_max=True)
            agent.logger.log_tabular("Eval_ret", with_min_and_max=True)
            agent.logger.log_tabular("Actor_CurFE", with_min_and_max=False)
            agent.logger.log_tabular("Actor_ExtMemory", with_min_and_max=False)
            if agent.double_critic:
                agent.logger.log_tabular("Q1_val", with_min_and_max=True)
                agent.logger.log_tabular("Q1_CurFE", with_min_and_max=False)
                agent.logger.log_tabular("Q1_ExtMemory", with_min_and_max=False)
                agent.logger.log_tabular("Q2_val", with_min_and_max=True)
                agent.logger.log_tabular("Q2_CurFE", with_min_and_max=False)
                agent.logger.log_tabular("Q2_ExtMemory", with_min_and_max=False)
            else:
                agent.logger.log_tabular("Q_val", with_min_and_max=True)
                agent.logger.log_tabular("Critic_CurFE", with_min_and_max=False)
                agent.logger.log_tabular("Critic_ExtMemory", with_min_and_max=False)
            agent.logger.log_tabular("Critic_loss", average_only=True)
            agent.logger.log_tabular("Actor_loss", average_only=True)
            agent.logger.dump_tabular()

            # create evaluation plot based on current 'progress.txt'
            plot_from_progress(dir=agent.logger.output_dir, alg=agent.name, env_str=env_str, info=POMDP_type)

            # save weights
            torch.save(agent.actor.state_dict(), f"{agent.logger.output_dir}/{agent.name}_actor_weights.pth")
            torch.save(agent.critic.state_dict(), f"{agent.logger.output_dir}/{agent.name}_critic_weights.pth")
    
            # save input normalizer values 
            if agent.input_norm:
                with open(f"{agent.logger.output_dir}/{agent.name}_inp_norm_values.pickle", "wb") as f:
                    pickle.dump(agent.inp_normalizer.get_for_save(), f)

if __name__ == "__main__":

    # helper function for parser
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # init and prepare argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_str", type=str, default="LCP")
    parser.add_argument("--POMDP_type", type=str, default="MDP")
    parser.add_argument("--lr_critic", type=float, default=0.0001)
    parser.add_argument("--history_length", type=int, default=2)
    parser.add_argument("--use_past_actions", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    
    # set number of torch threads
    torch.set_num_threads(torch.get_num_threads())

    # run main loop
    train(env_str=args.env_str, POMDP_type=args.POMDP_type, lr_critic=args.lr_critic, history_length=args.history_length,
          use_past_actions=args.use_past_actions, critic_weights=None, actor_weights=None, seed=args.seed, device="cpu")
