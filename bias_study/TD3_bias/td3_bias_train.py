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

from current_algos.common.eval_plot import plot_from_progress
from current_algos.common.LCP_environment import LCP_Environment
from current_algos.common.POMDP_wrapper import POMDP_Wrapper
from bias_study.TD3_bias.td3_bias_agent import *

# training config
TIMESTEPS = 1000000     # overall number of training interaction steps
EPOCH_LENGTH = 5000     # number of time steps between evaluation/logging events
EVAL_EPISODES = 10      # number of episodes to average per evaluation

############## ADDITION ###########
# To effectively measure existing bias, we compute for each visited (s,a) pair in the evaluation episodes the
# Q_val and the later realized MC return. The difference between those quantities is the bias, which is normalized
# via the average MC under this policy to make things comparable.

# helper fnc
def get_MC_ret_from_rew(rews, gamma):
    """Returns for a given episode of rewards (list) the corresponding list of MC-returns under a specified discount factor."""

    MC = 0
    MC_list = []
    
    for r in reversed(rews):
        # compute one-step backup
        backup = r + gamma * MC
        
        # add to MCs
        MC_list.append(backup)
        
        # update MC
        MC = backup
    
    return list(reversed(MC_list))

def evaluate_policy(test_env, test_agent):
    test_agent.mode = "test"
    
    # list which stores final (undiscounted) sum of rewards of ALL episodes (final length: len(EVAL_EPISODES))
    rets = []

    # list which stores Q-vals of all (s,a) pairs of ALL episodes (final expected length: len(EVAL_EPISODES) * E(len(one test_episode)))
    Q_vals_all_eps = []

    # list which stores MC-vals of all (s,a) pairs of ALL episodes (final expected length: len(EVAL_EPISODES) * E(len(one test_episode)))
    MC_all_eps = []

    for _ in range(EVAL_EPISODES):
        # get initial state
        s = test_env.reset()

        # potentially normalize it
        if test_agent.input_norm:
            s = test_agent.inp_normalizer.normalize(s, mode="test")
        
        # list which stores rewards of ONE episode (final expected length: E(len(one test_episode)))
        rews_one_eps = []
        
        # init current return (for rets)
        cur_ret = 0

        d = False
        
        while not d:
            # select action
            a = test_agent.select_action(s)
            
            # calculate Q-values of the given (s,a) pair
            s_t = torch.tensor(s.astype(np.float32)).view(1, test_agent.state_dim)
            a_t = torch.tensor(a.astype(np.float32)).view(1, test_agent.action_dim)
            Q_vals_all_eps.append(test_agent.critic(s_t, a_t).item())

            # perform step
            s2, r, d, _ = test_env.step(a)

            # potentially normalize s2
            if test_agent.input_norm:
                s2 = test_agent.inp_normalizer.normalize(s2, mode="test")

            # s becomes s2
            s = s2

            # add r and append it 
            cur_ret += r
            rews_one_eps.append(r)
        
        # append final sum of reward
        rets.append(cur_ret)

        # transform reward list in MC return list and append it to overall MC returns
        MC_all_eps += get_MC_ret_from_rew(rews=rews_one_eps, gamma=test_agent.gamma)
    
    # compute normalized bias
    bias = [Q_vals_all_eps[i] - MC_all_eps[i] for i in range(len(Q_vals_all_eps))]
    norm_bias = [b/np.mean(MC_all_eps) for b in bias]
    
    # return final (undiscounted) reward sums and the mean, std of the bias for the given policy
    return rets, np.mean(norm_bias), np.std(norm_bias)

def train(env_str, pomdp=False, actor_weights=None, critic_weights=None, seed=0, device="cpu"):
    """Main training loop."""

    # measure computation time
    start_time = time.time()
    
    # init env
    if env_str == "LCP":
        env = LCP_Environment()
        test_env = LCP_Environment()
        max_episode_steps = env._max_episode_steps
    elif pomdp:
        env = POMDP_Wrapper(env_str, pomdp_type="remove_velocity")
        test_env = POMDP_Wrapper(env_str, pomdp_type="remove_velocity")
        max_episode_steps = gym.make(env_str)._max_episode_steps
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
    agent = TD3_Agent(mode           = "train",
                      action_dim     = env.action_space.shape[0], 
                      state_dim      = env.observation_space.shape[0], 
                      action_high    = env.action_space.high[0],
                      action_low     = env.action_space.low[0], 
                      actor_weights  = actor_weights, 
                      critic_weights = critic_weights, 
                      device         = device)
    
    # get initial state and normalize it
    s = env.reset()
    if agent.input_norm:
        s = agent.inp_normalizer.normalize(s, mode="train")

    # init epi step counter and epi return
    epi_steps = 0
    epi_ret = 0
    
    # main loop    
    for total_steps in range(TIMESTEPS):

        epi_steps += 1
        
        # select action
        if total_steps <= agent.act_start_step:
            a = np.random.uniform(low=agent.action_low, high=agent.action_high, size=agent.action_dim)
        else:
            a = agent.select_action(s)
        
        # perform step
        s2, r, d, _ = env.step(a)
        
        # Ignore "done" if it comes from hitting the time horizon of the environment
        d = False if epi_steps == max_episode_steps else d

        # potentially normalize s2
        if agent.input_norm:
            s2 = agent.inp_normalizer.normalize(s2, mode="train")

        # add epi ret
        epi_ret += r
        
        # memorize
        agent.memorize(s, a, r, s2, d)

        # train
        if (total_steps >= agent.upd_start_step) and (total_steps % agent.upd_every == 0):
            for _ in range(agent.upd_every):
                agent.train()

        # s becomes s2
        s = s2

        # end of episode handling
        if d or (epi_steps == max_episode_steps):
 
            # reset noise after episode
            agent.noise.reset()
            
            # reset to initial state and normalize it
            s = env.reset()
            if agent.input_norm:
                s = agent.inp_normalizer.normalize(s, mode="train")
            
            # log episode return
            agent.logger.store(Epi_Ret=epi_ret)
            
            # reset epi steps and epi ret
            epi_steps = 0
            epi_ret = 0

        # end of epoch handling
        if (total_steps + 1) % EPOCH_LENGTH == 0:

            epoch = (total_steps + 1) // EPOCH_LENGTH

            # evaluate agent with deterministic policy
            eval_rets, avg_bias, std_bias = evaluate_policy(test_env=test_env, test_agent=copy.copy(agent))
            for ret in eval_rets:
                agent.logger.store(Eval_ret=ret)

            # log and dump tabular
            agent.logger.log_tabular("Epoch", epoch)
            agent.logger.log_tabular("Timestep", total_steps)
            agent.logger.log_tabular("Runtime_in_h", (time.time() - start_time) / 3600)
            agent.logger.log_tabular("Epi_Ret", with_min_and_max=True)
            agent.logger.log_tabular("Eval_ret", with_min_and_max=True)
            if agent.double_critic:
                agent.logger.log_tabular("Q1_val", with_min_and_max=True)
                agent.logger.log_tabular("Q2_val", with_min_and_max=True)
            else:
                agent.logger.log_tabular("Q_val", with_min_and_max=True)
            agent.logger.log_tabular("Critic_loss", average_only=True)
            agent.logger.log_tabular("Actor_loss", average_only=True)
            agent.logger.log_tabular("Avg_bias", avg_bias)
            agent.logger.log_tabular("Std_bias", std_bias)
            agent.logger.dump_tabular()

            # create evaluation plot based on current 'progress.txt'
            plot_from_progress(dir=agent.logger.output_dir, alg="DDPG", env_str=env_str, info=None)

            # save weights
            torch.save(agent.actor.state_dict(), f"{agent.logger.output_dir}/{agent.name}_actor_weights.pth")
            torch.save(agent.critic.state_dict(), f"{agent.logger.output_dir}/{agent.name}_critic_weights.pth")
    
            # save input normalizer values 
            if agent.input_norm:
                with open(f"{agent.logger.output_dir}/{agent.name}_inp_norm_values.pickle", "wb") as f:
                    pickle.dump(agent.inp_normalizer.get_for_save(), f)
    
if __name__ == "__main__":
    
    # init and prepare argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_str", type=str, default="HalfCheetahPyBulletEnv-v0")
    args = parser.parse_args()
    
    # set number of torch threads
    torch.set_num_threads(torch.get_num_threads())

    # run main loop
    train(env_str=args.env_str, pomdp=False, critic_weights=None, actor_weights=None, seed=10, device="cpu")
