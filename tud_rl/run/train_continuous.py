import argparse
import pickle
import random
import time

import gym
import gym_minatar
import gym_pygame
import numpy as np
import torch
from tud_rl.agents.base import AbstractAgent
from tud_rl.envs.MountainCar import MountainCar
from tud_rl.wrappers import get_wrapper
from tud_rl.agents.continuous import *
from tud_rl.common.logging_plot import plot_from_progress
from tud_rl.configs.continuous_actions import __path__


def evaluate_policy(test_env: gym.Env, agent: AbstractAgent, c: Configfile):

    # go greedy
    agent.mode = "test"
    
    rets = []
    
    for _ in range(c.eval_episodes):

        # LSTM: init history
        if "LSTM" in agent.name:
            s_hist = np.zeros((agent.history_length, agent.state_shape))
            a_hist = np.zeros((agent.history_length, agent.num_actions))
            hist_len = 0

        # get initial state
        s = test_env.reset()
        if c.input_norm:
            s = agent.inp_normalizer.normalize(s, mode=agent.mode)

        cur_ret = 0
        d = False
        eval_epi_steps = 0

        while not d:

            eval_epi_steps += 1

            # select action
            if "LSTM" in agent.name:
                a = agent.select_action(s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
            else:
                a = agent.select_action(s)
            
            # perform step
            s2, r, d, _ = test_env.step(a)

            # potentially normalize s2
            if c.input_norm:
                s2 = agent.inp_normalizer.normalize(s2, mode=agent.mode)

            # LSTM: update history
            if "LSTM" in agent.name:
                if hist_len == agent.history_length:
                    s_hist = np.roll(s_hist, shift = -1, axis = 0)
                    s_hist[agent.history_length - 1, :] = s

                    a_hist = np.roll(a_hist, shift = -1, axis = 0)
                    a_hist[agent.history_length - 1, :] = a
                else:
                    s_hist[hist_len] = s
                    a_hist[hist_len] = a
                    hist_len += 1

            # s becomes s2
            s = s2
            cur_ret += r

            # break option
            if eval_epi_steps == c.Env.max_episode_steps:
                break
        
        # append return
        rets.append(cur_ret)
    
    # continue training
    agent.mode = "train"

    return rets


def train(c: Configfile, agent_name: str):
    """Main training loop."""

    # measure computation time
    start_time = time.time()
    
    # init envs
    env: gym.Env = gym.make(c.Env.name, **c.Env.env_kwargs)
    test_env: gym.Env = gym.make(c.Env.name, **c.Env.env_kwargs)

    # wrappers
    for wrapper in c.Env.wrappers:
        wrapper_kwargs = c["env"]["wrapper_kwargs"][wrapper]
        env: gym.Env = get_wrapper(name=wrapper, env=env, **wrapper_kwargs)
        test_env: gym.Env = get_wrapper(wrapper,env=test_env, **wrapper_kwargs)

    # get state_shape
    if c.Env.state_type == "image":
        raise NotImplementedError("Currently, image input is not available for continuous action spaces.")
    
    elif c.Env.state_type == "feature":
        c.state_shape = env.observation_space.shape[0]

    # mode and action details
    c.mode = "train"
    c.num_actions = env.action_space.shape[0]
    c.action_high = env.action_space.high[0]
    c.action_low  = env.action_space.low[0]

    # seeding
    env.seed(c.seed)
    test_env.seed(c.seed)
    torch.manual_seed(c.seed)
    np.random.seed(c.seed)
    random.seed(c.seed)

    # init agent
    if agent_name[-1].islower():
        agent: AbstractAgent = eval(agent_name[:-2] + "Agent")(c, agent_name)
    else:
        agent: AbstractAgent = eval(agent_name + "Agent")(c, agent_name)

    # LSTM: init history
    if "LSTM" in agent.name:
        s_hist = np.zeros((agent.history_length, agent.state_shape))
        a_hist = np.zeros((agent.history_length, agent.num_actions))
        hist_len = 0

    # get initial state and normalize it
    s = env.reset()
    if c.input_norm:
        s = agent.inp_normalizer.normalize(s, mode=agent.mode)

    # init epi step counter and epi return
    epi_steps = 0
    epi_ret = 0
    
    # main loop    
    for total_steps in range(c.timesteps):

        epi_steps += 1
        
        # select action
        if total_steps < c.act_start_step:
            a = np.random.uniform(low=agent.action_low, high=agent.action_high, size=agent.num_actions)
        else:
            if "LSTM" in agent.name:
                a = agent.select_action(s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
            else:
                a = agent.select_action(s)
        
        # perform step
        s2, r, d, _ = env.step(a)
        
        # Ignore "done" if it comes from hitting the time horizon of the environment
        d = False if epi_steps == c.Env.max_episode_steps else d

        # potentially normalize s2
        if c.input_norm:
            s2 = agent.inp_normalizer.normalize(s2, mode=agent.mode)

        # add epi ret
        epi_ret += r
        
        # memorize
        agent.memorize(s, a, r, s2, d)

        # LSTM: update history
        if "LSTM" in agent.name:
            if hist_len == agent.history_length:
                s_hist = np.roll(s_hist, shift = -1, axis = 0)
                s_hist[agent.history_length - 1, :] = s

                a_hist = np.roll(a_hist, shift = -1, axis = 0)
                a_hist[agent.history_length - 1, :] = a
            else:
                s_hist[hist_len] = s
                a_hist[hist_len] = a
                hist_len += 1

        # train
        if (total_steps >= c.upd_start_step) and (total_steps % c.upd_every == 0):
            for _ in range(c.upd_every):
                agent.train()

        # s becomes s2
        s = s2

        # end of episode handling
        if d or (epi_steps == c.Env.max_episode_steps):
 
            # reset noise after episode
            if hasattr(agent, "noise"):
                agent.noise.reset()

            # LSTM: reset history
            if "LSTM" in agent.name:
                s_hist = np.zeros((agent.history_length, agent.state_shape))
                a_hist = np.zeros((agent.history_length, agent.num_actions))
                hist_len = 0

            # reset to initial state and normalize it
            s = env.reset()
            if c.input_norm:
                s = agent.inp_normalizer.normalize(s, mode=agent.mode)
            
            # log episode return
            agent.logger.store(Epi_Ret=epi_ret)
            
            # reset epi steps and epi ret
            epi_steps = 0
            epi_ret = 0

        # end of epoch handling
        if (total_steps + 1) % c.epoch_length == 0 and (total_steps + 1) > c.upd_start_step:

            epoch = (total_steps + 1) // c.epoch_length

            # evaluate agent with deterministic policy
            eval_ret = evaluate_policy(test_env=test_env, agent=agent, c=c)
            for ret in eval_ret:
                agent.logger.store(Eval_ret=ret)

            # log and dump tabular
            agent.logger.log_tabular("Epoch", epoch)
            agent.logger.log_tabular("Timestep", total_steps)
            agent.logger.log_tabular("Runtime_in_h", (time.time() - start_time) / 3600)
            agent.logger.log_tabular("Epi_Ret", with_min_and_max=True)
            agent.logger.log_tabular("Eval_ret", with_min_and_max=True)
            agent.logger.log_tabular("Q_val", with_min_and_max=True)
            agent.logger.log_tabular("Critic_loss", average_only=True)
            agent.logger.log_tabular("Actor_loss", average_only=True)

            if "LSTM" in agent.name:
                agent.logger.log_tabular("Actor_CurFE", with_min_and_max=False)
                agent.logger.log_tabular("Actor_ExtMemory", with_min_and_max=False)
                agent.logger.log_tabular("Critic_CurFE", with_min_and_max=False)
                agent.logger.log_tabular("Critic_ExtMemory", with_min_and_max=False)

            agent.logger.dump_tabular()

            # create evaluation plot based on current 'progress.txt'
            plot_from_progress(
                dir=agent.logger.output_dir, 
                alg=agent.name, env_str=c.Env.name, 
                info=c.Env.info)

            # save weights
            torch.save(agent.actor.state_dict(), f"{agent.logger.output_dir}/{agent.name}_actor_weights.pth")
            torch.save(agent.critic.state_dict(), f"{agent.logger.output_dir}/{agent.name}_critic_weights.pth")

            # save input normalizer values 
            if c["input_norm"]:
                with open(f"{agent.logger.output_dir}/{agent.name}_inp_norm_values.pickle", "wb") as f:
                    pickle.dump(agent.inp_normalizer.get_for_save(), f)
