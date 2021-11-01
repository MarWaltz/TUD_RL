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
from current_algos.LSTM_SAC.lstm_sac_agent import *
from current_envs.envs import *
from current_envs.wrappers.gym_POMDP_wrapper import gym_POMDP_wrapper


def evaluate_policy(test_env, test_agent, c):
    test_agent.mode = "test"
    rets = []
    
    for _ in range(c["eval_episodes"]):
        
        # init history
        o_hist = np.zeros((c["history_length"], test_agent.obs_dim))
        a_hist = np.zeros((c["history_length"], test_agent.action_dim))
        hist_len = 0

        # get initial state
        o = test_env.reset()

        # potentially normalize it
        if c["input_norm"]:
            o = test_agent.inp_normalizer.normalize(o, mode="test")
        
        cur_ret = 0
        d = False
        eval_epi_steps = 0
        
        while not d:

            eval_epi_steps += 1

            # select action
            a = test_agent.select_action(o=o, o_hist=o_hist, a_hist=a_hist, hist_len=hist_len)
            
            # perform step
            o2, r, d, _ = test_env.step(a)

            # potentially normalize o2
            if c["input_norm"]:
                o2 = test_agent.inp_normalizer.normalize(o2, mode="test")

            # update history
            if hist_len == c["history_length"]:
                o_hist = np.roll(o_hist, shift = -1, axis = 0)
                o_hist[c["history_length"] - 1, :] = o

                a_hist = np.roll(a_hist, shift = -1, axis = 0)
                a_hist[c["history_length"] - 1, :] = a
            else:
                o_hist[hist_len] = o
                a_hist[hist_len] = a
                hist_len += 1
            
            # o becomes o2
            o = o2
            cur_ret += r

            # break option
            if eval_epi_steps == c["env"]["max_episode_steps"]:
                break
        
        # compute average return and append it
        rets.append(cur_ret)
    
    return rets


def train(c):
    """Main training loop."""

    # measure computation time
    start_time = time.time()
    
    # init envs
    env = gym.make(c["env"]["name"], **c["env"]["env_kwargs"])
    test_env = gym.make(c["env"]["name"], **c["env"]["env_kwargs"])

    # wrappers
    for wrapper in c["env"]["wrappers"]:
        env = eval(wrapper)(env)
        test_env = eval(wrapper)(test_env)
    
    # get state_shape
    if c["env"]["state_type"] == "image":
        raise NotImplementedError("Currently, image input is not available for continuous action spaces.")
    
    elif c["env"]["state_type"] == "feature":
        obs_dim = env.observation_space.shape[0]

    # seeding
    env.seed(c["seed"])
    test_env.seed(c["seed"])
    torch.manual_seed(c["seed"])
    np.random.seed(c["seed"])
    random.seed(c["seed"])

    # init agent
    agent = LSTM_SAC_Agent(mode             = "train",
                           action_dim       = env.action_space.shape[0], 
                           action_high      = env.action_space.high[0],
                           action_low       = env.action_space.low[0],
                           obs_dim          = obs_dim,
                           actor_weights    = c["actor_weights"],
                           critic_weights   = c["critic_weights"], 
                           input_norm       = c["input_norm"],
                           input_norm_prior = c["input_norm_prior"],
                           gamma            = c["gamma"],
                           tau              = c["tau"],
                           net_struc_actor  = c["net_struc_actor"],
                           net_struc_critic = c["net_struc_critic"],
                           lr_actor         = c["lr_actor"],
                           lr_critic        = c["lr_critic"],
                           lr_temperature   = c["lr_temperature"],
                           buffer_length    = c["buffer_length"],
                           grad_clip        = c["grad_clip"],
                           grad_rescale     = c["grad_rescale"],
                           act_start_step   = c["act_start_step"],
                           upd_start_step   = c["upd_start_step"],
                           upd_every        = c["upd_every"],
                           batch_size       = c["batch_size"],
                           history_length   = c["history_length"],
                           use_past_actions = c["use_past_actions"],
                           temp_tuning      = c["temp_tuning"],
                           temperature      = c["temperature"],
                           device           = c["device"])
    
    # init history
    o_hist = np.zeros((c["history_length"], agent.obs_dim))
    a_hist = np.zeros((c["history_length"], agent.action_dim))
    hist_len = 0

    # get initial state and normalize it
    o = env.reset()
    if c["input_norm"]:
        o = agent.inp_normalizer.normalize(o, mode="train")

    # init epi step counter and epi return
    epi_steps = 0
    epi_ret = 0
    
    # main loop    
    for total_steps in range(c["timesteps"]):

        epi_steps += 1
        
        # select action
        if total_steps < agent.act_start_step:
            a = np.random.uniform(low=agent.action_low, high=agent.action_high, size=agent.action_dim)
        else:
            a = agent.select_action(o=o, o_hist=o_hist, a_hist=a_hist, hist_len=hist_len)
        
        # perform step
        o2, r, d, _ = env.step(a)
        
        # Ignore "done" if it comes from hitting the time horizon of the environment
        d = False if epi_steps == c["env"]["max_episode_steps"] else d

        # potentially normalize o2
        if c["input_norm"]:
            o2 = agent.inp_normalizer.normalize(o2, mode="train")

        # add epi ret
        epi_ret += r
        
        # memorize
        agent.memorize(o, a, r, o2, d)
                
        # update history
        if hist_len == c["history_length"] :
            o_hist = np.roll(o_hist, shift = -1, axis = 0)
            o_hist[c["history_length"] - 1, :] = o

            a_hist = np.roll(a_hist, shift = -1, axis = 0)
            a_hist[c["history_length"] - 1, :] = a
        else:
            o_hist[hist_len] = o
            a_hist[hist_len] = a
            hist_len += 1

        # train
        if (total_steps >= c["upd_start_step"]) and (total_steps % c["upd_every"] == 0):
            for _ in range(c["upd_every"]):
                agent.train()

        # o becomes o2
        o = o2

        # end of episode handling
        if d or (epi_steps == c["env"]["max_episode_steps"]):

            # reset history
            o_hist = np.zeros((c["history_length"], agent.obs_dim))
            a_hist = np.zeros((c["history_length"], agent.action_dim))
            hist_len = 0

            # reset env to initial state and normalize it
            o = env.reset()
            if c["input_norm"]:
                o = agent.inp_normalizer.normalize(o, mode="train")
            
            # log episode return
            agent.logger.store(Epi_Ret=epi_ret)
            
            # reset epi steps and epi ret
            epi_steps = 0
            epi_ret = 0

        # end of epoch handling
        if (total_steps + 1) % c["epoch_length"]  == 0 and (total_steps + 1) > c["upd_start_step"]:

            epoch = (total_steps + 1) // c["epoch_length"]

            # evaluate agent with deterministic policy
            eval_ret = evaluate_policy(test_env=test_env, test_agent=copy.copy(agent), c=c)
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
            agent.logger.log_tabular("Q1_val", with_min_and_max=True)
            agent.logger.log_tabular("Q1_CurFE", with_min_and_max=False)
            agent.logger.log_tabular("Q1_ExtMemory", with_min_and_max=False)
            agent.logger.log_tabular("Q2_val", with_min_and_max=True)
            agent.logger.log_tabular("Q2_CurFE", with_min_and_max=False)
            agent.logger.log_tabular("Q2_ExtMemory", with_min_and_max=False)
            agent.logger.log_tabular("Critic_loss", average_only=True)
            agent.logger.log_tabular("Actor_loss", average_only=True)
            agent.logger.dump_tabular()

            # create evaluation plot based on current 'progress.txt'
            plot_from_progress(dir=agent.logger.output_dir, alg=agent.name, env_str=c["env"]["name"], info=c["env"]["plot_info"])

            # save weights
            torch.save(agent.actor.state_dict(), f"{agent.logger.output_dir}/{agent.name}_actor_weights.pth")
            torch.save(agent.critic.state_dict(), f"{agent.logger.output_dir}/{agent.name}_critic_weights.pth")
    
            # save input normalizer values 
            if c["input_norm"]:
                with open(f"{agent.logger.output_dir}/{agent.name}_inp_norm_values.pickle", "wb") as f:
                    pickle.dump(agent.inp_normalizer.get_for_save(), f)


if __name__ == "__main__":

    # get config and name of agent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="ski_lstm_sac.json")
    args = parser.parse_args()

    # read config file
    with open(__path__._path[0] + "/" + args.config_file) as f:
        c = json.load(f)

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
    train(c)
