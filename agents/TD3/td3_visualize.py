import argparse

import gym
import numpy as np
#import pybulletgym
import torch
from configs.continuous_actions import __path__
from agents.TD3.td3_agent import *
from current_envs.envs import *
from current_envs.wrappers.gym_POMDP_wrapper import gym_POMDP_wrapper


def visualize_policy(c, agent_name, actor_weights, critic_weights):

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
        state_dim = env.observation_space.shape[0]

    # init agent
    agent = TD3_Agent(mode             = "test",
                      action_dim       = env.action_space.shape[0], 
                      action_high      = env.action_space.high[0],
                      action_low       = env.action_space.low[0],
                      state_dim        = state_dim,
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
                      device           = c["device"])

    rets = []
    
    for _ in range(10):
        # get initial state
        s = test_env.reset()

        # potentially normalize it
        if agent.input_norm:
            s = agent.inp_normalizer.normalize(s, mode="test")
        cur_ret = 0

        d = False
        eval_epi_steps = 0
        
        while not d:

            eval_epi_steps += 1
            
            # render
            test_env.render(agent_name=agent.name)

            # select action
            a = agent.select_action(s)
            
            # perform step
            s2, r, d, _ = test_env.step(a)

            # potentially normalize s2
            if agent.input_norm:
                s2 = agent.inp_normalizer.normalize(s2, mode="test")

            # s becomes s2
            s = s2
            cur_ret += r
            
            # break option
            if eval_epi_steps == c["env"]["max_episode_steps"]:
                break
        
        # compute average return and append it
        rets.append(cur_ret)
    
    return rets


if __name__ == "__main__":

 # get config and name of agent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="oa_td3_mdp.json")
    parser.add_argument("--agent_name", type=str, default="td3")
    args = parser.parse_args()

    # read config file
    with open(__path__._path[0] + "/" + args.config_file) as f:
        c = json.load(f)

    # convert certain keys in integers
    for key in ["seed", "timesteps", "epoch_length", "eval_episodes", "buffer_length", "act_start_step",\
         "upd_start_step", "upd_every", "batch_size"]:
        c[key] = int(c[key])

    # handle maximum episode steps
    if c["env"]["max_episode_steps"] == -1:
        c["env"]["max_episode_steps"] = np.inf
    else:
        c["env"]["max_episode_steps"] = int(c["env"]["max_episode_steps"])

    # set number of torch threads
    torch.set_num_threads(torch.get_num_threads())

    # run main loop
    visualize_policy(c=c, agent_name=args.agent_name, actor_weights="td3_agent_actor_weights.pth", critic_weights="td3_agent_critic_weights.pth")
