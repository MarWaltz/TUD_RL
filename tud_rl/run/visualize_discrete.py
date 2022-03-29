import argparse
import copy
import json
import pickle
import random
import time

import gym
import gym_minatar
import gym_pygame
import numpy as np
import torch
from tud_rl.envs.MountainCar import MountainCar
from tud_rl.envs.FossenEnv import *
from tud_rl.envs.PathFollower import PathFollower
from tud_rl.wrappers.MinAtar_wrapper import MinAtar_wrapper
from tud_rl.agents.discrete.BootDQN import BootDQNAgent
from tud_rl.agents.discrete.DDQN import DDQNAgent
from tud_rl.agents.discrete.DQN import DQNAgent
from tud_rl.agents.discrete.EnsembleDQN import EnsembleDQNAgent
from tud_rl.agents.discrete.KEBootDQN import KEBootDQNAgent
from tud_rl.agents.discrete.MaxMinDQN import MaxMinDQNAgent
from tud_rl.agents.discrete.SCDQN import SCDQNAgent
from tud_rl.agents.discrete.RecDQN import RecDQNAgent
from tud_rl.common.logging_plot import plot_from_progress
from tud_rl.configs.discrete_actions import __path__


def visualize_policy(env, agent, c):
    
    for _ in range(c["eval_episodes"]):

        # get initial state
        s = env.reset()

        # potentially normalize it
        if c["input_norm"]:
            s = agent.inp_normalizer.normalize(s, mode=agent.mode)

        cur_ret = 0
        d = False
        eval_epi_steps = 0

        while not d:

            eval_epi_steps += 1

            # render env
            env.render()

            # select action
            a = agent.select_action(s)
            
            # perform step
            s2, r, d, _ = env.step(a)

            # potentially normalize s2
            if c["input_norm"]:
                s2 = agent.inp_normalizer.normalize(s2, mode=agent.mode)

            # s becomes s2
            s = s2
            cur_ret += r

            # break option
            if eval_epi_steps == c["env"]["max_episode_steps"]:
                break

        print(cur_ret)


def test(c, agent_name, dqn_weights):
    # init env
    env = gym.make(c["env"]["name"], **c["env"]["env_kwargs"])

    # wrappers
    for wrapper in c["env"]["wrappers"]:
        wrapper_kwargs = c["env"]["wrapper_kwargs"][wrapper]
        env = eval(wrapper)(env, **wrapper_kwargs)

    # get state_shape
    if c["env"]["state_type"] == "image":
        assert "MinAtar" in c["env"]["name"], "Only MinAtar-interface available for images."

        # careful, MinAtar constructs state as (height, width, in_channels), which is NOT aligned with PyTorch
        c["state_shape"] = (env.observation_space.shape[2], *env.observation_space.shape[0:2])
    
    elif c["env"]["state_type"] == "feature":
        c["state_shape"] = env.observation_space.shape[0]

    # mode and num actions
    c["mode"] = "test"
    c["num_actions"] = env.action_space.n

    # prior weights
    c["dqn_weights"] = dqn_weights

    # seeding
    env.seed(c["seed"])
    torch.manual_seed(c["seed"])
    np.random.seed(c["seed"])
    random.seed(c["seed"])

    # init agent
    if agent_name[-1].islower():
        agent = eval(agent_name[:-2] + "Agent")(c, agent_name)
    else:
        agent = eval(agent_name + "Agent")(c, agent_name)

    # visualization
    visualize_policy(env=env, agent=agent, c=c)


if __name__ == "__main__":

    # get config and name of agent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="pathfollower.json")
    parser.add_argument("--agent_name", type=str, default="DQN")
    #parser.add_argument("--dqn_weights", type=str, default="/home/neural/Dropbox/TU Dresden/experiments/DQN_PathFollower-v0_2000-abs-ang-rew-incr-3°_2022-03-25_27611/DQN_weights.pth")
    #parser.add_argument("--dqn_weights", type=str, default="/home/neural/Dropbox/TU Dresden/experiments/DQN_PathFollower-v0_2000-3°-10rps-none_2022-03-28_27611/DQN_weights.pth")
    parser.add_argument("--dqn_weights", type=str, default="/home/neural/Dropbox/TU Dresden/experiments/DQN_PathFollower-v0_2000-3°-10rps-none-2_2022-03-28_27611/DQN_weights2.pth")
    args = parser.parse_args()

    # read config file
    with open(__path__._path[0] + "/" + args.config_file) as f:
        c = json.load(f)

    # convert certain keys in integers
    for key in ["seed", "timesteps", "epoch_length", "eval_episodes", "eps_decay_steps", "tgt_update_freq",\
        "buffer_length", "act_start_step", "upd_start_step", "upd_every", "batch_size"]:
        c[key] = int(c[key])

    # handle maximum episode steps
    if c["env"]["max_episode_steps"] == -1:
        c["env"]["max_episode_steps"] = np.inf
    else:
        c["env"]["max_episode_steps"] = int(c["env"]["max_episode_steps"])

    # set number of torch threads
    torch.set_num_threads(torch.get_num_threads())

    test(c, args.agent_name, args.dqn_weights)
