import argparse
import copy
import json
import pickle
import random
import time

import gym
import numpy as np
import torch
import tud_rl.agents.discrete as agents
from tud_rl.agents.base import _Agent
from tud_rl.common.configparser import ConfigFile
from tud_rl.configs.discrete_actions import __path__
from tud_rl.wrappers import get_wrapper


def visualize_policy(env, agent, c : ConfigFile):
    
    for _ in range(c.eval_episodes):

        # get initial state
        s = env.reset()

        # potentially normalize it
        if c.input_norm:
            s = agent.inp_normalizer.normalize(s, mode=agent.mode)

        cur_ret = 0
        d = False
        eval_epi_steps = 0

        while not d:

            eval_epi_steps += 1

            # render env
            env.render()

            # select action
            a = [agent.select_action(s_sub) for s_sub in s]
            
            # perform step
            s2, r, d, _ = env.step(a)

            # potentially normalize s2
            if c.input_norm:
                s2 = agent.inp_normalizer.normalize(s2, mode=agent.mode)

            # s becomes s2
            s = s2
            cur_ret += r

            # break option
            if eval_epi_steps == c.Env.max_episode_steps:
                break

        print(cur_ret)


def test(c : ConfigFile, agent_name, dqn_weights):
    # init env
    env = gym.make(c.Env.name, **c.Env.env_kwargs)

    # wrappers
    for wrapper in c.Env.wrappers:
        wrapper_kwargs = c.Env.wrapper_kwargs[wrapper]
        env: gym.Env = get_wrapper(name=wrapper, env=env, **wrapper_kwargs)

    # get state_shape
    if c.Env.state_type == "image":
        if "MinAtar" in c.Env.name:
            # careful, MinAtar constructs state as (height, width, in_channels), which is NOT aligned with PyTorch
            c.state_shape = (env.observation_space.shape[2], *env.observation_space.shape[0:2])
        else:
            c.state_shape = env.observation_space.shape[0]

    elif c.Env.state_type == "feature":
        c.state_shape = env.observation_space.shape[0]

    # mode and num actions
    c.mode = "test"
    c.num_actions = env.action_space.n

    # prior weights
    c.overwrite(dqn_weights=dqn_weights)

    # seeding
    env.seed(c.seed)
    torch.manual_seed(c.seed)
    np.random.seed(c.seed)
    random.seed(c.seed)

    if agent_name[-1].islower():
        agent_name_red = agent_name[:-2] + "Agent"
    else:
        agent_name_red = agent_name + "Agent"

    # init agent
    agent_ = getattr(agents, agent_name_red)  # Get agent class by name
    agent: _Agent = agent_(c, agent_name)  # Instantiate agent

    # visualization
    visualize_policy(env=env, agent=agent, c=c)


if __name__ == "__main__":

    # get config and name of agent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="FossenEnvRecDQNStarScene.json")
    parser.add_argument("--agent_name", type=str, default="RecDQN")
    parser.add_argument("--dqn_weights", type=str, default="RecDQN_weights.pth")
    args = parser.parse_args()

    # read config file
    c = ConfigFile(__path__._path[0] + "/" + args.config_file)

    # handle maximum episode steps
    c.max_episode_handler()

    # set number of torch threads
    torch.set_num_threads(torch.get_num_threads())

    test(c, args.agent_name, args.dqn_weights)
