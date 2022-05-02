import argparse
import random
import gym
import gym_minatar
import gym_pygame
import numpy as np
import torch

import tud_rl.agents.discrete as agents

from tud_rl.agents.base import _Agent
from tud_rl.common.configparser import ConfigFile
from tud_rl.wrappers import get_wrapper
from tud_rl.configs.discrete_actions import __path__ as c_path


def visualize_policy(env: gym.Env, agent: _Agent, c: ConfigFile):

    for _ in range(c.eval_episodes):

        # LSTM: init history
        if "LSTM" in agent.name:
            s_hist = np.zeros((agent.history_length, agent.state_shape))
            a_hist = np.zeros((agent.history_length, 1), dtype=np.int64)
            hist_len = 0

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
            if "LSTM" in agent.name:
                a = agent.select_action(s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
            else:
                a = agent.select_action(s)

            # perform step
            s2, r, d, _ = env.step(a)

            # potentially normalize s2
            if c.input_norm:
                s2 = agent.inp_normalizer.normalize(s2, mode=agent.mode)

            # LSTM: update history
            if "LSTM" in agent.name:
                if hist_len == agent.history_length:
                    s_hist = np.roll(s_hist, shift=-1, axis=0)
                    s_hist[agent.history_length - 1, :] = s

                    a_hist = np.roll(a_hist, shift=-1, axis=0)
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

        print(cur_ret)


def test(c: ConfigFile, agent_name: str):
    # init env
    env = gym.make(c.Env.name, **c.Env.env_kwargs)

    # wrappers
    for wrapper in c.Env.wrappers:
        wrapper_kwargs = c.Env.wrappers[wrapper]
        env: gym.Env = get_wrapper(name=wrapper, env=env, **wrapper_kwargs)

    # get state_shape
    if c.Env.state_type == "image":
        assert "MinAtar" in c.Env.name, "Only MinAtar-interface available for images."

        # careful, MinAtar constructs state as
        # (height, width, in_channels), which is NOT aligned with PyTorch
        c.state_shape = (env.observation_space.shape[2],
                         *env.observation_space.shape[0:2])

    elif c.Env.state_type == "feature":
        c.state_shape = env.observation_space.shape[0]

    # mode and num actions
    c.mode = "test"
    c.num_actions = env.action_space.n

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
