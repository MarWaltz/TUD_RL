import argparse
import json
import random
import gym
import gym_minatar
import gym_pygame
import numpy as np
import torch

import tud_rl.agents.continuous as agents

from tud_rl.agents.base import _Agent
from tud_rl.common.configparser import ConfigFile
from tud_rl.envs.MountainCar import MountainCar
from tud_rl.wrappers import get_wrapper
from tud_rl.wrappers.MinAtar_wrapper import MinAtar_wrapper
from tud_rl.common.logging_plot import plot_from_progress
from tud_rl.configs.continuous_actions import __path__ as c_path


def visualize_policy(env: gym.Env, agent: _Agent, c: ConfigFile):

    for _ in range(c.eval_episodes):

        # LSTM: init history
        if "LSTM" in agent.name:
            s_hist = np.zeros((agent.history_length, agent.state_shape))
            a_hist = np.zeros((agent.history_length, agent.num_actions))
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
                a = agent.select_action(
                    s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
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
    # init envs
    env = gym.make(c.Env.name, **c.Env.env_kwargs)

    # wrappers
    for wrapper in c.Env.wrappers:
        wrapper_kwargs = c.Env.wrapper_kwargs[wrapper]
        env: gym.Env = get_wrapper(name=wrapper, env=env, **wrapper_kwargs)

    # get state_shape
    if c.Env.state_type == "image":
        raise NotImplementedError(
            "Currently, image input is not available for continuous action spaces.")

    elif c.Env.state_type == "feature":
        c.state_shape = env.observation_space.shape[0]

    # mode and action details
    c.mode = "test"
    c.num_actions = env.action_space.shape[0]
    c.action_high = env.action_space.high[0]
    c.action_low = env.action_space.low[0]
    # seeding
    env.seed(c.seed)
    torch.manual_seed(c.seed)
    np.random.seed(c.seed)
    random.seed(c.seed)

    # Agent prep
    if agent_name[-1].islower():
        base_agent = agent_name[:-2] + "Agent"
    else:
        base_agent = agent_name + "Agent"

    # Init agent
    agent_: type = getattr(agents, base_agent)  # Get agent class by name
    agent: _Agent = agent_(c, agent_name)  # Instantiate agent

    # visualization
    visualize_policy(env=env, agent=agent, c=c)


if __name__ == "__main__":

    # get config and name of agent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="ski_mdp.json")
    parser.add_argument("--agent_name", type=str, default="TD3")
    parser.add_argument("--actor_weights", type=str,
                        default="TD3_actor_weights.pth")
    parser.add_argument("--critic_weights", type=str,
                        default="TD3_critic_weights.pth")
    args = parser.parse_args()

    config_path = c_path[0] + "/" + args.config_file

    # read config file
    config = ConfigFile(config_path)

    # handle maximum episode steps
    if config.Env.max_episode_steps == -1:
        config.Env.max_episode_steps = np.inf

    test(config, args.agent_name, args.actor_weights, args.critic_weights)
