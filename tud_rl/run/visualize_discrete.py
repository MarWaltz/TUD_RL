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
from tud_rl.envs.MountainCar import MountainCar
from tud_rl.envs.FossenEnv import *
from tud_rl.envs.PathFollower import PathFollower
from tud_rl.wrappers import get_wrapper
from tud_rl.configs.discrete_actions import __path__ as c_path


def visualize_policy(env: gym.Env, agent: _Agent, c: ConfigFile):

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
            a = agent.select_action(s)

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
        base_agent = agent_name[:-2] + "Agent"
    else:
        base_agent = agent_name + "Agent"

    # init agent
    agent_: type = getattr(agents, base_agent)  # Get agent class by name
    agent: _Agent = agent_(c, agent_name)  # Instantiate agent

    # visualization
    visualize_policy(env=env, agent=agent, c=c)


if __name__ == "__main__":

    # get config and name of agent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="pathfollower.yaml")
    parser.add_argument("--agent_name", type=str, default="SCDQN_b")
    #parser.add_argument("--dqn_weights", type=str, default="/home/niklaspaulig/Dropbox/TU Dresden/hpc/experiments/DQN_PathFollower-v0_5000-3°-nonstop_2022-03-28_27611/DQN_weights.pth")
    #parser.add_argument("--dqn_weights", type=str, default="/home/niklaspaulig/Dropbox/TU Dresden/experiments/DQN_PathFollower-v0_seiun_2022-03-30_27611/DQN_weights.pth")
    #parser.add_argument("--dqn_weights", type=str,default="/home/niklaspaulig/Dropbox/TU Dresden/hpc/experiments/SCDQN_b_PathFollower-v0_2000-5-seiun-norot-5th_2022-03-31_21442/SCDQN_b_weights.pth")
    #parser.add_argument("--dqn_weights", type=str,default="/home/niklaspaulig/Dropbox/TU Dresden/experiments/SCDQN_b_PathFollower-v0_2000-5-seiun-full-5th_2022-04-02_88341/SCDQN_b_weights.pth")
    parser.add_argument("--dqn_weights", type=str,
                        default="/home/neural/Dropbox/TU Dresden/hpc/experiments/SCDQN_b_PathFollower-v0_5000-k-pretrained_2022-04-05_91980/SCDQN_b_weights.pth")
    args = parser.parse_args()

    config_path = c_path[0] + "/" + args.config_file

    config = ConfigFile(config_path)

    # handle maximum episode steps
    if config.Env.max_episode_steps == -1:
        config.Env.max_episode_steps = np.inf

    test(config, args.agent_name, args.dqn_weights)
