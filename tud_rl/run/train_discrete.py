import pickle
import random
import time
import gym
import gym_minatar
import gym_pygame
import numpy as np
import torch

import tud_rl.agents.discrete as agents

from tud_rl.agents.base import _Agent, _BootAgent
from tud_rl.common.configparser import ConfigFile
from tud_rl.common.logging_func import EpochLogger
from tud_rl.wrappers import get_wrapper
from tud_rl.common.logging_plot import plot_from_progress


def evaluate_policy(test_env: gym.Env, agent: _Agent, c: ConfigFile):

    # go greedy
    agent.mode = "test"

    rets = []

    for _ in range(c.eval_episodes):

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
            a = agent.select_action(s)

            # perform step
            s2, r, d, _ = test_env.step(a)

            # potentially normalize s2
            if c.input_norm:
                s2 = agent.inp_normalizer.normalize(s2, mode=agent.mode)

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


def train(c: ConfigFile, agent_name: str):
    """Main training loop."""

    # measure computation time
    start_time = time.time()

    # init envs
    env = gym.make(c.Env.name, **c.Env.env_kwargs)
    test_env = gym.make(c.Env.name, **c.Env.env_kwargs)

    # wrappers
    for wrapper in c.Env.wrappers:
        wrapper_kwargs = c.Env.wrapper_kwargs[wrapper]
        env: gym.Env = get_wrapper(name=wrapper, env=env, **wrapper_kwargs)
        test_env: gym.Env = get_wrapper(
            name=wrapper, env=test_env, **wrapper_kwargs)

    # get state_shape
    if c.Env.state_type == "image":
        if "MinAtar" in c.Env.name:
            # careful, MinAtar constructs state as (height, width, in_channels), which is NOT aligned with PyTorch
            c.state_shape = (
                env.observation_space.shape[2], *env.observation_space.shape[0:2])
        else:
            c.state_shape = env.observation_space.shape[0]

    elif c.Env.state_type == "feature":
        c.state_shape = env.observation_space.shape[0]

    # mode and num actions
    c.mode = "train"
    c.num_actions = env.action_space.n

    # seeding
    env.seed(c.seed)
    test_env.seed(c.seed)
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

    # Initialize logging
    agent.logger = EpochLogger(
        alg_str=agent.name,
        env_str=c.Env.name,
        info=c.Env.info,
        output_dir=c.output_dir if hasattr(c, "output_dir") else None)

    agent.logger.save_config({"agent_name": agent.name, **c.config_dict})
    agent.print_params(agent.n_params, mode=0)

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
            a = np.random.randint(
                low=0, high=agent.num_actions, size=1, dtype=int).item()
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

        # train
        if (total_steps >= c.upd_start_step) and (total_steps % c.upd_every == 0):
            for _ in range(c.upd_every):
                agent.train()

        # s becomes s2
        s = s2

        # end of episode handling
        if d or (epi_steps == c.Env.max_episode_steps):

            # reset active head for BootDQN
            if "Boot" in agent_name:
                agent: _BootAgent
                agent.reset_active_head()

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
            agent.logger.log_tabular(
                "Runtime_in_h", (time.time() - start_time) / 3600)
            agent.logger.log_tabular("Epi_Ret", with_min_and_max=True)
            agent.logger.log_tabular("Eval_ret", with_min_and_max=True)
            agent.logger.log_tabular("Q_val", with_min_and_max=True)
            agent.logger.log_tabular("Loss", average_only=True)
            agent.logger.dump_tabular()

            # create evaluation plot based on current 'progress.txt'
            plot_from_progress(
                dir=agent.logger.output_dir,
                alg=agent.name,
                env_str=c.Env.name,
                info=c.Env.info)

            # save weights
            save_weights(agent)

            # save input normalizer values
            if c.input_norm:
                with open(f"{agent.logger.output_dir}/{agent.name}_inp_norm_values.pickle", "wb") as f:
                    pickle.dump(agent.inp_normalizer.get_for_save(), f)


def save_weights(agent: _Agent) -> None:

    # Save weights for agents that require a single net
    if not any([word in agent.name for word in ["ACCDDQN", "Ensemble", "MaxMin"]]):
        torch.save(agent.DQN.state_dict(),
                   f"{agent.logger.output_dir}/{agent.name}_weights.pth")

    # Save both nets of the ACCDDQN
    if "ACCDDQN" in agent.name:
        torch.save(agent.DQN_A.state_dict(),
                   f"{agent.logger.output_dir}/{agent.name}_A_weights.pth")
        torch.save(agent.DQN_B.state_dict(),
                   f"{agent.logger.output_dir}/{agent.name}_B_weights.pth")

    if any(w in agent.name for w in ["Ensemble", "MaxMin"]):
        for idx, net in enumerate(agent.EnsembleDQN):
            torch.save(net.state_dict(),
                       f"{agent.logger.output_dir}/{agent.name}_weights_{idx}.pth")
