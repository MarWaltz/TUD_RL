import csv
import pickle
import random
import shutil
import time

import gym
import numpy as np
import pandas as pd
import torch

import tud_rl.agents.discrete as agents
from tud_rl import logger
from tud_rl.agents._discrete.BootDQN import BootDQNAgent
from tud_rl.agents.base import _Agent
from tud_rl.common.configparser import ConfigFile
from tud_rl.common.logging_func import EpochLogger
from tud_rl.common.logging_plot import plot_from_progress
from tud_rl.wrappers import get_wrapper


def evaluate_policy(test_env: gym.Env, agent: _Agent, c: ConfigFile):

    # go greedy
    agent.mode = "test"

    rets = []

    for _ in range(c.eval_episodes):

        # LSTM: init history
        if agent.needs_history:
            s_hist = np.zeros((agent.history_length, agent.state_shape))
            a_hist = np.zeros((agent.history_length, 1), dtype=np.int64)
            hist_len = 0

        # get initial state
        s = test_env.reset()

        cur_ret = np.zeros((agent.N_agents, 1)) if agent.is_multi else 0.0
        d = False
        eval_epi_steps = 0

        while not d:

            eval_epi_steps += 1

            # select action
            if agent.needs_history:
                a = agent.select_action(s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
            else:
                a = agent.select_action(s)

            # perform step
            s2, r, d, _ = test_env.step(a)

            # LSTM: update history
            if agent.needs_history:
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
    env: gym.Env = gym.make(c.Env.name, **c.Env.env_kwargs)
    test_env: gym.Env = gym.make(c.Env.name, **c.Env.env_kwargs)

    # wrappers
    for wrapper in c.Env.wrappers:
        wrapper_kwargs = c.Env.wrapper_kwargs[wrapper]
        env: gym.Env = get_wrapper(name=wrapper, env=env, **wrapper_kwargs)
        test_env: gym.Env = get_wrapper(name=wrapper, env=test_env, **wrapper_kwargs)

    # get state shape
    if c.Env.state_type == "image":
        c.state_shape = env.observation_space.shape
    elif c.Env.state_type == "feature":
        c.state_shape = env.observation_space.shape[0]

    # mode and action details
    c.mode = "train"
    c.num_actions = env.action_space.n

    # seeding
    env.seed(c.seed)
    test_env.seed(c.seed)
    torch.manual_seed(c.seed)
    torch.cuda.manual_seed(c.seed)
    np.random.seed(c.seed)
    random.seed(c.seed)

    if agent_name[-1].islower():
        agent_name_red = agent_name[:-2] + "Agent"
    else:
        agent_name_red = agent_name + "Agent"

    # init agent
    agent_ = getattr(agents, agent_name_red)  # get agent class by name
    agent: _Agent = agent_(c, agent_name)  # instantiate agent

    # possibly load replay buffer for continued training
    if hasattr(c, "prior_buffer"):
        if c.prior_buffer is not None:
            with open(c.prior_buffer, "rb") as f:
                agent.replay_buffer = pickle.load(f)

    # initialize logging
    agent.logger = EpochLogger(alg_str    = agent.name,
                               seed       = c.seed,
                               env_str    = c.Env.name,
                               info       = c.Env.info,
                               output_dir = c.output_dir if hasattr(c, "output_dir") else None)

    agent.logger.save_config({"agent_name": agent.name, **c.config_dict})
    agent.print_params(agent.n_params, case=0)

    # save env-file for traceability
    try:
        entry_point = vars(gym.envs.registry[c.Env.name])["entry_point"][12:]
        shutil.copy2(src="tud_rl/envs/_envs/" + entry_point + ".py", dst=agent.logger.output_dir)
    except:
        logger.warning(
            f"Could not find the env file. Make sure that the file name matches the class name. Skipping..."
        )

    # LSTM: init history
    if agent.needs_history:
        s_hist = np.zeros((agent.history_length, agent.state_shape))
        a_hist = np.zeros((agent.history_length, 1), dtype=np.int64)
        hist_len = 0

    # get initial state
    s = env.reset()

    # init epi step counter and epi return
    epi_steps = 0
    epi_ret = np.zeros((agent.N_agents, 1)) if agent.is_multi else 0.0

    # main loop
    for total_steps in range(c.timesteps):

        epi_steps += 1

        # select action
        if total_steps < c.act_start_step:
            if agent.is_multi:
                a = np.random.randint(low=0, high=agent.num_actions, size=agent.N_agents, dtype=int)
            else:
                a = np.random.randint(low=0, high=agent.num_actions, size=1, dtype=int).item()
        else:
            if agent.needs_history:
                a = agent.select_action(s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
            else:
                a = agent.select_action(s)

        # perform step
        s2, r, d, _ = env.step(a)

        # Ignore "done" if it comes from hitting the time horizon of the environment
        d = False if epi_steps == c.Env.max_episode_steps else d

        # add epi ret
        epi_ret += r

        # memorize
        agent.memorize(s, a, r, s2, d)

        # LSTM: update history
        if agent.needs_history:
            if hist_len == agent.history_length:
                s_hist = np.roll(s_hist, shift=-1, axis=0)
                s_hist[agent.history_length - 1, :] = s

                a_hist = np.roll(a_hist, shift=-1, axis=0)
                a_hist[agent.history_length - 1, :] = a
            else:
                s_hist[hist_len] = s
                a_hist[hist_len] = a
                hist_len += 1

        # train
        if (total_steps >= c.upd_start_step) and (total_steps % c.upd_every == 0):
            agent.train()

        # s becomes s2
        s = s2

        # end of episode handling
        if d or (epi_steps == c.Env.max_episode_steps):

            # reset active head for BootDQN and its modifications
            if isinstance(agent, BootDQNAgent):
                agent.reset_active_head()

            # LSTM: reset history
            if agent.needs_history:
                s_hist = np.zeros((agent.history_length, agent.state_shape))
                a_hist = np.zeros((agent.history_length, 1), dtype=np.int64)
                hist_len = 0

            # reset to initial state
            s = env.reset()

            # log episode return
            if agent.is_multi:
                for i in range(agent.N_agents):
                    agent.logger.store(**{f"Epi_Ret_{i}" : epi_ret[i].item()})
            else:
                agent.logger.store(Epi_Ret=epi_ret)

            # reset epi steps and epi ret
            epi_steps = 0
            epi_ret = np.zeros((agent.N_agents, 1)) if agent.is_multi else 0.0

        # end of epoch handling
        if (total_steps + 1) % c.epoch_length == 0 and (total_steps + 1) > c.upd_start_step:

            epoch = (total_steps + 1) // c.epoch_length

            # evaluate agent with deterministic policy
            eval_ret = evaluate_policy(test_env=test_env, agent=agent, c=c)

            if agent.is_multi:
                for ret_list in eval_ret:
                    for i in range(agent.N_agents):
                        agent.logger.store(**{f"Eval_ret_{i}" : ret_list[i].item()})
            else:
                for ret in eval_ret:
                    agent.logger.store(Eval_ret=ret)

            # log and dump tabular
            agent.logger.log_tabular("Epoch", epoch)
            agent.logger.log_tabular("Timestep", total_steps)
            agent.logger.log_tabular("Runtime_in_h", (time.time() - start_time) / 3600)
            
            if agent.is_multi:
                for i in range(agent.N_agents):
                    agent.logger.log_tabular(f"Epi_Ret_{i}", with_min_and_max=True)
                    agent.logger.log_tabular(f"Eval_ret_{i}", with_min_and_max=True)
                    agent.logger.log_tabular(f"Q_val_{i}", average_only=True)
                    agent.logger.log_tabular(f"Critic_loss_{i}", average_only=True)
                    agent.logger.log_tabular(f"Actor_loss_{i}", average_only=True)
            else:
                agent.logger.log_tabular("Epi_Ret", with_min_and_max=True)
                agent.logger.log_tabular("Eval_ret", with_min_and_max=True)
                agent.logger.log_tabular("Q_val", with_min_and_max=True)
                agent.logger.log_tabular("Loss", average_only=True)

            agent.logger.dump_tabular()

            # create evaluation plot based on current 'progress.txt'
            plot_from_progress(dir     = agent.logger.output_dir,
                               alg     = agent.name,
                               env_str = c.Env.name,
                               info    = c.Env.info)
            # save weights
            save_weights(agent, eval_ret)

def save_weights(agent: _Agent, eval_ret) -> None:

    # check whether this was the best evaluation epoch so far
    with open(f"{agent.logger.output_dir}/progress.txt") as f:
        reader = csv.reader(f, delimiter="\t")
        d = list(reader)

    df = pd.DataFrame(d)
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df = df.astype(float)

    # no best-weight-saving for multi-agent problems since the definition of best weights is not straightforward anymore
    if agent.is_multi:
        best_weights = False
    elif len(df["Avg_Eval_ret"]) == 1:
        best_weights = True
    else:
        if np.mean(eval_ret) > max(df["Avg_Eval_ret"][:-1]):
            best_weights = True
        else:
            best_weights = False

    # save net
    if hasattr(agent, "DQN"):
        torch.save(agent.DQN.state_dict(),
                    f"{agent.logger.output_dir}/{agent.name}_weights.pth")

        if best_weights:
            torch.save(agent.DQN.state_dict(),
                        f"{agent.logger.output_dir}/{agent.name}_best_weights.pth")
    else:
        torch.save(agent.actor.state_dict(), f"{agent.logger.output_dir}/{agent.name}_actor_weights.pth")
        torch.save(agent.critic.state_dict(), f"{agent.logger.output_dir}/{agent.name}_critic_weights.pth")

        if best_weights:
            torch.save(agent.actor.state_dict(), f"{agent.logger.output_dir}/{agent.name}_actor_best_weights.pth")
            torch.save(agent.critic.state_dict(), f"{agent.logger.output_dir}/{agent.name}_critic_best_weights.pth")

    # stores the replay buffer
    with open(f"{agent.logger.output_dir}/buffer.pickle", "wb") as handle:
        pickle.dump(agent.replay_buffer, handle)
