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
from tud_env.envs.MountainCar import MountainCar
from tud_env.wrappers.MinAtar_wrapper import MinAtar_wrapper
from tud_rl.agents.discrete.BootDQN import BootDQNAgent
from tud_rl.agents.discrete.DDQN import DDQNAgent
from tud_rl.agents.discrete.DQN import DQNAgent
from tud_rl.agents.discrete.EnsembleDQN import EnsembleDQNAgent
from tud_rl.agents.discrete.KEBootDQN import KEBootDQNAgent
from tud_rl.agents.discrete.MaxMinDQN import MaxMinDQNAgent
from tud_rl.agents.discrete.SCDQN import SCDQNAgent
from tud_rl.common.logging_plot import plot_from_progress
from tud_rl.configs.discrete_actions import __path__


def get_MC_ret_from_rew(rews, gamma):
    """Returns for a given episode of rewards (list) the corresponding list of MC-returns under a specified discount factor."""

    MC = 0
    MC_list = []
    
    for r in reversed(rews):
        # compute one-step backup
        backup = r + gamma * MC
        
        # add to MCs
        MC_list.append(backup)
        
        # update MC
        MC = backup
    
    return list(reversed(MC_list))


def evaluate_policy(test_env, test_agent, c):
    test_agent.mode = "test"

    # final (undiscounted) sum of rewards of ALL episodes
    rets = []

    # Q-ests of all (s,a) pairs of ALL episodes
    Q_est_all_eps = []

    # Sd of Q-ests of all (s,a) pairs of ALL episodes
    Q_sd_all_eps = []

    # MC-vals of all (s,a) pairs of ALL episodes
    MC_all_eps = []
    
    for _ in range(c["eval_episodes"]):

        # get initial state
        s = test_env.reset()

        # potentially normalize it
        if c["input_norm"]:
            s = test_agent.inp_normalizer.normalize(s, mode=test_agent.mode)

        cur_ret = 0
        d = False
        eval_epi_steps = 0

        # rewards of ONE episode
        rews_one_eps = []

        while not d:

            eval_epi_steps += 1

            # select action
            a = test_agent.select_action(s)

            # get current Q estimate and its sd
            s = torch.tensor(s.astype(np.float32)).unsqueeze(0)
            q_ens = [net(s) for net in test_agent.EnsembleDQN]
            q_ens = torch.stack(q_ens)

            Q_est = test_agent._ensemble_reduction(q_ens)[0][a].item()
            Q_est_all_eps.append(Q_est)

            Q_sd = torch.std(q_ens, dim=0)[0][a].item()
            Q_sd_all_eps.append(Q_sd)

            #Q_est_all_eps.append(test_agent.DQN(s)[0][a].item())

            # perform step
            s2, r, d, _ = test_env.step(a)

            # save reward
            rews_one_eps.append(r)

            # potentially normalize s2
            if c["input_norm"]:
                s2 = test_agent.inp_normalizer.normalize(s2, mode=test_agent.mode)

            # s becomes s2
            s = s2
            cur_ret += r

            # break option
            if eval_epi_steps == c["env"]["max_episode_steps"]:
                break
        
        # append return
        rets.append(cur_ret)
        
        # transform reward list in MC return list and append it to overall MC returns
        MC_all_eps += get_MC_ret_from_rew(rews=rews_one_eps, gamma=test_agent.gamma)
        
    # compute bias
    bias = np.array(Q_est_all_eps) - np.array(MC_all_eps)
    Q_sd = np.array(Q_sd_all_eps)

    return rets, np.mean(bias), np.std(bias), np.max(bias), np.min(bias), np.corrcoef(bias, Q_sd)[0][1]


def train(c, agent_name):
    """Main training loop."""

    # measure computation time
    start_time = time.time()
    
    # init envs
    env = gym.make(c["env"]["name"], **c["env"]["env_kwargs"])
    test_env = gym.make(c["env"]["name"], **c["env"]["env_kwargs"])

    # wrappers
    for wrapper in c["env"]["wrappers"]:
        wrapper_kwargs = c["env"]["wrapper_kwargs"][wrapper]
        env = eval(wrapper)(env, **wrapper_kwargs)
        test_env = eval(wrapper)(test_env, **wrapper_kwargs)

    # get state_shape
    if c["env"]["state_type"] == "image":
        assert "MinAtar" in c["env"]["name"], "Only MinAtar-interface available for images."

        # careful, MinAtar constructs state as (height, width, in_channels), which is NOT aligned with PyTorch
        c["state_shape"] = (env.observation_space.shape[2], *env.observation_space.shape[0:2])
    
    elif c["env"]["state_type"] == "feature":
        c["state_shape"] = env.observation_space.shape[0]

    # mode and num actions
    c["mode"] = "train"
    c["num_actions"] = env.action_space.n

    # seeding
    env.seed(c["seed"])
    test_env.seed(c["seed"])
    torch.manual_seed(c["seed"])
    np.random.seed(c["seed"])
    random.seed(c["seed"])

    # init agent
    if agent_name[-1].islower():
        agent = eval(agent_name[:-2] + "Agent")(c, agent_name)
    else:
        agent = eval(agent_name + "Agent")(c, agent_name)

    # get initial state and normalize it
    s = env.reset()
    if c["input_norm"]:
        s = agent.inp_normalizer.normalize(s, mode=agent.mode)

    # init epi step counter and epi return
    epi_steps = 0
    epi_ret = 0
    
    # main loop    
    for total_steps in range(c["timesteps"]):

        epi_steps += 1
        
        # select action
        if total_steps < c["act_start_step"]:
            a = np.random.randint(low=0, high=agent.num_actions, size=1, dtype=int).item()
        else:
            a = agent.select_action(s)
        
        # perform step
        s2, r, d, _ = env.step(a)
        
        # Ignore "done" if it comes from hitting the time horizon of the environment
        d = False if epi_steps == c["env"]["max_episode_steps"] else d

        # potentially normalize s2
        if c["input_norm"]:
            s2 = agent.inp_normalizer.normalize(s2, mode=agent.mode)

        # add epi ret
        epi_ret += r
        
        # memorize
        agent.memorize(s, a, r, s2, d)

        # train
        if (total_steps >= c["upd_start_step"]) and (total_steps % c["upd_every"] == 0):
            for _ in range(c["upd_every"]):
                agent.train()

        # s becomes s2
        s = s2

        # end of episode handling
        if d or (epi_steps == c["env"]["max_episode_steps"]):
 
            # reset active head for BootDQN
            if "Boot" in agent_name:
                agent.reset_active_head()

            # reset to initial state and normalize it
            s = env.reset()
            if c["input_norm"]:
                s = agent.inp_normalizer.normalize(s, mode=agent.mode)
            
            # log episode return
            agent.logger.store(Epi_Ret=epi_ret)
            
            # reset epi steps and epi ret
            epi_steps = 0
            epi_ret = 0

        # end of epoch handling
        if (total_steps + 1) % c["epoch_length"] == 0 and (total_steps + 1) > c["upd_start_step"]:

            epoch = (total_steps + 1) // c["epoch_length"]

            # evaluate agent with deterministic policy
            eval_ret, avg_bias, std_bias, max_bias, min_bias, bias_unc_cor = evaluate_policy(test_env=test_env, test_agent=copy.copy(agent), c=c)
            for ret in eval_ret:
                agent.logger.store(Eval_ret=ret)

            # log and dump tabular
            agent.logger.log_tabular("Epoch", epoch)
            agent.logger.log_tabular("Timestep", total_steps)
            agent.logger.log_tabular("Runtime_in_h", (time.time() - start_time) / 3600)
            agent.logger.log_tabular("Epi_Ret", with_min_and_max=True)
            agent.logger.log_tabular("Eval_ret", with_min_and_max=True)
            agent.logger.log_tabular("Q_val", with_min_and_max=True)
            agent.logger.log_tabular("Loss", average_only=True)
            agent.logger.log_tabular("Avg_bias", avg_bias)
            agent.logger.log_tabular("Std_bias", std_bias)
            agent.logger.log_tabular("Max_bias", max_bias)
            agent.logger.log_tabular("Min_bias", min_bias)
            agent.logger.log_tabular("Bias_Unc_cor", bias_unc_cor)
            agent.logger.dump_tabular()

            # create evaluation plot based on current 'progress.txt'
            plot_from_progress(dir=agent.logger.output_dir, alg=agent.name, env_str=c["env"]["name"], info=f"lr = {c['lr']}")

            # save weights
            if not any([word in agent.name for word in ["ACCDDQN", "Ensemble", "MaxMin"]]):
                torch.save(agent.DQN.state_dict(), f"{agent.logger.output_dir}/{agent.name}_DQN_weights.pth")
    
            # save input normalizer values 
            if c["input_norm"]:
                with open(f"{agent.logger.output_dir}/{agent.name}_inp_norm_values.pickle", "wb") as f:
                    pickle.dump(agent.inp_normalizer.get_for_save(), f)


if __name__ == "__main__":

    # get config and name of agent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="asterix.json")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--agent_name", type=str, default="EnsembleDQN")
    args = parser.parse_args()

    # read config file
    with open(__path__._path[0] + "/" + args.config_file) as f:
        c = json.load(f)

    # potentially overwrite lr and seed
    if args.lr is not None:
        c["lr"] = args.lr
    if args.seed is not None:
        c["seed"] = args.seed

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

    train(c, args.agent_name)
