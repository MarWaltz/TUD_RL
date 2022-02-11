import argparse
import json
import pickle
import random
import time

import gym
import gym_minatar
import gym_pygame
import numpy as np
import pybulletgym
import torch
from tud_rl.agents.continuous.DDPG import DDPGAgent
from tud_rl.agents.continuous.TRYDDPG import TRYDDPGAgent
from tud_rl.agents.continuous.LSTMDDPG import LSTMDDPGAgent
from tud_rl.agents.continuous.LSTMSAC import LSTMSACAgent
from tud_rl.agents.continuous.LSTMTD3 import LSTMTD3Agent
from tud_rl.agents.continuous.SAC import SACAgent
from tud_rl.agents.continuous.TD3 import TD3Agent
from tud_rl.agents.continuous.TQC import TQCAgent
from tud_rl.common.logging_plot import plot_from_progress
from tud_rl.configs.continuous_actions import __path__
from tud_rl.envs.MountainCar import MountainCar
from tud_rl.wrappers.MinAtar_wrapper import MinAtar_wrapper


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


def evaluate_policy(test_env, agent, c):

    # go greedy
    agent.mode = "test"
    
    # final (undiscounted) sum of rewards of ALL episodes
    rets = []

    # Q-ests of all (s,a) pairs of ALL episodes
    Q_est_all_eps = []

    # Bias-ests of all (s,a) pairs of ALL episodes
    B_est_all_eps = []

    # MC-vals of all (s,a) pairs of ALL episodes
    MC_all_eps = []
    
    for _ in range(c["eval_episodes"]):

        # LSTM: init history
        if "LSTM" in agent.name:
            s_hist = np.zeros((agent.history_length, agent.state_shape))
            a_hist = np.zeros((agent.history_length, agent.num_actions))
            hist_len = 0

        # get initial state
        s = test_env.reset()

        # potentially normalize it
        if c["input_norm"]:
            s = agent.inp_normalizer.normalize(s, mode=agent.mode)

        cur_ret = 0
        d = False
        eval_epi_steps = 0

        # rewards of ONE episode
        rews_one_eps = []

        while not d:

            eval_epi_steps += 1

            # select action
            if "LSTM" in agent.name:
                a = agent.select_action(s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
            else:
                a = agent.select_action(s)

            # get current Q and bias estimate
            Q, B = agent.greedy_action_Q(s, with_bias=True)
            Q_est_all_eps.append(Q)
            B_est_all_eps.append(B)

            # perform step
            s2, r, d, _ = test_env.step(a)

            # save reward
            rews_one_eps.append(r)

            # potentially normalize s2
            if c["input_norm"]:
                s2 = agent.inp_normalizer.normalize(s2, mode=agent.mode)

            # LSTM: update history
            if "LSTM" in agent.name:
                if hist_len == agent.history_length:
                    s_hist = np.roll(s_hist, shift = -1, axis = 0)
                    s_hist[agent.history_length - 1, :] = s

                    a_hist = np.roll(a_hist, shift = -1, axis = 0)
                    a_hist[agent.history_length - 1, :] = a
                else:
                    s_hist[hist_len] = s
                    a_hist[hist_len] = a
                    hist_len += 1

            # s becomes s2
            s = s2
            cur_ret += r

            # break option
            if eval_epi_steps == c["env"]["max_episode_steps"]:
                break
        
        # append return
        rets.append(cur_ret)

        # transform reward list in MC return list and append it to overall MC returns
        MC_all_eps += get_MC_ret_from_rew(rews=rews_one_eps, gamma=agent.gamma)

    # compute difference between real and measured bias
    bias = np.array(Q_est_all_eps) - np.array(MC_all_eps)
    bias_est = np.array(B_est_all_eps)

    # continue training
    agent.mode = "train"

    return rets, np.mean(bias - bias_est)


def get_s_a_MC(env, agent, c):

    # go greedy
    agent.mode = "test"

    # s and a of ALL episodes
    s_all_eps = []
    a_all_eps = []

    # MC-vals of all (s,a) pairs of ALL episodes
    MC_ret_all_eps = []

    # init epi steps and rewards for ONE episode
    epi_steps = 0
    r_one_eps = []

    # LSTM: init history
    if "LSTM" in agent.name:
        s_hist = np.zeros((agent.history_length, agent.state_shape))
        a_hist = np.zeros((agent.history_length, agent.num_actions))
        hist_len = 0

    # get initial state and normalize it
    s = env.reset()
    if c["input_norm"]:
        s = agent.inp_normalizer.normalize(s, mode=agent.mode)

    for _ in range(c["agent"][agent.name]["MC_batch_size"]):

        epi_steps += 1

        # select action
        if "LSTM" in agent.name:
            a = agent.select_action(s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
        else:
            a = agent.select_action(s)

        # perform step
        s2, r, d, _ = env.step(a)

        # save s, a, r
        s_all_eps.append(s)
        a_all_eps.append(a)
        r_one_eps.append(r)

        # potentially normalize s2
        if c["input_norm"]:
            s2 = agent.inp_normalizer.normalize(s2, mode=agent.mode)

        # LSTM: update history
        if "LSTM" in agent.name:
            if hist_len == agent.history_length:
                s_hist = np.roll(s_hist, shift = -1, axis = 0)
                s_hist[agent.history_length - 1, :] = s

                a_hist = np.roll(a_hist, shift = -1, axis = 0)
                a_hist[agent.history_length - 1, :] = a
            else:
                s_hist[hist_len] = s
                a_hist[hist_len] = a
                hist_len += 1

        # s becomes s2
        s = s2

        # end of episode: for artificial time limit in env, we need to correct final reward to be a return
        if epi_steps == c["env"]["max_episode_steps"]:

            # backup from current Q-net: r + gamma * Q(s2, pi(s2)) with greedy pi
            r_one_eps[-1] += agent.gamma * agent.greedy_action_Q(s2, with_bias=False)

        # end of episode: artificial or true done signal
        if epi_steps == c["env"]["max_episode_steps"] or d:

            # transform rewards to returns and store them
            MC_ret_all_eps += get_MC_ret_from_rew(rews=r_one_eps, gamma=agent.gamma)

            # reset
            epi_steps = 0
            r_one_eps = []

            # LSTM: init history
            if "LSTM" in agent.name:
                s_hist = np.zeros((agent.history_length, agent.state_shape))
                a_hist = np.zeros((agent.history_length, agent.num_actions))
                hist_len = 0

            # get initial state and normalize it
            s = env.reset()
            if c["input_norm"]:
                s = agent.inp_normalizer.normalize(s, mode=agent.mode)

    # store MC from final unfinished episode
    if len(r_one_eps) > 0:

        # backup from current Q-net: r + gamma * Q(s2, pi(s2)) with greedy pi
        r_one_eps[-1] += agent.gamma * agent.greedy_action_Q(s2, with_bias=False)

        # transform rewards to returns and store them
        MC_ret_all_eps += get_MC_ret_from_rew(rews=r_one_eps, gamma=agent.gamma)

    # continue training
    agent.mode = "train"

    return np.stack(s_all_eps), np.expand_dims(a_all_eps, 1), np.expand_dims(MC_ret_all_eps, 1)


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
        raise NotImplementedError("Currently, image input is not available for continuous action spaces.")
    
    elif c["env"]["state_type"] == "feature":
        c["state_shape"] = env.observation_space.shape[0]

    # mode and action details
    c["mode"] = "train"
    c["num_actions"] = env.action_space.shape[0]
    c["action_high"] = env.action_space.high[0]
    c["action_low"]  = env.action_space.low[0]

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

    # LSTM: init history
    if "LSTM" in agent.name:
        s_hist = np.zeros((agent.history_length, agent.state_shape))
        a_hist = np.zeros((agent.history_length, agent.num_actions))
        hist_len = 0

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
            a = np.random.uniform(low=agent.action_low, high=agent.action_high, size=agent.num_actions)
        else:
            if "LSTM" in agent.name:
                a = agent.select_action(s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
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

        # LSTM: update history
        if "LSTM" in agent.name:
            if hist_len == agent.history_length:
                s_hist = np.roll(s_hist, shift = -1, axis = 0)
                s_hist[agent.history_length - 1, :] = s

                a_hist = np.roll(a_hist, shift = -1, axis = 0)
                a_hist[agent.history_length - 1, :] = a
            else:
                s_hist[hist_len] = s
                a_hist[hist_len] = a
                hist_len += 1

        # train
        if (total_steps >= c["upd_start_step"]) and (total_steps % c["upd_every"] == 0):
            for _ in range(c["upd_every"]):
                agent.train()

        # s becomes s2
        s = s2

        # end of episode handling
        if d or (epi_steps == c["env"]["max_episode_steps"]):
 
            # reset noise after episode
            if hasattr(agent, "noise"):
                agent.noise.reset()

            # LSTM: reset history
            if "LSTM" in agent.name:
                s_hist = np.zeros((agent.history_length, agent.state_shape))
                a_hist = np.zeros((agent.history_length, agent.num_actions))
                hist_len = 0

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
            eval_ret, Diff_real_est_bias = evaluate_policy(test_env=test_env, agent=agent, c=c)
            for ret in eval_ret:
                agent.logger.store(Eval_ret=ret)

            # log and dump tabular
            agent.logger.log_tabular("Epoch", epoch)
            agent.logger.log_tabular("Timestep", total_steps)
            agent.logger.log_tabular("Runtime_in_h", (time.time() - start_time) / 3600)
            agent.logger.log_tabular("Epi_Ret", with_min_and_max=True)
            agent.logger.log_tabular("Eval_ret", with_min_and_max=True)
            agent.logger.log_tabular("Q_val", with_min_and_max=True)
            agent.logger.log_tabular("Critic_loss", average_only=True)
            agent.logger.log_tabular("Actor_loss", average_only=True)

            agent.logger.log_tabular("Bias_val", with_min_and_max=True)
            agent.logger.log_tabular("Bias_loss", average_only=True)
            agent.logger.log_tabular("Diff_real_est_bias", Diff_real_est_bias)

            if "LSTM" in agent.name:
                agent.logger.log_tabular("Actor_CurFE", with_min_and_max=False)
                agent.logger.log_tabular("Actor_ExtMemory", with_min_and_max=False)
                agent.logger.log_tabular("Critic_CurFE", with_min_and_max=False)
                agent.logger.log_tabular("Critic_ExtMemory", with_min_and_max=False)

            agent.logger.dump_tabular()

            # create evaluation plot based on current 'progress.txt'
            plot_from_progress(dir=agent.logger.output_dir, alg=agent.name, env_str=c["env"]["name"], info=c["env"]["info"])

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
    parser.add_argument("--config_file", type=str, default="halfcheetah.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--agent_name", type=str, default="DDPG")
    args = parser.parse_args()

    # read config file
    with open(__path__._path[0] + "/" + args.config_file) as f:
        c = json.load(f)

    # potentially overwrite seed
    if args.seed is not None:
        c["seed"] = args.seed

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

    train(c, args.agent_name)
