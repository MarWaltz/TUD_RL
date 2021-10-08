import argparse
import copy
import pickle
import random
import time

import gym
import gym_minatar
import matplotlib.pyplot as plt
import numpy as np
import torch
from current_algos.CNN_based.DQN_MinAtar.dqn_agent_MinAtar import *
from current_algos.common.custom_envs import MountainCar
from current_algos.common.eval_plot import plot_from_progress

# training config
TIMESTEPS = 50000     # overall number of training interaction steps
EPOCH_LENGTH = 5000     # number of time steps between evaluation/logging events
EVAL_EPISODES = 10      # number of episodes to average per evaluation
RUNS = 1                # number of runs to repeat the experiment

def evaluate_policy(test_env, test_agent):
    test_agent.mode = "test"
    rets = []
    
    for _ in range(EVAL_EPISODES):
        # get initial state
        s = test_env.reset()

        # potentially normalize it
        if test_agent.input_norm:
            s = test_agent.inp_normalizer.normalize(s, mode="test")

        # change s to be of shape (in_channels, height, width) instead of (height, width, in_channels)
        s = np.moveaxis(s, -1, 0)

        cur_ret = 0
        d = False
        
        while not d:

            # select action
            a = test_agent.select_action(s)
            
            # perform step
            s2, r, d, _ = test_env.step(a)

            # potentially normalize s2
            if test_agent.input_norm:
                s2 = test_agent.inp_normalizer.normalize(s2, mode="test")

            # change s2 to be of shape (in_channels, height, width) instead of (height, width, in_channels)
            s2 = np.moveaxis(s2, -1, 0)

            # s becomes s2
            s = s2
            cur_ret += r
        
        # compute average return and append it
        rets.append(cur_ret)
    
    return rets


def train(env_str, double, lr, dqn_weights=None, seed=0, device="cpu"):
    """Main training loop."""

    # measure computation time
    start_time = time.time()
    
    # init env
    if env_str == "MountainCar":
        env = MountainCar(rewardStd=0)
        test_env = MountainCar(rewardStd=0)
        max_episode_steps = env._max_episode_steps
    else:
        env = gym.make(env_str)
        test_env = gym.make(env_str)
        max_episode_steps = np.inf if "MinAtar" in env_str else env._max_episode_steps

    # seeding
    env.seed(seed)
    test_env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # careful, MinAtar constructs state as (height, width, in_channels), which is NOT aligned with PyTorch
    state_shape = (env.observation_space.shape[2], *env.observation_space.shape[0:2])

    # init agent
    agent = CNN_DQN_Agent(mode        = "train",
                          num_actions = env.action_space.n, 
                          state_shape = state_shape,
                          double      = double,
                          lr          = lr,
                          dqn_weights = dqn_weights,
                          device      = device)
    
    # get initial state and normalize it
    s = env.reset()
    if agent.input_norm:
        s = agent.inp_normalizer.normalize(s, mode="train")

    # change s to be of shape (in_channels, height, width) instead of (height, width, in_channels)
    s = np.moveaxis(s, -1, 0)

    # init epi step counter and epi return
    epi_steps = 0
    epi_ret = 0
    epi_ret_list = []
    epi_ret_step_list = []
    
    # main loop    
    for total_steps in range(TIMESTEPS):

        epi_steps += 1
        
        # select action
        if total_steps < agent.act_start_step:
            a = np.random.randint(low=0, high=agent.num_actions, size=1, dtype=int).item()
        else:
            a = agent.select_action(s)
        
        # perform step
        s2, r, d, _ = env.step(a)
        
        # Ignore "done" if it comes from hitting the time horizon of the environment
        d = False if epi_steps == max_episode_steps else d

        # potentially normalize s2
        if agent.input_norm:
            s2 = agent.inp_normalizer.normalize(s2, mode="train")

        # change s2 to be of shape (in_channels, height, width) instead of (height, width, in_channels)
        s2 = np.moveaxis(s2, -1, 0)

        # add epi ret
        epi_ret += r
        
        # memorize
        agent.memorize(s, a, r, s2, d)

        # train
        if (total_steps >= agent.upd_start_step) and (total_steps % agent.upd_every == 0):
            for _ in range(agent.upd_every):
                agent.train()

        # s becomes s2
        s = s2

        # end of episode handling
        if d or (epi_steps == max_episode_steps):
 
            # reset to initial state and normalize it
            s = env.reset()
            if agent.input_norm:
                s = agent.inp_normalizer.normalize(s, mode="train")
            
            # change s to be of shape (in_channels, height, width) instead of (height, width, in_channels)
            s = np.moveaxis(s, -1, 0)

            # log episode return
            agent.logger.store(Epi_Ret=epi_ret)
            
            # append episode return list
            epi_ret_list.append(epi_ret)
            epi_ret_step_list.append(total_steps)

            # reset epi steps and epi ret
            epi_steps = 0
            epi_ret = 0

        # end of epoch handling
        if (total_steps + 1) % EPOCH_LENGTH == 0 and (total_steps + 1) > agent.upd_start_step:

            epoch = (total_steps + 1) // EPOCH_LENGTH

            # save epi ret list and the corresponding time steps
            np.save(file=f"{agent.logger.output_dir}/G_list.npy", arr=np.array(epi_ret_list))
            np.save(file=f"{agent.logger.output_dir}/G_step_list.npy", arr=np.array(epi_ret_step_list))

            # evaluate agent with deterministic policy
            eval_ret = evaluate_policy(test_env=test_env, test_agent=copy.copy(agent))
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
            agent.logger.dump_tabular()

            # create evaluation plot based on current 'progress.txt'
            plot_from_progress(dir=agent.logger.output_dir, alg=agent.name, env_str=env_str, info=f"lr = {agent.lr}")

            # save weights
            torch.save(agent.DQN.state_dict(), f"{agent.logger.output_dir}/{agent.name}_DQN_weights.pth")
    
            # save input normalizer values 
            if agent.input_norm:
                with open(f"{agent.logger.output_dir}/{agent.name}_inp_norm_values.pickle", "wb") as f:
                    pickle.dump(agent.inp_normalizer.get_for_save(), f)
    
    # return G and t
    return np.array(epi_ret_list), np.array(epi_ret_step_list)


def exponential_smoothing(x, alpha=0.05):
    s = np.zeros_like(x)

    for idx, x_val in enumerate(x):
        if idx == 0:
            s[idx] = x[idx]
        else:
            s[idx] = alpha * x_val + (1-alpha) * s[idx-1]

    return s


def get_MA(x, n):
    """Calculates the moving average. Return size identical to input size.

    Args:
        x (np.array): Time series
        n (int): Number of points to average
    """
    start = np.cumsum(x[:n-1]) / np.arange(1, n)
    rest  = np.convolve(x, np.ones(n)/n, mode="valid")
    return np.append(start, rest)


if __name__ == "__main__":
    
    # helper function for parser
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # init and prepare argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_str", type=str, default="Breakout-MinAtar-v0")
    parser.add_argument("--double", type=str2bool, default=True)
    args = parser.parse_args()

    # set number of torch threads
    torch.set_num_threads(torch.get_num_threads())

    # learning rates
    LRS = [10**exp for exp in [-5.0, -4.0, -3.0]]

    # empty G array
    G_array = np.empty((RUNS, len(LRS), TIMESTEPS))

    # run main loop for all runs and lrs
    for run_id in range(RUNS):
        
        for lr_id, lr in enumerate(LRS):
        
            G, t = train(env_str=args.env_str, double=args.double, lr=lr, seed=int(100+lr))

            # average and smoothed G array
            G = exponential_smoothing(get_MA(G, 100), 0.05)

            # need to fill G for all time steps
            G_filled = np.zeros(TIMESTEPS)
            G_filled[0] = G[0]
            G_filled[t] = G

            # replace zeros with last non-zero element
            for g_idx, g in enumerate(G_filled):
                if g == 0:
                    G_filled[g_idx] = G_filled[g_idx-1]

            G_array[run_id, lr_id, :] = G_filled


    # --------------- create two plots ---------------
    fig = plt.figure(figsize=(17, 10))

    gs  = fig.add_gridspec(1, 2)
    ax0 = fig.add_subplot(gs[0, 0])   
    ax1 = fig.add_subplot(gs[0, 1])   

    # 1. whole time series
    G_mean = np.mean(G_array, axis=0)
    G_error = np.std(G_array, axis=0) / np.sqrt(RUNS)

    for lr_id, lr in enumerate(LRS):
        ax0.plot(np.arange(TIMESTEPS), G_mean[lr_id], label=lr)
        ax0.fill_between(np.arange(TIMESTEPS), G_mean[lr_id] - 0.5*G_error[lr_id], G_mean[lr_id] + 0.5*G_error[lr_id])
    ax0.legend()

    # 2. final performance
    ax1.plot(LRS, G_mean[:, -1])

    plt.show()
