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
from common.eval_plot import plot_from_progress
from configs.discrete_actions import __path__
from current_algos.Bootstrapped_DQN.bootstrapped_dqn_agent import *
from current_envs.envs import *
from current_envs.wrappers.MinAtar_wrapper import MinAtar_wrapper


def evaluate_policy(test_env, test_agent, c):
    test_agent.mode = "test"
    rets = []
    
    for _ in range(c["eval_episodes"]):

        # get initial state
        s = test_env.reset()

        # potentially normalize it
        if c["input_norm"]:
            s = test_agent.inp_normalizer.normalize(s, mode="test")

        cur_ret = 0
        d = False
        eval_epi_steps = 0
        
        while not d:

            eval_epi_steps += 1

            # select action
            a = test_agent.select_action(s, active_head=None)
            
            # perform step
            s2, r, d, _ = test_env.step(a)

            # potentially normalize s2
            if c["input_norm"]:
                s2 = test_agent.inp_normalizer.normalize(s2, mode="test")

            # s becomes s2
            s = s2
            cur_ret += r

            # break option
            if eval_epi_steps == c["env"]["max_episode_steps"]:
                break

        # compute average return and append it
        rets.append(cur_ret)
    
    return rets


def train(c, agent_name):
    """Main training loop."""

    # measure computation time
    start_time = time.time()
    
    # init envs
    env = gym.make(c["env"]["name"], **c["env"]["env_kwargs"])
    test_env = gym.make(c["env"]["name"], **c["env"]["env_kwargs"])

    # wrappers
    for wrapper in c["env"]["wrappers"]:
        env = eval(wrapper)(env)
        test_env = eval(wrapper)(test_env)

    # get state_shape
    if c["env"]["state_type"] == "image":
        assert "MinAtar" in c["env"]["name"], "Only MinAtar-interface available for images."

        # careful, MinAtar constructs state as (height, width, in_channels), which is NOT aligned with PyTorch
        state_shape = (env.observation_space.shape[2], *env.observation_space.shape[0:2])
    
    elif c["env"]["state_type"] == "feature":
        state_shape = env.observation_space.shape[0]

    # seeding
    env.seed(c["seed"])
    test_env.seed(c["seed"])
    torch.manual_seed(c["seed"])
    np.random.seed(c["seed"])
    random.seed(c["seed"])

    # init agent
    agent = Bootstrapped_DQN_Agent(mode             = "train",
                                   num_actions      = env.action_space.n, 
                                   state_shape      = state_shape,
                                   state_type       = c["env"]["state_type"],
                                   dqn_weights      = c["dqn_weights"], 
                                   input_norm       = c["input_norm"],
                                   input_norm_prior = c["input_norm_prior"],
                                   double           = c["agent"][agent_name]["double"],
                                   kernel           = c["agent"][agent_name]["kernel"],
                                   kernel_param     = c["agent"][agent_name]["kernel_param"],
                                   K                = c["agent"][agent_name]["K"],
                                   mask_p           = c["agent"][agent_name]["mask_p"],
                                   gamma            = c["gamma"],
                                   tgt_update_freq  = c["tgt_update_freq"],
                                   net_struc_dqn    = c["net_struc_dqn"],
                                   optimizer        = c["optimizer"],
                                   loss             = c["loss"],
                                   lr               = c["lr"],
                                   buffer_length    = c["buffer_length"],
                                   grad_clip        = c["grad_clip"],
                                   grad_rescale     = c["grad_rescale"],
                                   act_start_step   = c["act_start_step"],
                                   upd_start_step   = c["upd_start_step"],
                                   upd_every        = c["upd_every"],
                                   batch_size       = c["batch_size"],
                                   device           = c["device"],
                                   env_str          = c["env"]["name"])

    # init the active head for action selection
    active_head = np.random.choice(agent.K)

    # get initial state and normalize it
    s = env.reset()
    if c["input_norm"]:
        s = agent.inp_normalizer.normalize(s, mode="train")

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
            a = agent.select_action(s, active_head)
        
        # perform step
        s2, r, d, _ = env.step(a)
        
        # Ignore "done" if it comes from hitting the time horizon of the environment
        d = False if epi_steps == c["env"]["max_episode_steps"] else d

        # potentially normalize s2
        if c["input_norm"]:
            s2 = agent.inp_normalizer.normalize(s2, mode="train")

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
 
            # reset active head for action selection
            active_head = np.random.choice(agent.K)

            # reset to initial state and normalize it
            s = env.reset()
            if c["input_norm"]:
                s = agent.inp_normalizer.normalize(s, mode="train")
            
            # log episode return
            agent.logger.store(Epi_Ret=epi_ret)

            # reset epi steps and epi ret
            epi_steps = 0
            epi_ret = 0

        # end of epoch handling
        if (total_steps + 1) % c["epoch_length"] == 0 and (total_steps + 1) > c["upd_start_step"]:

            epoch = (total_steps + 1) // c["epoch_length"]

            # evaluate agent with deterministic policy
            eval_ret = evaluate_policy(test_env=test_env, test_agent=copy.copy(agent), c=c)
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
            plot_from_progress(dir=agent.logger.output_dir, alg=agent.name, env_str=c["env"]["name"], info=f"lr = {c['lr']}")

            # save weights
            torch.save(agent.DQN.state_dict(), f"{agent.logger.output_dir}/{agent.name}_DQN_weights.pth")
    
            # save input normalizer values 
            if c["input_norm"]:
                with open(f"{agent.logger.output_dir}/{agent.name}_inp_norm_values.pickle", "wb") as f:
                    pickle.dump(agent.inp_normalizer.get_for_save(), f)


if __name__ == "__main__":
 
    # get config and name of agent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="seaquest.json")
    parser.add_argument("--agent_name", type=str, default="our_bootstrapped_dqn")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # read config file
    with open(__path__._path[0] + "\\" + args.config_file) as f:
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
