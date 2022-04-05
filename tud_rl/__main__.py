
import argparse
import numpy as np
import tud_rl.run.train_continuous as cont
import tud_rl.run.train_discrete as discr

from tud_rl.common.configparser import Configfile 
from tud_rl.configs.continuous_actions import __path__ as cont_path
from tud_rl.configs.discrete_actions import __path__ as discr_path


if __name__ == "__main__":

    # get config and name of agent
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--mode",type=str, default="discr", choices=["discr", "cont"],
                        help="Train mode: Use `di` for discre training environments "
                        "and `co` for continuous ones.")
    parser.add_argument("-c","--config_file", type=str, default="ski_mdp.json",
                        help="Name of configuration file with file extension.")
    parser.add_argument("-s","--seed", type=int, default=None,
                        help="Random number generator seed.")
    parser.add_argument("-a","--agent_name", type=str, default="LSTMDDPG",
                        help="Agent from config for training. Example: `DQN` or `DQN_b`.")
    args = parser.parse_args()
    
    base_path = cont_path[0] if args.mode == "cont" else discr_path[0]
    config_path = base_path + "/" + args.config_file
        
    config = Configfile(config_path)

    # potentially overwrite seed
    if args.seed is not None:
        config.seed = args.seed

    # handle maximum episode steps
    if config.Env.max_episode_steps == -1:
        config.Env.max_episode_steps = np.inf

    if args.mode == "discr":
        discr.train(config, args.agent_name)
    elif args.mode == "cont":
        cont.train(config, args.agent_name)