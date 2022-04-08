from typing import Optional
import numpy as np
import tud_rl.run.train_continuous as cont
import tud_rl.run.visualize_continuous as vizcont
import tud_rl.run.train_discrete as discr
import tud_rl.run.visualize_discrete as vizdiscr

from argparse import ArgumentParser, Namespace
from tud_rl.common.configparser import ConfigFile
from tud_rl.configs.continuous_actions import __path__ as cont_path
from tud_rl.configs.discrete_actions import __path__ as discr_path
from tud_rl.agents._ddispatch import _DAGENTS

# OVERRIDE default parser values. This enables you to run the `tud_rl`
# package from this file in an IDE without the command line
MODE = "train"
CONFIG_FILE = "pathfollower.yaml"
SEED = 123
AGENT_NAME = "SCDQN_b"
LEARNING_RATE = None


# get config and name of agent
parser = ArgumentParser()
parser.add_argument(
    "-m", "--mode", type=str, default=MODE, choices=["train", "viz"],
    help="Agent mode. Use `train` for training and `viz` for visualization")
parser.add_argument(
    "-c", "--config_file", type=str, default=CONFIG_FILE,
    help="Name of configuration file with file extension.")
parser.add_argument(
    "-s", "--seed", type=int, default=SEED,
    help="Random number generator seed.")
parser.add_argument(
    "-l", "--lr", type=Optional[float], default=LEARNING_RATE,
    help="Override learning rate.")
parser.add_argument(
    "-a", "--agent_name", type=str, default=AGENT_NAME,
    help="Agent from config for training. Example: `DQN` or `DQN_b`.")

args: Namespace = parser.parse_args()

# Get the configurarion file path depending on the chosen mode
base_path = cont_path[0] if args.mode == "cont" else discr_path[0]
config_path = base_path + "/" + args.config_file

# Parse the config file
config = ConfigFile(config_path)

# potentially overwrite seed
if args.seed is not None:
    config.seed = args.seed

# handle maximum episode steps
if config.Env.max_episode_steps == -1:
    config.Env.max_episode_steps = np.inf

base_agent = args.agent_name

if args.agent_name[-1].islower():
    base_agent = args.agent_name[:-2]

if args.mode == "train":
    if any(agent.startswith(base_agent) for agent in _DAGENTS):
        discr.train(config, args.agent_name)
    else:
        cont.train(config, args.agent_name)
elif args.mode == "viz":
    if any(agent.startswith(base_agent) for agent in _DAGENTS):
        vizdiscr.test(config, args.agent_name)
    else:
        vizcont.test(config, args.agent_name)
