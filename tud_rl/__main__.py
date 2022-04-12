import tud_rl.run.train_continuous as cont
import tud_rl.run.visualize_continuous as vizcont
import tud_rl.run.train_discrete as discr
import tud_rl.run.visualize_discrete as vizdiscr

from argparse import ArgumentParser

from tud_rl.common.configparser import ConfigFile
from tud_rl.agents import validate_agent, is_discrete
from tud_rl.configs.continuous_actions import __path__ as cont_path
from tud_rl.configs.discrete_actions import __path__ as discr_path

# OVERRIDE parser values. This enables you to run the `tud_rl`
# package from this file in an IDE without the command line
MODE = "train"
CONFIG_FILE = "pathfollower.yaml"
SEED = 123
AGENT_NAME = "SCDQN_b"


# Initialize parser
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
    "-a", "--agent_name", type=str, default=AGENT_NAME,
    help="Agent from config for training. Example: `DQN` or `DQN_b`.")

args = parser.parse_args()

base_agent = args.agent_name
if args.agent_name[-1].islower():
    base_agent = args.agent_name[:-2]

# Check if supplied agent name
# matches any available agents.
validate_agent(base_agent)

# Get the configurarion file path depending on the chosen mode
base_path = discr_path[0] if is_discrete(base_agent) else cont_path[0]
config_path = f"{base_path}/{args.config_file}"

# Parse the config file
config = ConfigFile(config_path)

# potentially overwrite seed
if args.seed is not None:
    config.overwrite(seed=args.seed)

# handle maximum episode steps
config.max_episode_handler()

if args.mode == "train":
    if is_discrete(base_agent):
        discr.train(config, args.agent_name)
    else:
        cont.train(config, args.agent_name)
elif args.mode == "viz":
    if is_discrete(base_agent):
        vizdiscr.test(config, args.agent_name)
    else:
        vizcont.test(config, args.agent_name)
