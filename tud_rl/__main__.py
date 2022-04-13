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
TASK = "viz"
CONFIG_FILE = "FossenEnvRecDQN.json"
SEED = 1230
AGENT_NAME = "RecDQN"

# initialize parser
parser = ArgumentParser()

parser.add_argument(
    "-t", "--task", type=str, default=TASK, choices=["train", "viz"],
    help="Agent task. Use `train` for training and `viz` for visualization")

parser.add_argument(
    "-c", "--config_file", type=str, default=CONFIG_FILE,
    help="Name of configuration file with file extension.")

parser.add_argument(
    "-s", "--seed", type=int, default=SEED,
    help="Random number generator seed.")

parser.add_argument(
    "-a", "--agent_name", type=str, default=AGENT_NAME,
    help="Agent from config for training. Example: `DQN` or `KEBootDQN_b`.")

args = parser.parse_args()

agent_name = args.agent_name
if args.agent_name[-1].islower():
    agent_name = args.agent_name[:-2]

# check if supplied agent name matches any available agents
validate_agent(agent_name)

# get the configuration file path depending on the chosen mode
base_path = discr_path[0] if is_discrete(agent_name) else cont_path[0]
config_path = f"{base_path}/{args.config_file}"

# parse the config file
config = ConfigFile(config_path)

# potentially overwrite seed
if args.seed is not None:
    config.overwrite(seed=args.seed)

# handle maximum episode steps
config.max_episode_handler()

if args.task == "train":
    if is_discrete(agent_name):
        discr.train(config, args.agent_name)
    else:
        cont.train(config, args.agent_name)
elif args.task == "viz":
    if is_discrete(agent_name):
        vizdiscr.test(config, args.agent_name)
    else:
        vizcont.test(config, args.agent_name)
