from argparse import ArgumentParser

import tud_rl.envs
import tud_rl.run.train_continuous as cont
import tud_rl.run.train_discrete as discr
import tud_rl.run.visualize_continuous as vizcont
import tud_rl.run.visualize_discrete as vizdiscr
from tud_rl.agents import is_discrete, validate_agent
from tud_rl.common.configparser import ConfigFile
from tud_rl.configs.continuous_actions import __path__ as cont_path
from tud_rl.configs.discrete_actions import __path__ as discr_path

# initialize parser
parser = ArgumentParser()

parser.add_argument(
    "-t", "--task", type=str, default=None, choices=["train", "viz"],
    help="Agent task. Use `train` for training and `viz` for visualization")

parser.add_argument(
    "-c", "--config_file", type=str, default=None,
    help="Name of configuration file with file extension.")

parser.add_argument(
    "-s", "--seed", type=int, default=None,
    help="Random number generator seed.")

parser.add_argument(
    "-a", "--agent_name", type=str, default=None,
    help="Agent from config for training. Example: `DQN` or `KEBootDQN_b`.")

parser.add_argument(
    "-w", "--dqn_weights", type=str, default=None,
    help="Weights for visualization in discrete action spaces. Example: `dqn_weights.pth`.")

parser.add_argument(
    "-aw", "--actor_weights", type=str, default=None,
    help="Weights (actor) for visualization in continuous action spaces. Example: `actor_weights.pth`.")

parser.add_argument(
    "-cw", "--critic_weights", type=str, default=None,
    help="Weights (critic) for visualization in continuous action spaces. Example: `critic_weights.pth`.")

# OA START
parser.add_argument(
    "-od", "--output_dir", type=str, default=None)

parser.add_argument(
    "-pb", "--prior_buffer", type=str, default=None)
# OA END

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

# consider weights
if args.dqn_weights is not None:
    config.overwrite(dqn_weights=args.dqn_weights)

if args.actor_weights is not None:
    config.overwrite(actor_weights=args.actor_weights)

if args.critic_weights is not None:
    config.overwrite(critic_weights=args.critic_weights)

# OA START
if args.output_dir is not None:
    config.overwrite(output_dir=args.output_dir)

if args.prior_buffer is not None:
    config.overwrite(prior_buffer=args.prior_buffer)
# OA END

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
