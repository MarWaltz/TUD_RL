import tud_rl.run.train_continuous as cont
import tud_rl.run.visualize_continuous as vizcont
import tud_rl.run.train_discrete as discr
import tud_rl.run.visualize_discrete as vizdiscr

from argparse import ArgumentParser

from tud_rl.common.configparser import ConfigFile
from tud_rl.agents import validate_agent, is_discrete
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

# COLREGs start
parser.add_argument(
    "-wh", "--w_head", type=float, default=None)

parser.add_argument(
    "-wc", "--w_comf", type=float, default=None)

parser.add_argument(
    "-wcl", "--w_coll", type=float, default=None)

parser.add_argument(
    "-sd", "--state_design", type=str, default=None)
# COLREGs end

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

# COLREGs start
if args.w_head is not None:
    config.overwrite(w_head=args.w_head)

if args.w_comf is not None:
    config.overwrite(w_comf=args.w_comf)

if args.w_coll is not None:
    config.overwrite(w_coll=args.w_coll)

if args.state_design is not None:
    config.overwrite(state_design=args.state_design)
# COLREGs end


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