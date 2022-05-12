
"""
Main script for running the tud_rl package 
from inside an editor/IDE.
Basically the same as __main__.py but 
without the argument parser. 
"""
import tud_rl.run.train_continuous as cont
import tud_rl.run.visualize_continuous as vizcont
import tud_rl.run.train_discrete as discr
import tud_rl.run.visualize_discrete as vizdiscr

from tud_rl.common.configparser import ConfigFile
from tud_rl.agents import validate_agent, is_discrete
from tud_rl.configs.continuous_actions import __path__ as cont_path
from tud_rl.configs.discrete_actions import __path__ as discr_path

# ---------------- User Settings -----------------------------
# ------------------------------------------------------------

TASK        = "train"           # ["train", "viz"]
CONFIG_FILE = "oa_mdp.yaml"     # configuration file as `.yaml` or `.json`
SEED        = 42                # set a seed different to the one specified in your config
AGENT_NAME  = "TD3"             # agent to train/viz
DQN_WEIGHTS = None              # path to file for weight initialization (discrete actions)
ACTOR_WEIGHTS = None            # path to file for weight initialization (continuous actions)
CRITIC_WEIGHTS = None           # path to file for weight initialization (continuous actions)

# ------------------------------------------------------------
# ------------------------------------------------------------

if AGENT_NAME[-1].islower():
    validate_agent(AGENT_NAME[:-2])
    discrete = is_discrete(AGENT_NAME[:-2])
else:
    validate_agent(AGENT_NAME)
    discrete = is_discrete(AGENT_NAME)

# get the configuration file path depending on the chosen mode
base_path = discr_path[0] if discrete else cont_path[0]
config_path = f"{base_path}/{CONFIG_FILE}"

# parse the config file
config = ConfigFile(config_path)

# potentially overwrite seed
if SEED is not None:
    config.overwrite(seed=SEED)

# consider weights
if DQN_WEIGHTS is not None:
    config.overwrite(dqn_weights=DQN_WEIGHTS)

if CRITIC_WEIGHTS is not None:
    config.overwrite(critic_weights=CRITIC_WEIGHTS)

if ACTOR_WEIGHTS is not None:
    config.overwrite(actor_weights=ACTOR_WEIGHTS)

# handle maximum episode steps
config.max_episode_handler()

if TASK == "train":
    if discrete:
        discr.train(config, AGENT_NAME)
    else:
        cont.train(config, AGENT_NAME)
elif TASK == "viz":
    if discrete:
        vizdiscr.test(config, AGENT_NAME)
    else:
        vizcont.test(config, AGENT_NAME)