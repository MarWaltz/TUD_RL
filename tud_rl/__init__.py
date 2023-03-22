import logging
import sys
import gym

loc = "tud_rl.envs:"

# ----------- Custom Env Registration --------------------------

gym.register(
    id="MyMountainCar-v0",
    entry_point=loc + "MountainCar"
)
gym.register(
    id="Ski-v0",
    entry_point=loc + "Ski"
)
gym.register(
    id="ObstacleAvoidance-v0",
    entry_point=loc + "ObstacleAvoidance"
)
gym.register(
    id="ComplexOA-v0",
    entry_point=loc + "ComplexOA"
)
gym.register(
    id="MMGStar-v0",
    entry_point=loc + "MMG_Star"
)
gym.register(
    id="MMGEnv-v0",
    entry_point=loc + "MMG_Env"
)
gym.register(
    id="MMGImazu-v0",
    entry_point=loc + "MMG_Imazu"
)
gym.register(
    id="MMGWorld-v0",
    entry_point=loc + "MMG_World"
)

gym.register(
    id="HHOS-Following-v0",
    entry_point=loc + "HHOS_Following_Env"
)
gym.register(
    id="HHOS-Following-Validation-v0",
    entry_point=loc + "HHOS_Following_Validation"
)

gym.register(
    id="HHOS-OpenPlanning-v0",
    entry_point=loc + "HHOS_OpenPlanning_Env"
)
gym.register(
    id="HHOS-OpenPlanning-Validation-v0",
    entry_point=loc + "HHOS_OpenPlanning_Validation"
)

gym.register(
    id="HHOS-RiverPlanning-v0",
    entry_point=loc + "HHOS_RiverPlanning_Env"
)
gym.register(
    id="HHOS-RiverPlanning-Validation-v0",
    entry_point=loc + "HHOS_RiverPlanning_Validation"
)
gym.register(
    id="HHOS-RiverPipeline-v0",
    entry_point=loc + "HHOS_RiverPipeline_Env"
)

gym.register(
    id="PredatorPrey-v0",
    entry_point=loc + "PredatorPrey"
)
gym.register(
    id="CoopNavigation-v0",
    entry_point=loc + "CoopNavigation"
)
gym.register(
    id="UAM-v0",
    entry_point=loc + "UAM"
)
gym.register(
    id="SimpleComm-v0",
    entry_point=loc + "SimpleComm"
)

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] - %(message)s')
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)
