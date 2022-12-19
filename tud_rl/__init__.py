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
    id="HHOS-v0",
    entry_point=loc + "HHOS_Env"
)
gym.register(
    id="HHOS-PathFollowing-v0",
    entry_point=loc + "HHOS_PathFollowing_Env"
)
gym.register(
    id="HHOS-PathFollowing-Validation-v0",
    entry_point=loc + "HHOS_PathFollowing_Validation"
)
gym.register(
    id="HHOS-PathPlanning-v0",
    entry_point=loc + "HHOS_PathPlanning_Env"
)
gym.register(
    id="HHOS-PathPlanning-Validation-v0",
    entry_point=loc + "HHOS_PathPlan_Validation"
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

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] - %(message)s')
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)
