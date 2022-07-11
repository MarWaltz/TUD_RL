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
    id="FossenEnv-v0",
    entry_point=loc + "FossenEnv"
)
gym.register(
    id="FossenImazu-v0",
    entry_point=loc + "FossenImazu"
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
    id="MMGSEval-v0",
    entry_point=loc + "MMG_SEval"
)

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] - %(message)s')
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)
