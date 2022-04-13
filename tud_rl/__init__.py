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
    id="FossenEnv-v0",
    entry_point=loc + "FossenEnv"
)
gym.register(
    id="FossenEnvStarScene-v0",
    entry_point=loc + "FossenEnvStarScene"
)
gym.register(
    id="FossenEnvSingleScene-v0",
    entry_point=loc + "FossenEnvSingleScene"
)


# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] - %(message)s')
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)
