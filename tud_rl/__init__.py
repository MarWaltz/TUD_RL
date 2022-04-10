import logging
import sys
from gym.envs.registration import register

loc = "tud_rl.envs:"

register(id="MyMountainCar-v0", entry_point=loc + "MountainCar")
register(id="Ski-v0", entry_point=loc + "Ski")
register(id="ObstacleAvoidance-v0",entry_point=loc + "ObstacleAvoidance")
register(id="FossenEnv-v0", entry_point=loc + "FossenEnv",)
register(id="FossenEnvScenarioOne-v0",entry_point=loc + "FossenEnvScenarioOne")
register(id="PathFollower-v0", entry_point=loc + "PathFollower")

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] - %(message)s')
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)

