from gym.envs.registration import register

loc = "tud_rl.envs:"

register(id="MyMountainCar-v0", entry_point= loc + "MountainCar")
register(id="Ski-v0", entry_point= loc + "Ski")
register(id="ObstacleAvoidance-v0", entry_point= loc + "ObstacleAvoidance",)
register(id="FossenEnv-v0", entry_point= loc + "FossenEnv")
register(id="FossenEnvScenarioOne-v0", entry_point= loc + "FossenEnvScenarioOne")
register(id="PathFollower-v0", entry_point= loc + "PathFollower")
