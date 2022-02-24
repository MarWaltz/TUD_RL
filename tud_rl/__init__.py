from gym.envs.registration import register 

register(
    id="MyMountainCar-v0", 
    entry_point="tud_rl.envs:MountainCar",
)

register(
    id="Ski-v0", 
    entry_point="tud_rl.envs:Ski",
)

register(
    id="ObstacleAvoidance-v0", 
    entry_point="tud_rl.envs:ObstacleAvoidance",
)

register(
    id="FossenCS2-v0", 
    entry_point="tud_rl.envs:FossenCS2",
)
