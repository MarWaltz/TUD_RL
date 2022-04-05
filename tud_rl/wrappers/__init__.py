import tud_rl.wrappers.gym_POMDP_wrapper as gw
import tud_rl.wrappers.MinAtar_wrapper as Min

def get_wrapper(name: str, *args, **kwargs) -> type:
    
    if "MinAtar" in name:
        return Min.MinAtar_wrapper(*args, **kwargs)
    elif "POMDP" in name:
        return gw.gym_POMDP_wrapper(*args, **kwargs)