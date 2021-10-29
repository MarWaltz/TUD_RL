import gym
import gym_minatar
import numpy as np

from current_algos.Bootstrapped_DQN.bootstrapped_dqn_agent import *
from current_envs.envs import *
from current_envs.wrappers import MinAtari_wrapper, gym_POMDP_wrapper

# training config
TIMESTEPS = 5000000     # overall number of training interaction steps
EPOCH_LENGTH = 5000     # number of time steps between evaluation/logging events
EVAL_EPISODES = 1      # number of episodes to average per evaluation


def visualize_policy(env_str, state_type, dqn_weights):

    # init env
    test_env = gym.make(env_str)

    # MinAtari observation wrapper
    if "MinAtar" in env_str:
        test_env = MinAtari_Wrapper(test_env)

    # maximum episode steps
    if "MinAtar" in env_str:
        max_episode_steps = 1e4 if "Seaquest" in env_str else np.inf
    else:
        max_episode_steps = test_env._max_episode_steps
            
    # get state_shape
    if state_type == "image":
        assert "MinAtar" in env_str, "Only MinAtar-interface available for images."

        # careful, MinAtar constructs state as (height, width, in_channels), which is NOT aligned with PyTorch
        state_shape = (test_env.observation_space.shape[2], *test_env.observation_space.shape[0:2])
    
    elif state_type == "feature":
        state_shape = test_env.observation_space.shape[0]

    # init agent
    test_agent = Bootstrapped_DQN_Agent(mode         = "test",
                                        num_actions  = test_env.action_space.n, 
                                        state_type   = state_type,
                                        state_shape  = state_shape,
                                        dqn_weights  = dqn_weights,
                                        env_str      = env_str)

    for _ in range(EVAL_EPISODES):

        # get initial state
        s = test_env.reset()

        # potentially normalize it
        if test_agent.input_norm:
            s = test_agent.inp_normalizer.normalize(s, mode="test")

        cur_ret = 0
        d = False
        eval_epi_steps = 0
        
        while not d:
            
            test_env.render()

            eval_epi_steps += 1

            # select action
            a = test_agent.select_action(s, active_head=None)
            
            # perform step
            s2, r, d, _ = test_env.step(a)

            # potentially normalize s2
            if test_agent.input_norm:
                s2 = test_agent.inp_normalizer.normalize(s2, mode="test")

            # s becomes s2
            s = s2
            cur_ret += r

            # break option
            if eval_epi_steps == max_episode_steps:
                break    

visualize_policy(env_str="Breakout-MinAtar-v0", state_type="image", dqn_weights="CNN_OurBootDQN_Agent_gaussian_cdf_DQN_weights.pth")
