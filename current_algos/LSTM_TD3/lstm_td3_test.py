import time

import gym
import numpy as np
import pybulletgym

from lstm_td3_agent import *
from LCP_environment import LCP_Environment

# testing config
TEST_EPISODES = 20

def test(env_str, actor_weights=None, critic_weights=None, device="cpu"):
    """Main testing loop."""
    
    if env_str != "LCP":
        raise Exception("Please use the 'LCP' environment.")
    env = LCP_Environment()

    agent = LSTM_TD3_Agent(mode           = "test",
                           action_dim     = env.action_space.shape[0], 
                           state_dim      = env.observation_space.shape[0], 
                           action_high    = env.action_space.high[0],
                           action_low     = env.action_space.low[0], 
                           actor_weights  = actor_weights, 
                           critic_weights = critic_weights, 
                           device         = device)
    episode_rewards = []
    
    for episode in range(TEST_EPISODES):

        s = env.reset()
        curr_reward = 0
        d = False
        
        r, t = 0, 0
        while not d:
            # render
            env.render(agent_name=agent.name, reward=r, episode_timestep=t)
            t += 1

            # select action
            a = agent.select_action(s)
            
            # perform step
            s2, r, d, _ = env.step(a)
            
            # s becomes s2
            s = s2
            curr_reward += r
            
        # add rewards and compute rolling reward
        episode_rewards.append(curr_reward)
        print(f"Episode: {episode}, R_20: {np.mean(episode_rewards[-20:])}")

if __name__ == "__main__":
    test(env_str="LCP", critic_weights="LSTM_TD3_Agent_critic_weights.pth", actor_weights="LSTM_TD3_Agent_actor_weights.pth", device="cpu")
