# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from torch import nn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

import gym
import torch
import numpy as np


import random



training = True
no_rendering = True
continue_training = True
SEED_for_ct = 11255

if no_rendering: 
    from CarlaEnv_no_rendering import CarlaEnv
else: 
    from CarlaEnv import CarlaEnv


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record("Num timesteps", self.num_timesteps)
        return True



class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True
    
def evaluate(model, env, num_episodes):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            env.render()
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward

def launch_env(id=None):
    env = None
    if id is None:
        env = CarlaEnv()
    else:
        env = gym.make(id)

    return env

if __name__ == "__main__":
    SEED = random.randint(0,20000)
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[128, 128], vf=[128, 128]))
    # Parallel environments
    env = launch_env()
    
    if training:
        if continue_training:
            log_dir = str(SEED_for_ct) + "_tmp/"
            model_path = log_dir + "best_model.zip"
            log_path = "./ppo_" + str(SEED_for_ct) + "/"
            env = Monitor(env, log_dir)
            callback = TensorboardCallback()
            callback_save_best = SaveOnBestTrainingRewardCallback(check_freq=2000, log_dir=log_dir)

            model = PPO.load(model_path, tensorboard_log=log_path)
            model.set_env(env)
            model.learn(total_timesteps=int(2e6),progress_bar=True, callback=callback_save_best, reset_num_timesteps=False)

            model.save("ppo_"+str(SEED_for_ct))
            del model # remove to demonstrate saving and loading

            model = PPO.load("ppo_"+str(SEED_for_ct))
            mean_reward = evaluate(model, env, num_episodes=10)

        else:
            # Create log dir
            log_dir = str(SEED) + "_tmp/"
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, log_dir)
            callback = TensorboardCallback()
            callback_save_best = SaveOnBestTrainingRewardCallback(check_freq=2000, log_dir=log_dir)
            model = PPO("MlpPolicy", env, learning_rate=0.0005, n_steps=2048, tensorboard_log = "./ppo_" + str(SEED) + "/",policy_kwargs=policy_kwargs,verbose=1)
            model.learn(total_timesteps=int(2e6), progress_bar=True, callback=callback_save_best)
            model.save("ppo_"+str(SEED))
            del model # remove to demonstrate saving and loading

            model = PPO.load("ppo_"+str(SEED))
            mean_reward = evaluate(model, env, num_episodes=10)
    else:
        model = PPO.load("ppo_")
        mean_reward = evaluate(model, env, num_episodes=10)
        vec_env = env
        obs = vec_env.reset()

        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = vec_env.step(action)
            vec_env.render()

