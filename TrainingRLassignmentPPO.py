import os
import numpy as np
import pandas as pd
import requests
import stable_baselines3
from stable_baselines3.common.type_aliases import GymEnv
from typing import Optional
from RLassignment import envWrapper
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback

env1 = envWrapper(apiurl="http://3.77.211.177:5005")
#env1 = TimeLimit(env1, max_episode_steps=20)

cnn = dict(net_arch=[64, 64])

#model = PPO("MlpPolicy", env1, policy_kwargs=cnn, verbose=1, tensorboard_log="./ppo_tensorboard03/", learning_rate=0.001, ent_coef=0.1, gamma=0.95, gae_lambda = 1.0, batch_size=256, clip_range=0.4)
model = PPO("MlpPolicy", env1, policy_kwargs=cnn, verbose=1, tensorboard_log="./ppo_tensorboard001/", learning_rate= 0.001)

model.learn(total_timesteps=50000)
model.save("agent1")

env2 = envWrapper(apiurl="http://3.77.211.177:5005")
env2 = TimeLimit(env2, max_episode_steps=20)

mean_reward, standard_deviation = evaluate_policy(model, env2, n_eval_episodes=100, render=False)
print(f"mean reward: {mean_reward:.2f}, standard deviation: {standard_deviation:.2f}")

# tensorboard --logdir=./ppo_tensorboard001/ --port=6006
# tensorboard --logdir=./ppo_tensorboardcustom02/ --port=6001
