import os
import numpy as np
import pandas as pd
import requests
import stable_baselines3
from stable_baselines3.common.type_aliases import GymEnv
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from typing import Optional
import torch
import gc

class envWrapper(gym.Env):
    def __init__(self, apiurl):
        super().__init__()
        self.apiurl = apiurl
        self.uuid = None
        self.info = None
        self.current_position = None
        self.new_game()

    def new_game(self):
        response = requests.post(f"{self.apiurl}/new_game")
        if response.status_code == 200:
            data = response.json()
            self.uuid = data["uuid"]

            self.action_space = Discrete(data["action_space"]["n"])
            observation_space_info = data["observation_space"]
            #print(observation_space_info)
            self.info = data["info"]
            self.observation_space = Box(
                low=observation_space_info["low"],
                high=observation_space_info["high"],
                shape=(2,),
                #low=np.array([observation_space_info["low"]] * 2),
                #high=np.array([observation_space_info["high"]] * 2),
                #dtype = np.dtype(observation_space_info["dtype"])
                dtype=np.float32
            )
        elif response.status_code == 500:
            raise Exception("Internal Server Error")

    def reset(self, seed=None, options=None):
    #def reset(self):
        response = requests.post(f"{self.apiurl}/reset/{self.uuid}", json={"uuid": self.uuid})
        #print(response)
        if response.status_code == 200:
            observation = np.array(response.json()["observation"], dtype=np.float32)
            self.current_position = tuple(observation)
            return observation, {}
        elif response.status_code == 400:
            error_message = response.json()
            print(f"{error_message}")
            #raise ValueError(f"Bad request: {response.json().get('error', 'Unknown error')}")
        elif response.status_code == 500:
            error_message = response.json()
            print(f"{error_message}")
            #raise Exception("Internal Server Error")
        #else:
            #raise Exception("Internal Server Error")

    def manhattan_distance(self, position, goal=(4, 4)):
        return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

    def step(self, action):
        old_position = self.current_position

        response = requests.post(f"{self.apiurl}/step/{self.uuid}", json={"uuid": self.uuid, "action": int(action)})
        if response.status_code == 200:
            data = response.json()
            new_observation = np.array(data["observation"], dtype=np.float32)
            new_position = tuple(new_observation)
            new_distance = self.manhattan_distance(new_position)
            old_distance = self.manhattan_distance(old_position)
            bonus_reward = (old_distance - new_distance) #* 0.5

            total_reward = data["reward"]
            if data["reward"] == -0.04:
                total_reward = 1
            if data["reward"] == -0.04:
                total_reward = -5
            if bonus_reward > 0:
                total_reward += 0.5

            self.current_position = new_position
            return (
                new_observation,
                total_reward,
                data["done"],
                data["truncated"],
                data["info"]
            )
        elif response.status_code == 400:
            raise ValueError(f"Bad request: {response.json().get('error', 'Unknown error')}")
        elif response.status_code == 500:
            raise Exception("Internal Server Error")




