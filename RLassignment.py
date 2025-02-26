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
            return np.array(response.json()["observation"], dtype=np.float32), {}
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

    def step(self, action):
        response = requests.post(f"{self.apiurl}/step/{self.uuid}", json={"uuid": self.uuid, "action": int(action)})
        if response.status_code == 200:
            data = response.json()
            return (
                np.array(data["observation"], dtype=np.float32),
                data["reward"],
                data["done"],
                data["truncated"],
                data["info"]
            )
        elif response.status_code == 400:
            raise ValueError(f"Bad request: {response.json().get('error', 'Unknown error')}")
        elif response.status_code == 500:
            raise Exception("Internal Server Error")


