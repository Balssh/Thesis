import time
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from dino import Dino
from homemade_ppo_conv import Policy


def make_env(gym_env):
    """Helper function for making multiple environments"""

    def thunk():
        env = gym_env
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


envs = gym.vector.SyncVectorEnv([make_env(Dino())])
model = Policy(envs).to("cpu")
optimizer = optim.Adam(model.parameters(), lr=2.5e-04, eps=1e-5)

checkpoint = torch.load("models/DinoChrome_homemade_ppo_conv_1680193947.pt")
model.load_state_dict(checkpoint)
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# epoch = checkpoint["epoch"]
# loss = checkpoint["loss"]
model.eval()

for episode in range(5):
    obs = torch.Tensor(envs.reset()[0]).to("cpu")
    done = False
    total_reward = 0
    while not done:
        (
            action,
            _,
            _,
            _,
        ) = model.get_action_and_value(obs)
        obs, reward, done, _, info = envs.step(action.cpu().numpy())
        obs = torch.Tensor(obs).to("cpu")
        time.sleep(0.01)
        total_reward += reward
    print("Total Reward for episode {} is {}".format(episode, total_reward))
    time.sleep(2)
