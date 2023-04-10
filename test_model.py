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
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk


envs = gym.vector.SyncVectorEnv([make_env(Dino())])
model = Policy(envs).to("cuda")
optimizer = optim.Adam(model.parameters(), lr=2.5e-04, eps=1e-5)

checkpoint = torch.load("models/GOOD_MODEL_100k.pt")
model.load_state_dict(checkpoint)
model.eval()
rewards_rand = np.zeros(10)
rewards_100k = np.zeros(10)
rewards_400k = np.zeros(10)

print(f"Random model performance:")
for episode in range(10):
    obs = torch.Tensor(envs.reset()[0]).to("cuda")
    done = False
    total_reward = 0
    while not done:
        obs, reward, done, _, info = envs.step(envs.action_space.sample())
        obs = torch.Tensor(obs).to("cuda")
        total_reward += reward
    print("Total Reward for episode {} is {}".format(episode, total_reward))
    rewards_rand[episode] = total_reward

print(f"Trained 100k model performance:")
for episode in range(10):
    obs = torch.Tensor(envs.reset()[0]).to("cuda")
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
        obs = torch.Tensor(obs).to("cuda")
        total_reward += reward
    print("Total Reward for episode {} is {}".format(episode, total_reward))
    rewards_100k[episode] = total_reward

print(f"Trained 400k model performance:")
checkpoint = torch.load("models/GOOD_MODEL_400k.pt")
model.load_state_dict(checkpoint)
model.eval()

for episode in range(10):
    obs = torch.Tensor(envs.reset()[0]).to("cuda")
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
        obs = torch.Tensor(obs).to("cuda")
        total_reward += reward
    print("Total Reward for episode {} is {}".format(episode, total_reward))
    rewards_400k[episode] = total_reward

print(f"Random model average reward: {np.mean(rewards_rand)}")
print(f"Trained 100k model average reward: {np.mean(rewards_100k)}")
print(f"Trained 400k model average reward: {np.mean(rewards_400k)}")
