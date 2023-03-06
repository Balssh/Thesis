import time
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# Global variables
HYPER_PARAMS = {
    "ENV_ID": "HalfCheetah-v2",
    "EXPERIMENT_NAME": "homemade_ppo_disc_separate_nn",
    "SEED": 1,
    "TORCH_DETERMINISTIC": True,
    "DEVICE": "cpu",
    "LEARNING_RATE": 3e-04,
    "ENV_NUM": 1,
    "ENV_TIMESTEPS": 2048,
    "TIMESTEPS": 25000,
    "ANNEAL_LR": True,
    "USE_GAE": True,
    "MINIBATCH_NUM": 32,
    "GAE_GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "UPDATE_EPOCHS": 10,
    "NORMALIZE_ADVANTAGE": True,
    "CLIP_VALUELOSS": True,
    "NORMALIZE_GRADIENTS": True,
    "CLIPPING_COEFFICIENT": 0.2,
    "ENTROPY_COEFFICIENT": 0.01,
    "VALUE_LOSS_COEFFICIENT": 0.5,
    "MAX_GRADIENT_NORM": 0.5,
    "TARGET_KL": 0.015,
}

HYPER_PARAMS["BATCH_SIZE"] = HYPER_PARAMS["ENV_TIMESTEPS"] * HYPER_PARAMS["ENV_NUM"]
HYPER_PARAMS["MINIBATCH_SIZE"] = (
    HYPER_PARAMS["BATCH_SIZE"] // HYPER_PARAMS["MINIBATCH_NUM"]
)


def make_env(gym_id, seed):
    """Helper function for making multiple environments"""

    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeAction(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    """Helper function for initializing layers"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)

    return layer


class Policy(nn.Module):
    """Policy with 2 different neural networks"""

    def __init__(self, envs):
        super(Policy, self).__init__()

        self.value_net = nn.Sequential(
            init_layer(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            init_layer(nn.Linear(64, 1), std=1),
        )

        self.policy_net = nn.Sequential(
            init_layer(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            init_layer(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

        self.policy_logstd = nn.Parameter(
            torch.zeros(1, np.prod(env.single_action_space.shape))
        )
        self.value = self.value_net(observation)
        self.probs = Normal()
