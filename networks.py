import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    """Helper function for initializing layers"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)

    return layer


class PolicySeparate(nn.Module):
    """Policy with 2 separate neural networks for actor and critic"""

    def __init__(self, envs):
        super(PolicySeparate, self).__init__()

        self.critic = nn.Sequential(
            init_layer(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            init_layer(nn.Linear(64, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            init_layer(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            init_layer(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, obs):
        """Get value from critic"""
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None):
        """Get action and value from actor and critic"""
        action_logits = self.actor(obs)
        action_dist = Categorical(logits=action_logits)
        if action is None:
            action = action_dist.sample()
        return (
            action,
            action_dist.log_prob(action),
            action_dist.entropy(),
            self.get_value(obs),
        )


class PolicyShared(nn.Module):
    def __init__(self, env):
        super(PolicyShared, self).__init__()

        # Shared layers between the policy and value networks
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

        # Separate heads
        self.value_head = nn.Linear(64, 1)
        self.policy_head = nn.Linear(64, env.single_action_space.n)

    def get_value(self, observation):
        # Returns the value of an observation/state
        hidden = self.network(observation)
        value = self.value_head(hidden)
        return value

    def get_action_and_value(self, observation, action=None):
        # Returns the action which should be taken based on current observation/state
        hidden = self.network(observation)
        logits = self.policy_head(hidden)
        probabilities = Categorical(logits=logits)
        if action is None:
            action = probabilities.sample()
        value = self.value_head(hidden)
        return action, probabilities.log_prob(action), probabilities.entropy(), value


class PolicyConv(nn.Module):
    def __init__(self, envs):
        super(PolicyConv, self).__init__()
        self.network = nn.Sequential(
            init_layer(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            init_layer(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_layer(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            init_layer(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = init_layer(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = init_layer(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class PolicyContinuous(nn.Module):
    """Policy with 2 different neural networks"""

    def __init__(self, envs):
        super(PolicyContinuous, self).__init__()

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
            init_layer(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            init_layer(
                nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01
            ),
        )

        self.policy_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def get_value(self, observation):
        return self.value_net(observation)

    def get_action_and_value(self, observation, action=None):
        action_mean = self.policy_net(observation)
        action_logstd = self.policy_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.value_net(observation),
        )
