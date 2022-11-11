import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def parse_arguments():
    """Parsing the command line arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="The name of the current experiment",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="CartPole-v1",
        help="Name of the gym environment",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-05,
        help="Learning rate of the optimizer",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the random number generation"
    )
    parser.add_argument(
        "--timesteps", type=int, default=5000, help="Total timesteps of the experiment"
    )
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="(Un)Set torch.backends.cudnn.deterministic",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="The device to be used: cpu or cuda(gpu)",
    )
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggle tracking results in cloud via Weights&Biases",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default="PPO",
        help="Name of the Weights&Biases project",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Name of the Weights&Biases entity (person or team)",
    )
    parser.add_argument(
        "--env_num",
        type=int,
        default=4,
        help="The number of environments to be run in paralell",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=64,
        help="The number of steps in an episode",
    )
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="(Un)Set if the learning rate will be annealed",
    )
    parser.add_argument(
        "--gae",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="If true we will use generalized advantage estimation",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="The gamma discount factor used in GAE",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="The lambda factor used in GAE",
    )
    args = parser.parse_args()
    args.batch_size = args.env_num * args.num_steps

    return args


def make_env(gym_id, seed):
    """Helper function for making multiple environments"""

    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def init_layer(layer, std=np.sqrt(2), bias_const=0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)

    return layer


class Policy(nn.Module):
    def __init__(self, envs) -> None:
        super(Policy, self).__init__()

        self.critic = nn.Sequential(
            init_layer(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            init_layer(nn.Linear(64, 1), std=1.1),
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

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probabilities = Categorical(logits=logits)
        if action == None:
            action = probabilities.sample()
        return (
            action,
            probabilities.log_prob(action),
            probabilities.entropy(),
            self.critic(x),
        )


if __name__ == "__main__":
    """Main entrypoint of application"""
    args = parse_arguments()

    run_name = f"{args.env_id}_{args.experiment}_{args.seed}_{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(args.device)

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i) for i in range(args.env_num)]
    )
    # observation = envs.reset()
    # for i in range(200):
    #     action = envs.action_space.sample()
    #     observation, reward, terminated, truncated, info = envs.step(action)
    #     if "final_info" in info.keys():
    #         print(f"{i} Episodic return")
    #         for j, r in enumerate(info["final_info"]):
    #             if r is not None:
    #                 print(f"\tEnv[{j}] return: {r['episode']['r']}")
    # envs.close()

    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "Only discrete action spaces supported!"
    print(f"{args.env_id} single action_space: {envs.single_action_space}")
    print(f"{args.env_id} single observation_space: {envs.single_observation_space}")

    policy = Policy(envs).to(device)
    print(policy)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-05)

    observations = torch.zeros(
        (args.num_steps, args.env_num) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.env_num) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.env_num)).to(device)
    rewards = torch.zeros((args.num_steps, args.env_num)).to(device)
    terminateds = torch.zeros((args.num_steps, args.env_num)).to(device)
    truncateds = torch.zeros((args.num_steps, args.env_num)).to(device)
    values = torch.zeros((args.num_steps, args.env_num)).to(device)

    global_step = 0
    start_time = time.time()
    next_observations = torch.Tensor(envs.reset()[0]).to(device)
    next_terminateds = torch.zeros(args.env_num).to(device)
    num_updates = args.timesteps // args.batch_size

    # print(num_updates)
    # print(f"Next observations shape {next_observations.shape}")
    # print(f"Policy get_value(next_observations) {policy.get_value(next_observations)}")
    # print(f"Policy get_value(next_observations).shape {policy.get_value(next_observations).shape}")
    # print(f"Policy get_action_and_value(next_observations) {policy.get_action_and_value(next_observations)}")

    for update in range(num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            new_lr = frac * args.learning_rate
            optimizer.param_groups[0]['lr'] = new_lr

        for step in range(0, args.num_steps):
            global_step += 1 * args.env_num
            observations[step] = next_observations
            terminateds[step] = next_terminateds

            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(next_observations)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_observation, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_observation = torch.Tensor(next_observation).to(device)
            terminated = torch.Tensor(terminated).to(device)
            truncated = torch.Tensor(truncated).to(device)

            if "final_info" in info.keys():
                for j, r in enumerate(info["final_info"]):
                    if r is not None:
                        writer.add_scalar("charts/episodic_return", r['episode']['r'], global_step)
                        writer.add_scalar("charts/episodic_length", r['episode']['l'], global_step)
                        break

            with torch.no_grad():
                next_value = policy.get_value(next_observation).reshape(1, -1)
                if args.gae:
                    advantages = torch.zeros_like(rewards).to(device)
                    last_gae_lambda = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            next_nonterminal = 1 - terminated
                            next_values = next_value
                        else:
                            next_nonterminal = 1 - terminateds[t + 1]
                            next_values = values[t + 1]
                        delta = rewards[t] + args.gamma * next_values * next_nonterminal - values[t]
                        advantages[t] = last_gae_lambda = delta + args.gamma * args.gae_lambda * last_gae_lambda
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            next_nonterminal = 1 - terminated
                            next_return = next_value
                        else:
                            next_nonterminal = 1 - terminateds[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = rewards[t] + args.gamma * next_nonterminal * next_return
                    advantages = returns - values

            batch_observations = observations.reshape((-1, ) + envs.single_observation_space.shape)
            batch_actions = actions.reshape((-1, ) + envs.single_action_space.shape)
            batch_logprobs = logprobs.reshape(-1)
            batch_returns = returns.reshape(-1)
            batch_values = returns.reshape(-1)
            batch_advantages = advantages.reshape(-1)

