import os
import argparse
import time
import random
import torch
import numpy as np
import gymnasium as gym
from distutils.util import strtobool
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
    return parser.parse_args()


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

    def make_gym(gym_id):
        def thunk():
            env = gym.make(gym_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env
        return thunk

    envs = gym.vector.SyncVectorEnv([make_gym(args.env_id)])
    observation = envs.reset()
    for i in range(200):
        action = envs.action_space.sample()
        observation, reward, terminated, truncated, info = envs.step(action)

        if "final_info" in info.keys():
            print(f"{i} Episodic return: {info['final_info'][0]['episode']['r']}")

    envs.close()