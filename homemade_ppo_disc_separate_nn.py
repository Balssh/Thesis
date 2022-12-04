import time
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# Global variables
HYPER_PARAMS = {
    "ENV_ID": "CartPole-v1",
    "EXPERIMENT_NAME": "homemade_ppo_disc_separate_nn",
    "SEED": 1,
    "TORCH_DETERMINISTIC": True,
    "DEVICE": "cuda",
    "LEARNING_RATE": 2.5e-04,
    "ENV_NUM": 4,
    "ENV_TIMESTEPS": 128,
    "TIMESTEPS": 25000,
    "ANNEAL_LR": True,
    "USE_GAE": True,
    "MINIBATCH_NUM": 4,
    "GAE_GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "UPDATE_EPOCHS": 4,
    "NORMALIZE_ADVANTAGE": True,
    "CLIP_VALUELOSS": True,
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
    """Policy with 2 separate neural networks for actor and critic"""

    def __init__(self, envs):
        super(Policy, self).__init__()

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


if __name__ == "__main__":
    """Main entrypoint of application"""
    run_name = f"{HYPER_PARAMS['ENV_ID']}_{HYPER_PARAMS['EXPERIMENT_NAME']}_{HYPER_PARAMS['SEED']}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n %s"
        % ("\n".join([f"|{key}|{value}|" for key, value in HYPER_PARAMS.items()])),
    )

    # Set seed
    random.seed(HYPER_PARAMS["SEED"])
    np.random.seed(HYPER_PARAMS["SEED"])
    torch.manual_seed(HYPER_PARAMS["SEED"])
    if HYPER_PARAMS["TORCH_DETERMINISTIC"]:
        torch.backends.cudnn.deterministic = True

    device = torch.device(HYPER_PARAMS["DEVICE"])

    # Create environments
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(HYPER_PARAMS["ENV_ID"], HYPER_PARAMS["SEED"] + i)
            for i in range(HYPER_PARAMS["ENV_NUM"])
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "Only discrete action spaces supported!"
    print("SUCCESSFULLY CREATED ENVIRONMENTS")

    # Create policy
    policy = Policy(envs).to(device)
    optimizer = optim.Adam(
        policy.parameters(), lr=HYPER_PARAMS["LEARNING_RATE"], eps=1e-5
    )
    print(policy)
    print("SUCCESSFULLY CREATED POLICY")

    # Create buffers
    obs = torch.zeros(
        (HYPER_PARAMS["ENV_TIMESTEPS"], HYPER_PARAMS["ENV_NUM"])
        + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (HYPER_PARAMS["ENV_TIMESTEPS"], HYPER_PARAMS["ENV_NUM"])
        + envs.single_action_space.shape
    ).to(device)
    log_probs = torch.zeros(
        (HYPER_PARAMS["ENV_TIMESTEPS"], HYPER_PARAMS["ENV_NUM"])
    ).to(device)
    rewards = torch.zeros((HYPER_PARAMS["ENV_TIMESTEPS"], HYPER_PARAMS["ENV_NUM"])).to(
        device
    )
    terminateds = torch.zeros(
        (HYPER_PARAMS["ENV_TIMESTEPS"], HYPER_PARAMS["ENV_NUM"])
    ).to(device)
    values = torch.zeros((HYPER_PARAMS["ENV_TIMESTEPS"], HYPER_PARAMS["ENV_NUM"])).to(
        device
    )

    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_terminal = torch.zeros(HYPER_PARAMS["ENV_NUM"]).to(device)
    num_updates = HYPER_PARAMS["TOTAL_TIMESTEPS"] // HYPER_PARAMS["BATCH_SIZE"]

    # Begin the training loop
    for update in range(1, num_updates + 1):
        # Anneal learning rate
        if HYPER_PARAMS["ANNEAL_LR"]:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = HYPER_PARAMS["LEARNING_RATE"] * frac
            optimizer.param_groups[0]["lr"] = lr_now

        # Collect trajectories
        for step in range(0, HYPER_PARAMS["ENV_TIMESTEPS"]):
            global_step += 1 * HYPER_PARAMS["ENV_NUM"]
            obs[step] = next_obs
            terminateds[step] = next_terminal
            with torch.no_grad():
                actions[step], log_probs[step], _, values = policy.get_action_and_value(
                    obs[step]
                )
                values[step] = values.flatten()
            # Step environments
            next_obs, rewards[step], next_terminal, _, info = envs.step(
                actions[step].cpu().numpy()
            )
            next_obs = torch.Tensor(next_obs).to(device)
            next_terminal = torch.Tensor(next_terminal).to(device)

            # Log rewards
            if "final_info" in info.keys():
                for j, r in enumerate(info["final_info"]):
                    if r is not None:
                        writer.add_scalar(
                            "charts/episodic_return", r["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", r["episode"]["l"], global_step
                        )
