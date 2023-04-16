import time
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from config import HYPER_PARAMS


def make_env(gym_id, seed):
    """Helper function for making multiple environments"""

    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
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
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    # Create policy
    policy = Policy(envs).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=HYPER_PARAMS["LEARNING_RATE"])

    # Create buffers
    observations = torch.zeros(
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
    num_updates = HYPER_PARAMS["TIMESTEPS"] // HYPER_PARAMS["BATCH_SIZE"]

    # Main loop
    for update in range(1, num_updates + 1):
        # Anneal learning rate
        if HYPER_PARAMS["ANNEAL_LR"]:
            lr = HYPER_PARAMS["LEARNING_RATE"] * (1 - (update - 1) / num_updates)
            optimizer.param_groups[0]["lr"] = lr

        # Collect data
        for step in range(0, HYPER_PARAMS["ENV_TIMESTEPS"]):
            global_step += 1 * HYPER_PARAMS["ENV_NUM"]
            observations[step] = next_obs
            terminateds[step] = next_terminal

            # Sample action
            with torch.no_grad():
                action, log_prob, _, value = policy.get_action_and_value(next_obs)
            actions[step] = action
            log_probs[step] = log_prob

            # Step environment
            next_obs, reward, next_terminal, _, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_terminal = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_terminal
            ).to(device)

            # Log data
            if "final_info" in info.keys():
                for j, r in enumerate(info["final_info"]):
                    if r is not None:
                        writer.add_scalar(
                            "charts/episodic_return", r["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", r["episode"]["l"], global_step
                        )

        # Compute advantages
        with torch.no_grad():
            next_value = policy.get_value(next_obs).reshape(1, -1)

            if HYPER_PARAMS["USE_GAE"]:
                # Generalized Advantage Estimation
                advantages = torch.zeros_like(rewards).to(device)
                last_advantage = 0
                for t in reversed(range(HYPER_PARAMS["ENV_TIMESTEPS"])):
                    if t == HYPER_PARAMS["ENV_TIMESTEPS"] - 1:
                        next_nonterminal = 1.0 - next_terminal
                        next_return = next_value
                    else:
                        next_nonterminal = 1.0 - terminateds[t + 1]
                        next_return = values[t + 1]
                    delta = (
                        rewards[t]
                        + HYPER_PARAMS["GAE_GAMMA"] * next_return * next_nonterminal
                        - values[t]
                    )
                    advantages[t] = last_advantage = (
                        delta
                        + HYPER_PARAMS["GAE_LAMBDA"]
                        * HYPER_PARAMS["GAE_GAMMA"]
                        * next_nonterminal
                        * last_advantage
                    )
                returns = advantages + values
            else:
                # Monte Carlo
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(HYPER_PARAMS["ENV_TIMESTEPS"])):
                    if t == HYPER_PARAMS["ENV_TIMESTEPS"] - 1:
                        next_nonterminal = 1.0 - next_terminal
                        next_return = next_value
                    else:
                        next_nonterminal = 1.0 - terminateds[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = (
                        rewards[t]
                        + HYPER_PARAMS["GAE_GAMMA"] * next_nonterminal * next_return
                    )
                advantages = returns - values

        # Flatten batch data
        batch_obs = observations.reshape((-1,) + envs.single_observation_space.shape)
        batch_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        batch_log_probs = log_probs.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_values = values.reshape(-1)
        batch_advantages = advantages.reshape(-1)

        # Optimizing the policy
        batch_ind = np.arange(HYPER_PARAMS["BATCH_SIZE"])
        clipped_fracs = []
        for epoch in range(HYPER_PARAMS["UPDATE_EPOCHS"]):
            # Shuffle batch
            np.random.shuffle(batch_ind)

            # Iterate over batches
            for start in range(
                0, HYPER_PARAMS["BATCH_SIZE"], HYPER_PARAMS["MINIBATCH_SIZE"]
            ):
                # Get minibatch indices
                end = start + HYPER_PARAMS["MINIBATCH_SIZE"]
                minibatch_ind = batch_ind[start:end]

                # Get value and log probs
                _, new_log_probs, entropy, new_value = policy.get_action_and_value(
                    batch_obs[minibatch_ind], batch_actions[minibatch_ind]
                )

                # Compute ratios
                log_ratio = new_log_probs - batch_log_probs[minibatch_ind]
                ratio = torch.exp(log_ratio)

                # Compute KL divergence
                with torch.no_grad():
                    old_kl = (-log_probs).mean()
                    new_kl = ((ratio - 1) - log_ratio).mean()
                    clipped_fracs += [
                        ((ratio - 1).abs() > HYPER_PARAMS["CLIPPING_COEFFICIENT"])
                        .float()
                        .mean()
                        .item()
                    ]

                # Normalize advantages
                minibatch_advantages = batch_advantages[minibatch_ind]
                if HYPER_PARAMS["NORMALIZE_ADVANTAGE"]:
                    minibatch_advantages = (
                        minibatch_advantages - minibatch_advantages.mean()
                    ) / (minibatch_advantages.std() + 1e-8)

                # Compute surrogate loss
                policy_loss1 = -minibatch_advantages * ratio
                policy_loss2 = -minibatch_advantages * torch.clamp(
                    ratio,
                    1 - HYPER_PARAMS["CLIPPING_COEFFICIENT"],
                    1 + HYPER_PARAMS["CLIPPING_COEFFICIENT"],
                )
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                # Compute value loss
                new_value = new_value.view(-1)
                if HYPER_PARAMS["CLIP_VALUELOSS"]:
                    unclipped_value_loss = (
                        new_value - batch_returns[minibatch_ind]
                    ) ** 2
                    clipped_value = batch_values[minibatch_ind] + torch.clamp(
                        new_value - batch_values[minibatch_ind],
                        -HYPER_PARAMS["CLIPPING_COEFFICIENT"],
                        HYPER_PARAMS["CLIPPING_COEFFICIENT"],
                    )
                    clipped_value_loss = (
                        clipped_value - batch_returns[minibatch_ind]
                    ) ** 2
                    max_value_loss = torch.max(unclipped_value_loss, clipped_value_loss)
                    # NOT SURE IF 0.5 IS CORRECT
                    value_loss = 0.5 * max_value_loss.mean()
                else:
                    # NOT SURE IF 0.5 IS CORRECT
                    value_loss = (
                        0.5 * ((new_value - batch_returns[minibatch_ind]) ** 2).mean()
                    )

                # Compute entropy loss
                entropy_loss = entropy.mean()

                # Compute total loss
                loss = (
                    policy_loss
                    - HYPER_PARAMS["ENTROPY_COEFFICIENT"] * entropy_loss
                    + HYPER_PARAMS["VALUE_LOSS_COEFFICIENT"] * value_loss
                )

                # Update policy
                optimizer.zero_grad()
                loss.backward()
                if HYPER_PARAMS["NORMALIZE_GRADIENTS"]:
                    nn.utils.clip_grad_norm_(
                        policy.parameters(), HYPER_PARAMS["MAX_GRADIENT_NORM"]
                    )
                optimizer.step()

                # Early stopping
                if (
                    HYPER_PARAMS["TARGET_KL"] is not None
                    and new_kl > HYPER_PARAMS["TARGET_KL"]
                ):
                    break
        # Calculate explained variance
        y_predict, y_true = batch_values.cpu().numpy(), batch_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_predict) / var_y

        # Log results
        # Debug variables are:
        # - policy_loss: the mean policy loss across the minibatches
        # - value_loss: the mean value loss across the minibatches
        # - entropy_loss: the mean entropy loss across the minibatches
        # - clipped_frac: the fraction of the minibatches where the policy loss was clipped
        # - new_kl: the KL divergence between the old and new policy
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy_loss", entropy_loss.item(), global_step)
        writer.add_scalar("losses/new_KL", new_kl.item(), global_step)
        writer.add_scalar("losses/old_KL", old_kl.item(), global_step)
        writer.add_scalar(
            "losses/clippedfractions", np.mean(clipped_fracs), global_step
        )
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

    envs.close()
    writer.close()
