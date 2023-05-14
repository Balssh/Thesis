import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import HYPER_PARAMS_DINO as HYPER_PARAMS
from dino import Dino
from networks import PolicyConv as Policy

HYPER_PARAMS["BATCH_SIZE"] = HYPER_PARAMS["ENV_TIMESTEPS"] * HYPER_PARAMS["ENV_NUM"]
HYPER_PARAMS["MINIBATCH_SIZE"] = (
    HYPER_PARAMS["BATCH_SIZE"] // HYPER_PARAMS["MINIBATCH_NUM"]
)


def make_env(gym_env):
    """Helper function for making multiple environments"""

    def thunk():
        env = gym_env
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.FrameStack(env, 4)

        return env

    return thunk


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
        [make_env(Dino()) for _ in range(HYPER_PARAMS["ENV_NUM"])]
    )

    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "Only discrete action spaces supported!"

    policy = Policy(envs).to(device)
    optimizer = optim.Adam(
        policy.parameters(), lr=HYPER_PARAMS["LEARNING_RATE"], eps=1e-5
    )

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
    num_updates = HYPER_PARAMS["TIMESTEPS"] // HYPER_PARAMS["BATCH_SIZE"]

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
                action, log_prob, _, value = policy.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            log_probs[step] = log_prob
            next_obs, reward, next_terminal, _, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_terminal = torch.Tensor(next_terminal).to(device)

            # Log episode returns and lengths
            if "final_info" in info.keys():
                mean_episodic_return = 0
                episodes_ended = 0
                for item in info["final_info"]:
                    if item is not None:
                        mean_episodic_return += item["episode"]["r"]
                        episodes_ended += 1
                        writer.add_scalar(
                            "charts/episodic_return", item["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", item["episode"]["l"], global_step
                        )
                print(
                    f"Global step: {global_step}, mean episodic return: {mean_episodic_return / episodes_ended}; episodes ended: {episodes_ended}"
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
                    # Calculate delta
                    delta = (
                        rewards[t]
                        + HYPER_PARAMS["GAE_GAMMA"] * next_nonterminal * next_return
                        - values[t]
                    )
                    # Calculate advantage
                    advantages[t] = last_advantage = (
                        delta
                        + HYPER_PARAMS["GAE_GAMMA"]
                        * HYPER_PARAMS["GAE_LAMBDA"]
                        * next_nonterminal
                        * last_advantage
                    )
                returns = advantages + values
            else:
                # Calculate returns using discounted rewards
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

        # Batch data
        batch_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        batch_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        batch_log_probs = log_probs.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_values = values.reshape(-1)
        batch_advantages = advantages.reshape(-1)

        # Optimize policy
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

                # Get values and log probs
                _, new_log_probs, entropy, new_value = policy.get_action_and_value(
                    batch_obs[minibatch_ind], batch_actions.long()[minibatch_ind]
                )

                # Calculate ratios between old and new policy
                log_ratio = new_log_probs - batch_log_probs[minibatch_ind]
                ratio = torch.exp(log_ratio)

                # Calculate KL divergence
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

                # Calculate surrogate loss
                policy_loss1 = -ratio * minibatch_advantages
                policy_loss2 = -minibatch_advantages * torch.clamp(
                    ratio,
                    1.0 - HYPER_PARAMS["CLIPPING_COEFFICIENT"],
                    1.0 + HYPER_PARAMS["CLIPPING_COEFFICIENT"],
                )
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                # Calculate value loss
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

                # Calculate entropy loss
                entropy_loss = entropy.mean()

                # Calculate total loss
                loss = (
                    policy_loss
                    - HYPER_PARAMS["ENTROPY_COEFFICIENT"] * entropy_loss
                    + HYPER_PARAMS["VALUE_LOSS_COEFFICIENT"] * value_loss
                )

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

    torch.save(
        policy.state_dict(),
        f"models/{HYPER_PARAMS['ENV_ID']}_{HYPER_PARAMS['EXPERIMENT_NAME']}_{int(time.time())}.pt",
    )
    envs.close()
    writer.close()
