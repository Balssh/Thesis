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
    "EXPERIMENT_NAME": "homemade_ppo",
    "SEED": 1,
    "TORCH_DETERMINISTIC": True,
    "DEVICE": "cuda",
    "LEARNING_RATE": 2.5e-04,
    "ENV_TIMESTEPS": 128,
    "TIMESTEPS": 5000,
    "MINIBATCH_NUM": 4,
    "GAE_GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "UPDATE_EPOCHS": 4,
    "CLIPPING_COEFFICIENT": 0.2,
    "ENTROPY_COEFICIENT": 0.5,
    "VALUE_LOSS_COEFFICIENT": 0.5,
    "MAX_GRADIENT_NORM": 0.5,
}


class Agent(nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()

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


def make_env(gym_id, seed):
    # Helper function
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


if __name__ == "__main__":
    # Basic setup for recording the run
    run_name = f"{HYPER_PARAMS['ENV_ID']}_{HYPER_PARAMS['EXPERIMENT_NAME']}_{HYPER_PARAMS['SEED']}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n %s"
        % ("\n".join([f"|{key}|{value}|" for key, value in HYPER_PARAMS.items()])),
    )

    # Making the environment, using SyncVectorEnv because it's easier to manage the terminated flags
    env = gym.vector.SyncVectorEnv(
        [make_env(HYPER_PARAMS["ENV_ID"], HYPER_PARAMS["SEED"])]
    )
    assert isinstance(
        env.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    # Seeding the random number generators
    random.seed(HYPER_PARAMS["SEED"])
    np.random.seed(HYPER_PARAMS["SEED"])
    torch.manual_seed(HYPER_PARAMS["SEED"])

    # Initialization
    device = torch.device(HYPER_PARAMS["DEVICE"])

    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters())

    observations = torch.zeros(
        (HYPER_PARAMS["TIMESTEPS"],) + env.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (HYPER_PARAMS["TIMESTEPS"],) + env.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((HYPER_PARAMS["TIMESTEPS"],)).to(device)
    rewards = torch.zeros((HYPER_PARAMS["TIMESTEPS"],)).to(device)
    terminateds = torch.zeros((HYPER_PARAMS["TIMESTEPS"],)).to(device)
    values = torch.zeros((HYPER_PARAMS["TIMESTEPS"],)).to(device)

    updates = HYPER_PARAMS["TIMESTEPS"] // HYPER_PARAMS["ENV_TIMESTEPS"]
    minibatch_size = HYPER_PARAMS["ENV_TIMESTEPS"] // HYPER_PARAMS["MINIBATCH_NUM"]
    obsertation = torch.Tensor(env.reset()[0]).to(device)
    terminated = torch.zeros(1).to(device)
    global_step = 0

    # Training loop
    start_time = time.time()
    for update in range(1, updates + 1):
        # Better results with learning rate annealing
        fraction = 1.0 - (update - 1.0) / updates
        new_learning_rate = fraction * HYPER_PARAMS["LEARNING_RATE"]
        optimizer.param_groups[0]["lr"] = new_learning_rate

        # Rollout phase (collecting data about the env, following the last policy)
        for step in range(0, HYPER_PARAMS["ENV_TIMESTEPS"]):
            global_step += 1
            observations[step] = obsertation
            terminateds[step] = terminated

            # Get action
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    observation=obsertation
                )
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # Do a step with the selected action
            observation, reward, terminated, _, info = env.step(action.cpu().numpy())

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            obsertation = torch.Tensor(observation).to(device)
            terminated = torch.Tensor(terminated).to(device)

            # Log the reward and episodic length (below code works for multiple envs)
            if "final_info" in info.keys():
                for j, r in enumerate(info["final_info"]):
                    if r is not None:
                        writer.add_scalar(
                            "charts/episodic_return", r["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", r["episode"]["l"], global_step
                        )
                        break

        # Learning phase
        with torch.no_grad():
            # Getting the GAEs
            next_value = agent.get_value(observation=obsertation)
            advantages = torch.zeros_like(rewards).to(device)
            last_advantage = 0
            for t in reversed(range(HYPER_PARAMS["ENV_TIMESTEPS"])):
                delta = (
                    rewards[t]
                    + HYPER_PARAMS["GAE_GAMMA"] * next_value * (1 - terminateds[t])
                    - values[t]
                )
                advantages[t] = last_advantage = (
                    delta
                    + HYPER_PARAMS["GAE_GAMMA"]
                    * HYPER_PARAMS["GAE_LAMBDA"]
                    * (1 - terminateds[t])
                    * last_advantage
                )

            returns = advantages + values

        # Flatten the batch (not needed because only 1 env present?)
        batch_observation = observations.reshape((-1,) + env.observation_space.shape)
        batch_logprobs = logprobs.reshape(-1)
        batch_actions = actions.reshape((-1,) + env.action_space.shape)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_values = values.reshape(-1)

        # Optimizing the policy
        batch_indices = np.arange(HYPER_PARAMS["ENV_TIMESTEPS"])
        for epoch in range(HYPER_PARAMS["UPDATE_EPOCHS"]):
            np.random.shuffle(batch_indices)
            # Optimize the loss for K epochs on minibatches
            for start in range(0, HYPER_PARAMS["ENV_TIMESTEPS"], minibatch_size):
                end = start + minibatch_size
                minibatch_indices = batch_indices[start:end]

                # Get new logprob for the policy
                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    batch_observation[minibatch_indices],
                    batch_actions.long()[minibatch_indices],
                )

                log_ratio = new_logprob - batch_logprobs[minibatch_indices]
                ratio = log_ratio.exp()

                # Calculate KL divergence
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()

                minibatch_advantages = batch_advantages[minibatch_indices]
                # Normalize the advantages
                minibatch_advantages = (
                    minibatch_advantages - minibatch_advantages.mean()
                ) / (minibatch_advantages.std() + 1e-08)
                # Calculate the policy loss, value loss and entropy loss
                policy_loss1 = minibatch_advantages * ratio
                policy_loss2 = minibatch_advantages * torch.clamp(
                    ratio,
                    1 - HYPER_PARAMS["CLIPPING_COEFFICIENT"],
                    1 + HYPER_PARAMS["CLIPPING_COEFFICIENT"],
                )
                policy_loss = torch.min(policy_loss1, policy_loss2).mean()
                # Clip the value loss
                unclipped_value_loss = (
                    new_value - batch_returns[minibatch_indices]
                ) ** 2
                clipped_value = batch_values[minibatch_indices] + torch.clamp(
                    new_value,
                    -HYPER_PARAMS["CLIPPING_COEFFICIENT"],
                    HYPER_PARAMS["CLIPPING_COEFFICIENT"],
                )
                clipped_value_loss = (new_value - batch_returns[minibatch_indices]) ** 2
                value_loss = torch.max(unclipped_value_loss, clipped_value_loss).mean()
                entropy_loss = entropy.mean()

                # Calculate the final loss function
                loss = (
                    policy_loss
                    + HYPER_PARAMS["VALUE_LOSS_COEFFICIENT"] * value_loss
                    - HYPER_PARAMS["ENTROPY_COEFICIENT"] * entropy_loss
                )

                # Do a gradient descent
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), HYPER_PARAMS["MAX_GRADIENT_NORM"])
                optimizer.step()

        # Log some stats
        y_pred, y_true = batch_values.cpu().numpy(), batch_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
