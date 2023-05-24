import torch

# Config for DinoChrome
# TODO: Change the parameters to be for DinoChrome
HYPER_PARAMS_DINO = {
    "ENV_ID": "DinoChrome",
    "EXPERIMENT_NAME": "homemade_ppo_conv",
    "SEED": 1,
    "TORCH_DETERMINISTIC": True,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "LEARNING_RATE": 2.5e-04,
    "ENV_NUM": 1,
    "ENV_TIMESTEPS": 128,
    "TIMESTEPS": 1000000,
    "ANNEAL_LR": True,
    "USE_GAE": True,
    "MINIBATCH_NUM": 4,
    "GAE_GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "UPDATE_EPOCHS": 4,
    "NORMALIZE_ADVANTAGE": True,
    "CLIP_VALUELOSS": True,
    "NORMALIZE_GRADIENTS": True,
    "CLIPPING_COEFFICIENT": 0.2,
    "ENTROPY_COEFFICIENT": 0.01,
    "VALUE_LOSS_COEFFICIENT": 0.5,
    "MAX_GRADIENT_NORM": 0.5,
    "TARGET_KL": 0.015,
}


# Config for ClassicControl
# TODO: Change the parameters to be for ClassicControl
HYPER_PARAMS_CLASSIC = {
    "ENV_ID": "CartPole-v1",
    "EXPERIMENT_NAME": "homemade_ppo_separate_nn",  # TODO: remember to change names of shared/separate nns
    "SEED": 1,
    "TORCH_DETERMINISTIC": True,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "LEARNING_RATE": 2.5e-04,
    "ENV_NUM": 4,
    "ENV_TIMESTEPS": 128,
    "TIMESTEPS": 400000,  # 400k steps to validate the results are comparable to original implementation
    "ANNEAL_LR": True,
    "USE_GAE": True,
    "MINIBATCH_NUM": 4,
    "GAE_GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "UPDATE_EPOCHS": 4,
    "NORMALIZE_ADVANTAGE": True,
    "CLIP_VALUELOSS": True,
    "NORMALIZE_GRADIENTS": True,
    "CLIPPING_COEFFICIENT": 0.2,
    "ENTROPY_COEFFICIENT": 0.01,
    "VALUE_LOSS_COEFFICIENT": 0.5,
    "MAX_GRADIENT_NORM": 0.5,
    "TARGET_KL": 0.015,
}

# Config for MuJoCo
# TODO: Change the parameters to be for MuJoCo
HYPER_PARAMS_MUJOCO = {
    "ENV_ID": "DinoChrome",
    "EXPERIMENT_NAME": "homemade_ppo_conv",
    "SEED": 1,
    "TORCH_DETERMINISTIC": True,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "LEARNING_RATE": 2.5e-04,
    "ENV_NUM": 1,
    "ENV_TIMESTEPS": 128,
    "TIMESTEPS": 500000,
    "ANNEAL_LR": True,
    "USE_GAE": True,
    "MINIBATCH_NUM": 4,
    "GAE_GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "UPDATE_EPOCHS": 4,
    "NORMALIZE_ADVANTAGE": True,
    "CLIP_VALUELOSS": True,
    "NORMALIZE_GRADIENTS": True,
    "CLIPPING_COEFFICIENT": 0.2,
    "ENTROPY_COEFFICIENT": 0.01,
    "VALUE_LOSS_COEFFICIENT": 0.5,
    "MAX_GRADIENT_NORM": 0.5,
    "TARGET_KL": 0.015,
}
