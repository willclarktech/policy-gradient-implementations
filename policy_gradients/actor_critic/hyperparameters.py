from core import Hyperparameters

default_hyperparameters = Hyperparameters(
    env_name="LunarLander-v2",
    n_episodes=2000,
    log_period=1,
    hidden_features=[2048, 1536],
    alpha=5e-6,
    gamma=0.99,
)
