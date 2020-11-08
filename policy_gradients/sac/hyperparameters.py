from core import Hyperparameters

default_hyperparameters = Hyperparameters(
    env_name="InvertedPendulumBulletEnv-v0",
    n_episodes=250,
    log_period=1,
    # hidden_features=[400, 300],
    # alpha=1e-4,
    gamma=0.99,
    tau=5e-3,
    epsilon=1e-6,
    batch_size=256,
    replay_buffer_capacity=1_000_000,
    reward_scale=2.0,
)